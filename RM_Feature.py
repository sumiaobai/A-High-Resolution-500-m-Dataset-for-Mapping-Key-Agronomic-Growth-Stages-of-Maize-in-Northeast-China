import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.ndimage import convolve1d
from osgeo import gdal
import os
import warnings
import logging

# Explicitly set GDAL exception handling
gdal.UseExceptions()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define SG smoothing coefficients
COEFFS_LONG_TREND = np.array([1 / 7] * 7)  # Example coefficients, adjust as needed
COEFFS_SHORT_TREND = np.array([1 / 3, 1 / 3, 1 / 3])  # Example coefficients, adjust as needed


def chen_sg_filter(curve_0, max_iteration=10):
    """
    Apply Chen's Savitzky-Golay filter for smoothing time series data.
    
    Args:
        curve_0: Original curve data
        max_iteration: Maximum number of iterations
        
    Returns:
        Smoothed curve
    """
    curve_tr = convolve1d(curve_0, COEFFS_LONG_TREND, mode="wrap")
    d = curve_tr - curve_0
    dmax = np.max(np.abs(d))
    w_func = np.frompyfunc(lambda d_i: min((1, 1 - d_i / dmax)), 1, 1)
    W = w_func(d)
    curve_k = np.copy(curve_tr)
    f_arr = np.zeros(max_iteration)
    curve_previous = None
    
    for i in range(max_iteration):
        curve_k = np.maximum(curve_k, curve_0)
        curve_k = convolve1d(curve_k, COEFFS_SHORT_TREND, mode="wrap")
        f_arr[i] = np.sum(np.abs(curve_k - curve_0) * W)
        if i >= 1 and f_arr[i] > f_arr[i - 1]:
            return curve_previous
        curve_previous = curve_k
    
    return curve_previous


def read_image_as_array(file_path):
    """
    Read image data and return as array.
    
    Args:
        file_path: Path to the input image file
        
    Returns:
        Tuple of (image_array, dataset)
    """
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise FileNotFoundError(f"File not found: {file_path}")
    return dataset.ReadAsArray(), dataset


def process_pixel(grvi_series, doy_series, emergence_day, doy_upper_limit, interp_num=10000):
    """
    Process a single pixel to find target DOY and create valid mask.
    
    Args:
        grvi_series: GRVI time series for the pixel
        doy_series: Day of year time series
        emergence_day: Emergence day threshold
        doy_upper_limit: Upper limit for DOY search
        interp_num: Number of interpolation points
        
    Returns:
        Tuple of (target_doy, valid_mask) or (None, None) if processing fails
    """
    # Remove invalid data points
    valid_mask = ~np.isnan(grvi_series) & ~np.isnan(doy_series)
    grvi_series_valid = grvi_series[valid_mask]
    doy_series_valid = doy_series[valid_mask]

    if len(grvi_series_valid) == 0 or len(doy_series_valid) == 0:
        return None, None

    # Apply SG filter to smooth GRVI curve
    grvi_series_valid = chen_sg_filter(grvi_series_valid)

    # Remove duplicate DOY values
    unique_doy, unique_indices = np.unique(doy_series_valid, return_index=True)
    grvi_series_valid = grvi_series_valid[unique_indices]

    try:
        # Interpolate GRVI curve
        f = interp1d(unique_doy, grvi_series_valid, kind='linear', fill_value='extrapolate')
        x_new = np.linspace(unique_doy.min(), min(unique_doy.max(), doy_upper_limit), num=interp_num)
        y_new = f(x_new)

        # Find zero crossings (where GRVI changes sign)
        zero_crossings = x_new[np.where(np.diff(np.sign(y_new)))[0]]
        valid_zero_crossings = zero_crossings[(zero_crossings > emergence_day) & (zero_crossings <= doy_upper_limit)]

        if valid_zero_crossings.size > 0:
            # Use root_scalar to find precise zero crossing point
            for crossing in valid_zero_crossings:
                try:
                    root_result = root_scalar(f, bracket=[crossing - 1, crossing + 1], method='brentq')
                    if root_result.converged:
                        target_doy = round(root_result.root)
                        return target_doy, valid_mask
                except ValueError:
                    continue
    except ValueError as e:
        logging.error(f"Interpolation error: {e}")

    return None, None


def process_block(reference_ts_block, start_row, end_row, cols, emergence_day, doy_upper_limit):
    """
    Process a block of image data.
    
    Args:
        reference_ts_block: Block of reference time series data
        start_row: Starting row index
        end_row: Ending row index
        cols: Number of columns
        emergence_day: Emergence day threshold
        doy_upper_limit: Upper limit for DOY search
        
    Returns:
        Tuple of (output_bands_block, valid_pixel_count)
    """
    rows = end_row - start_row
    output_bands_block = np.zeros((6, rows, cols), dtype=np.float32)
    valid_pixel_count = 0

    for row in range(rows):
        for col in range(cols):
            # Extract GRVI and DOY series for current pixel
            grvi_series = reference_ts_block[4::6, row, col]
            doy_series = reference_ts_block[5::6, row, col]

            target_doy, valid_mask = process_pixel(grvi_series, doy_series, emergence_day, doy_upper_limit)
            if target_doy is None:
                continue

            valid_pixel_count += 1

            # Process each band
            for band in range(6):
                if band != 4 and band != 5:  # Skip GRVI and DOY bands
                    band_series = reference_ts_block[band::6, row, col]
                    band_series_valid = band_series[valid_mask]

                    unique_doy_band, unique_indices_band = np.unique(doy_series[valid_mask], return_index=True)
                    band_series_valid = band_series_valid[unique_indices_band]

                    try:
                        # Interpolate band value at target DOY
                        f_band = interp1d(unique_doy_band, band_series_valid, kind='linear', fill_value='extrapolate')
                        interpolated_value = f_band(target_doy)
                    except ValueError as e:
                        logging.error(f"Band interpolation error: {e}")
                        interpolated_value = 0
                    
                    if np.isnan(interpolated_value):
                        interpolated_value = 0
                    
                    output_bands_block[band, row, col] = interpolated_value

            # Set GRVI band to 0 and DOY band to target DOY
            output_bands_block[4, row, col] = 0
            output_bands_block[5, row, col] = target_doy

    return output_bands_block, valid_pixel_count


def save_output_file(output_file, output_bands, reference_dataset):
    """
    Save processed output to GeoTIFF file.
    
    Args:
        output_file: Path for output file
        output_bands: Processed image bands
        reference_dataset: Reference dataset for georeferencing
    """
    try:
        driver = gdal.GetDriverByName("GTiff")
        out_dataset = driver.Create(output_file, output_bands.shape[2], output_bands.shape[1], 6, gdal.GDT_Float32)

        if out_dataset is None:
            raise IOError(f"Cannot create file: {output_file}")

        # Copy georeferencing information from reference dataset
        out_dataset.SetProjection(reference_dataset.GetProjection())
        out_dataset.SetGeoTransform(reference_dataset.GetGeoTransform())

        # Write each band
        for i in range(6):
            out_dataset.GetRasterBand(i + 1).WriteArray(output_bands[i, :, :])

        out_dataset.FlushCache()
        out_dataset = None
        logging.info(f"Output file saved at: {output_file}")
    except Exception as e:
        logging.error(f"Error saving file: {e}")
        raise


def process_state(state, emergence_day, year, doy_upper_limit, input_dir, output_dir, chunk_size=100):
    """
    Process a single state/province.
    
    Args:
        state: State/province code (e.g., 'HLJ')
        emergence_day: Emergence day for this state
        year: Target year
        doy_upper_limit: Upper DOY limit for processing
        input_dir: Directory containing input images
        output_dir: Directory for output files
        chunk_size: Size of processing chunks
    """
    # TODO: Replace these paths with your actual input/output paths
    reference_image_path = os.path.join(input_dir, f"{state}_{year}.tif")
    output_file = os.path.join(output_dir, state, f"{state}_{year}_Feature.tif")

    logging.info(f"Processing state: {state}, year: {year}")
    logging.info(f"Reference image path: {reference_image_path}")
    logging.info(f"Output file path: {output_file}")

    if not os.path.exists(reference_image_path):
        logging.warning(f"File not found: {reference_image_path}")
        return

    # Create output directory if it doesn't exist
    output_subdir = os.path.dirname(output_file)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    # Read input image
    img_reference_ts, reference_dataset = read_image_as_array(reference_image_path)
    rows, cols = img_reference_ts.shape[1], img_reference_ts.shape[2]

    # Initialize output array
    output_bands = np.zeros((6, rows, cols), dtype=np.float32)
    total_valid_pixel_count = 0

    # Process image in chunks
    for start_row in range(0, rows, chunk_size):
        end_row = min(start_row + chunk_size, rows)
        logging.info(f"Processing rows {start_row} to {end_row} for state {state}")

        reference_ts_block = img_reference_ts[:, start_row:end_row, :]
        output_bands_block, valid_pixel_count_block = process_block(
            reference_ts_block, start_row, end_row, cols, emergence_day, doy_upper_limit
        )
        output_bands[:, start_row:end_row, :] = output_bands_block
        total_valid_pixel_count += valid_pixel_count_block

    logging.info(f"Processing for state {state} completed. Valid pixels processed: {total_valid_pixel_count}")

    # Save output
    save_output_file(output_file, output_bands, reference_dataset)


def main():
    """
    Main function to process maize phenology data for Northeast China.
    
    TODO: Replace the placeholder paths with your actual input/output directories
    """
    # State/province codes and their emergence days
    states_and_emergence_days = {
        'HLJ': 120,  # Heilongjiang
        # Add other provinces as needed:
        # 'JL': 125,   # Jilin
        # 'LN': 130,   # Liaoning
    }
    
    year = 2024
    doy_upper_limit = 250
    chunk_size = 100
    
    # TODO: Set your input and output directories here
    input_directory = ""  # Replace with your input directory
    output_directory = ""     # Replace with your output directory

    for state, emergence_day in states_and_emergence_days.items():
        process_state(state, emergence_day, year, doy_upper_limit, input_directory, output_directory, chunk_size)


if __name__ == "__main__":
    # Ignore division by zero warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()
