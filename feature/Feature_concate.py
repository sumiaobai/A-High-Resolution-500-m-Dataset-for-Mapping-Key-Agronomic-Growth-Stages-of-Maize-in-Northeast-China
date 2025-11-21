# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global constant for NoData value
OUT_NODATA = -9999.0


def find_meteo_path(base_dir, state, doy_int):
    """
    Try to find meteorology file with different naming patterns (1.tif, 001.tif, etc.)
    
    Args:
        base_dir: Base directory for meteorology data
        state: State/province code
        doy_int: Day of year as integer
        
    Returns:
        Path to meteorology file or None if not found
    """
    d = int(doy_int)
    # Try different naming patterns
    p1 = os.path.join(base_dir, state, f"{d}.tif")
    p2 = os.path.join(base_dir, state, f"{d:03d}.tif")
    if os.path.exists(p1): 
        return p1
    if os.path.exists(p2): 
        return p2
    return None


def process_state(state, year, base_input_path, base_meteo_path, output_dir_name="Feature_result"):
    """
    Process a single state/province to add meteorology data to feature images.
    
    Args:
        state: State/province code (e.g., 'HLJ', 'JL', 'LN', 'WSM')
        year: Target year
        base_input_path: Base directory containing input feature images
        base_meteo_path: Base directory containing meteorology data
        output_dir_name: Name of output subdirectory
        
    Returns:
        None
    """
    logging.info(f"Processing state: {state}, year: {year}")

    # TODO: Set your input and output paths here
    # Example:
    # base_input_path = "G:/东北作物产量数据集/特征影像-2"
    # base_meteo_path = f"J:/东北作物产量数据集/累积气象数据/{year}"
    
    if not base_input_path or not base_meteo_path:
        logging.error("Please set base_input_path and base_meteo_path parameters")
        return

    # Read feature image (masked=True to preserve original NoData)
    input_tif = os.path.join(base_input_path, state, f"{state}_{year}_Feature_output.tif")
    if not os.path.exists(input_tif):
        logging.error(f"File not found: {input_tif}")
        return

    with rasterio.open(input_tif) as src:
        # Read with masked arrays, then fill with OUT_NODATA and convert to float32
        input_ma = src.read(masked=True)  # shape: (bands, rows, cols)
        rows, cols = src.height, src.width
        dst_crs = src.crs
        dst_transform = src.transform
        profile = src.profile.copy()

    # Convert first 11 bands and DOY band from masked to float32 with NoData filled
    input_f32 = np.array([band.filled(OUT_NODATA).astype(np.float32) for band in input_ma])
    # 6th band is DOY (0-based index 5)
    doy_band = input_f32[5]
    # Round DOY to integer; values <=0 or equal to OUT_NODATA are considered invalid
    doy_int_band = np.where(np.isfinite(doy_band), np.rint(doy_band), 0).astype(np.int32)
    doy_int_band[doy_band <= 0] = 0
    unique_doys = np.unique(doy_int_band[doy_int_band > 0])
    logging.info(f"Unique valid DOYs: {len(unique_doys)}")

    # New array: 14 bands, all filled with NoData
    new_data = np.full((14, rows, cols), OUT_NODATA, dtype=np.float32)
    # Copy first 11 bands directly (already float32 with NoData handled)
    new_data[:11, :, :] = input_f32[:11, :, :]

    # Process each DOY, adding meteorology data
    filled_total = 0
    for doy in unique_doys:
        meteo_path = find_meteo_path(base_meteo_path, state, int(doy))
        if meteo_path is None:
            logging.warning(f"DOY {doy}: meteorology file not found under {os.path.join(base_meteo_path, state)}")
            continue

        with rasterio.open(meteo_path) as ms:
            src_crs = ms.crs
            src_transform = ms.transform
            m_count = ms.count
            max_use = min(3, m_count)  # Use up to 3 bands

            # Pre-allocate destination arrays with NoData
            dst_arrays = [np.full((rows, cols), OUT_NODATA, dtype=np.float32) for _ in range(max_use)]

            # Reproject each band
            for bi in range(max_use):
                src_band_ma = ms.read(bi + 1, masked=True)
                src_fill = ms.nodata if (ms.nodata is not None) else OUT_NODATA
                src_band = src_band_ma.filled(src_fill).astype(np.float32)

                reproject(
                    source=src_band,
                    destination=dst_arrays[bi],
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    src_nodata=src_fill,
                    dst_nodata=OUT_NODATA  # Uncovered areas remain NoData
                )

        # Assign values only for pixels with this DOY
        mask = (doy_int_band == int(doy))
        if max_use >= 1:
            new_data[11, mask] = dst_arrays[0][mask]
        if max_use >= 2:
            new_data[12, mask] = dst_arrays[1][mask]
        if max_use >= 3:
            new_data[13, mask] = dst_arrays[2][mask]

        filled = int(mask.sum())
        filled_total += filled
        logging.info(f"DOY {doy}: used {os.path.basename(meteo_path)}, bands used={max_use}, filled pixels={filled}")

    # Update profile: 14 bands, float32, unified NoData
    # Also add common compression/tiling options
    profile.update(
        count=14,
        dtype="float32",
        nodata=OUT_NODATA,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )

    # Create output directory/file
    output_dir = os.path.join(base_input_path, state, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    output_tif = os.path.join(output_dir, f"{state}_{year}_Feature_with_Meteorology.tif")

    # Write output
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(new_data)

    logging.info(f"Saved: {output_tif} (total filled pixels across DOYs: {filled_total})")


def main():
    """
    Main function to process multiple states and years.
    
    TODO: Set your input and output paths here
    """
    # TODO: Set your input and output paths here
    # Example:
    # base_input_path = ""
    # base_meteo_path_template = ""  # Use {} for year placeholder
    
    base_input_path = ""  # TODO: Set your base input path here
    base_meteo_path_template = ""  # TODO: Set your meteorology data path template here
    
    if not base_input_path or not base_meteo_path_template:
        logging.error("Please set base_input_path and base_meteo_path_template in main() function")
        return
    
    # States to process
    states = ['HLJ']
    
    # Years to process
    start_year = 2018
    end_year = 2019  # Exclusive, so processes 2018 only
    
    for state in states:
        for year in range(start_year, end_year):
            # Format meteorology path with current year
            base_meteo_path = base_meteo_path_template.format(year)
            process_state(state, year, base_input_path, base_meteo_path)


if __name__ == "__main__":
    main()
