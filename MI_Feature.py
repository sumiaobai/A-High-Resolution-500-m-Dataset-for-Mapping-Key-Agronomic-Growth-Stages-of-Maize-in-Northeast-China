import os
import re
import numpy as np
import rasterio
from tqdm import tqdm
import glob
import datetime
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def process_daily_gdd_accumulation(folder_path, output_folder=None):
    """
    Process daily GDD accumulation data, output files named by DOY value of the day.
    
    Args:
        folder_path: Path to folder containing daily TIFF files
        output_folder: Optional, output folder path (defaults to input folder)
        
    Returns:
        None
    """
    if output_folder is None:
        output_folder = folder_path

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all TIFF files and sort by date
    files = glob.glob(os.path.join(folder_path, "Test_*.tif"))
    pattern = r"Test_(\d{4})_(\d{2})_(\d{2})\.tif"

    def extract_date(file_path):
        """Extract date from filename for sorting."""
        match = re.search(pattern, os.path.basename(file_path))
        if match:
            year, month, day = map(int, match.groups())
            return datetime.datetime(year, month, day)
        return datetime.datetime.min

    # Sort files by date
    files.sort(key=extract_date)

    if not files:
        print("No TIFF files found matching the naming pattern")
        return

    # Read first file to get metadata
    with rasterio.open(files[0]) as src:
        profile = src.profile
        height, width = src.shape
        crs = src.crs
        transform = src.transform
        dtype = src.dtypes[0]
        meta = src.meta
        doy_dtype = src.dtypes[4]  # Assuming DOY is the 5th band

    # Initialize accumulation arrays
    accum_precip = np.zeros((height, width), dtype=np.float32)
    accum_ssrd = np.zeros((height, width), dtype=np.float32)
    accum_gdd = np.zeros((height, width), dtype=np.float32)

    # Prepare accumulation flag array
    accumulation_active = np.zeros((height, width), dtype=bool)

    # Process each file (daily processing)
    for i, file_path in enumerate(tqdm(files, desc="Processing daily data")):
        with rasterio.open(file_path) as src:
            # Read bands in known order:
            # Band order: 1-Temperature, 2-Precipitation, 3-SSRD, 4-GDD, 5-DOY
            temperature, precip, ssrd, gdd, doy_map = src.read()

            # Get current DOY value (take center pixel value)
            height, width = doy_map.shape
            current_doy = int(doy_map[height // 2, width // 2])

            # Convert to float32 for calculation (except DOY)
            precip = precip.astype(np.float32)
            ssrd = ssrd.astype(np.float32)
            gdd = gdd.astype(np.float32)

            # Mark positions where GDD > 0
            gdd_positive = gdd > 0

            # Update activation status
            new_activations = gdd_positive & (~accumulation_active)
            accumulation_active |= new_activations

            # Accumulate only in active areas
            if np.any(accumulation_active):
                accum_precip[accumulation_active] += precip[accumulation_active]
                accum_ssrd[accumulation_active] += ssrd[accumulation_active]
                accum_gdd[accumulation_active] += gdd[accumulation_active]

                # Save if current DOY >= 100
                if current_doy >= 100:
                    # Prepare output file
                    output_filename = f"{current_doy}.tif"
                    output_path = os.path.join(output_folder, output_filename)

                    # Create daily output data
                    output_precip = np.where(accumulation_active, accum_precip, 0)
                    output_ssrd = np.where(accumulation_active, accum_ssrd, 0)
                    output_gdd = np.where(accumulation_active, accum_gdd, 0)

                    # Update output metadata
                    output_meta = meta.copy()
                    output_meta.update(
                        count=4,
                        dtype='float32',
                        nodata=-9999
                    )
                    output_meta['dtype'] = 'float32'

                    # Save daily accumulation result
                    with rasterio.open(output_path, 'w', **output_meta) as dst:
                        dst.write(output_precip, 1)
                        dst.set_band_description(1, "Cumulative Precipitation")

                        dst.write(output_ssrd, 2)
                        dst.set_band_description(2, "Cumulative SSRD")

                        dst.write(output_gdd, 3)
                        dst.set_band_description(3, "Cumulative GDD")

                        # Write DOY band directly with original values (integer)
                        dst.write(doy_map.astype(doy_dtype), 4)
                        dst.set_band_description(4, "Day of Year")

    print(f"Processing completed! All results saved to: {output_folder}")


def process_states_years(states, start_year, end_year, input_template, output_template):
    """
    Process GDD accumulation for multiple states and years.
    
    Args:
        states: List of state codes to process
        start_year: First year to process (inclusive)
        end_year: Last year to process (exclusive)
        input_template: Template for input folder paths, use {} for state and year placeholders
        output_template: Template for output folder paths, use {} for state and year placeholders
        
    Returns:
        None
    """
    for state in states:
        for year in range(start_year, end_year):
            # TODO: Set your input and output paths here
            # Example:
            # tif_folder = f""
            # output_folder = f""
            
            if not input_template or not output_template:
                print("Please set input_template and output_template parameters")
                return
                
            tif_folder = input_template.format(state=state, year=year)
            output_folder = output_template.format(state=state, year=year)
            
            print(f"Processing state: {state}, year: {year}")
            print(f"Input folder: {tif_folder}")
            print(f"Output folder: {output_folder}")
            
            process_daily_gdd_accumulation(tif_folder, output_folder)


def main():
    """
    Main function to process GDD accumulation data.
    
    TODO: Set your input and output path templates here
    """
    # TODO: Set your input and output path templates here
    # Example:
    # input_template = ""
    # output_template = ""
    
    input_template = ""  # TODO: Set your input path template here
    output_template = ""  # TODO: Set your output path template here
    
    if not input_template or not output_template:
        print("Please set input_template and output_template in main() function")
        return
    
    # States to process
    states = ['WSM']
    
    # Years to process
    start_year = 2000
    end_year = 2001  # Exclusive, so processes 2000 only
    
    process_states_years(states, start_year, end_year, input_template, output_template)


if __name__ == "__main__":
    main()
