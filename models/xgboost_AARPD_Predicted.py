import os
import logging
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as rio_transform
from rasterio.transform import xy
from rasterio.windows import Window
from joblib import load

# ================== Runtime Switches and Configuration ==================
# Train with longitude and latitude → True; No scaling → False
INCLUDE_GEO = True
USE_SCALER = False
CHUNKED = True  # True: Process by chunks to save memory; False: Process entire image at once

# Growth stages, regions, and year range
PHASES = ['emergence', 'three-leaf', 'seven-leaf', 'jointing', 'tasseling', 'flowering', 'silking', 'milk-stage', 'maturity']
STATES = ['HLJ', 'JL', 'WSM', 'LN']
YEAR_RANGE = (2001, 2025)  # Left-inclusive, right-exclusive: 2001..2024

# TODO: Set your input/output path templates here
FEATURE_TIF_TEMPLATE = ""  # TODO: Set your input feature image path template
OUTPUT_TIF_TEMPLATE = ""   # TODO: Set your output prediction result path template

# Model path template (consistent with training)
MODEL_PATH_TEMPLATE = "../pth/DB/xgb_{sg}.joblib"
# SCALER_PATH_TEMPLATE removed as scaler is not used

# === Band Indices (must be consistent with training) ===
# Total features during training = 15 = [Lon, Lat] + 13 bands; indices of these 13 bands in the image:
BAND_INDICES = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# Prediction target is "light difference", need to add back to baseline band to get absolute value;
# Index of baseline band:
BASELINE_BAND_INDEX = 12

# ================== Environment and Logging ==================
os.environ.setdefault('NUMEXPR_MAX_THREADS', '32')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================== Utility Functions ==================
def load_models_and_scalers(stage_name: str):
    """Load model for specified growth stage (scaler not used)."""
    model_path = MODEL_PATH_TEMPLATE.format(sg=stage_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = load(model_path)
    scaler = None
    # Removed scaler-related code as it's not used
    logging.info(f"Model loaded: {stage_name}")
    return model, scaler

def to_lonlat_grid(src: rasterio.io.DatasetReader):
    """
    Convert pixel center coordinates to WGS84 longitude/latitude grid (lon_grid, lat_grid).
    Prefer pyproj with always_xy=True, compatible with older rasterio.warp.transform.
    """
    h, w = src.height, src.width
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    xs, ys = xy(src.transform, rows, cols, offset='center')
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()

    if src.crs is None:
        lons = xs
        lats = ys
    else:
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs(src.crs, CRS.from_epsg(4326), always_xy=True)
            lons, lats = transformer.transform(xs, ys)
        except Exception:
            lons, lats = rio_transform(src.crs, CRS.from_epsg(4326), xs.tolist(), ys.tolist())

    lon_grid = np.asarray(lons, dtype=np.float32).reshape(h, w)
    lat_grid = np.asarray(lats, dtype=np.float32).reshape(h, w)
    return lon_grid, lat_grid

def build_features(bands_block, lats_block, lons_block, nodata):
    """
    Construct feature matrix based on switches. Column order must match training:
    INCLUDE_GEO=True → [Lon, Lat, band_0, ..., band_12]  # Modified: lon first then lat
    """
    hb, wb = lats_block.shape
    valid = np.isfinite(lats_block) & np.isfinite(lons_block)
    if nodata is not None:
        for bi in BAND_INDICES:
            valid &= (bands_block[bi] != nodata)

    rows, cols = np.where(valid)
    if rows.size == 0:
        return None, None, None

    n_features = (2 if INCLUDE_GEO else 0) + len(BAND_INDICES)
    X = np.empty((rows.size, n_features), dtype=np.float32)

    c = 0
    if INCLUDE_GEO:
        # Modified: longitude (lon) first, then latitude (lat)
        X[:, c] = lons_block[rows, cols]
        c += 1
        X[:, c] = lats_block[rows, cols]
        c += 1

    for i, bi in enumerate(BAND_INDICES):
        X[:, c + i] = bands_block[bi, rows, cols]

    base_cum = bands_block[BASELINE_BAND_INDEX, rows, cols].astype(np.float32)
    return X, (rows, cols), base_cum

# ================== Core Processing ==================
def process_state(state: str, year: int, stage_name: str, model, scaler):
    """Process a single state/year/growth stage combination."""
    
    # TODO: Set your input and output paths here
    if not FEATURE_TIF_TEMPLATE or not OUTPUT_TIF_TEMPLATE:
        logging.error("Please set FEATURE_TIF_TEMPLATE and OUTPUT_TIF_TEMPLATE at the top of the script")
        return
    
    input_tif_path = FEATURE_TIF_TEMPLATE.format(state=state, year=year)
    output_tif_path = OUTPUT_TIF_TEMPLATE.format(state=state, year=year, sg=stage_name)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_tif_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_tif_path):
        logging.warning(f"Input image missing: {input_tif_path} (skipping {state}-{year}-{stage_name})")
        return

    logging.info(f"Loading image: {input_tif_path}")
    with rasterio.open(input_tif_path) as src:
        profile = src.profile
        h, w = src.height, src.width
        nodata = src.nodata
        # Unified NoData output value
        out_nodata = nodata if nodata is not None else -9999.0

        # Prepare longitude/latitude grid
        lons, lats = to_lonlat_grid(src)

        # Result array
        pred_abs = np.full((h, w), out_nodata, dtype=rasterio.float32)

        def predict_block(window: Window):
            """Predict values for a single window/chunk."""
            bands_block = src.read(window=window)  # shape: (nb, hb, wb)
            r0, c0 = int(window.row_off), int(window.col_off)
            hb, wb = bands_block.shape[1], bands_block.shape[2]

            lats_block = lats[r0:r0+hb, c0:c0+wb]
            lons_block = lons[r0:r0+hb, c0:c0+wb]

            X, rc, base_cum = build_features(bands_block, lats_block, lons_block, nodata)
            if X is None:
                return

            # Removed scaler-related code as it's not used
            y_diff = model.predict(X).astype(np.float32)
            y_abs = y_diff + base_cum

            rows, cols = rc
            pred_abs[r0 + rows, c0 + cols] = y_abs

        if CHUNKED:
            for _, window in src.block_windows(1):  # Use block definition of first band
                predict_block(window)
        else:
            bands = src.read()
            X, rc, base_cum = build_features(bands, lats, lons, nodata)
            if X is not None:
                y_diff = model.predict(X).astype(np.float32)
                y_abs = y_diff + base_cum
                rows, cols = rc
                pred_abs[rows, cols] = y_abs

    # Write GeoTIFF (compressed + tiled)
    profile.update(
        dtype=rasterio.float32,
        count=1,
        nodata=out_nodata,
        tiled=True,
        compress='deflate',
        zlevel=1,
        blockxsize=256,
        blockysize=256
    )
    logging.info(f"Saving prediction result: {output_tif_path}")
    with rasterio.open(output_tif_path, 'w', **profile) as dst:
        dst.write(pred_abs, 1)
    logging.info(f"Completed: {state}-{year}-{stage_name}")

def main():
    """Main processing function."""
    
    # Check if path templates are set
    if not FEATURE_TIF_TEMPLATE or not OUTPUT_TIF_TEMPLATE:
        logging.error("Please set FEATURE_TIF_TEMPLATE and OUTPUT_TIF_TEMPLATE at the top of the script")
        return
    
    for sg in PHASES:
        model, scaler = load_models_and_scalers(sg)
        for state in STATES:
            for year in range(YEAR_RANGE[0], YEAR_RANGE[1]):
                logging.info(f"Processing {state} {year} {sg} ...")
                try:
                    process_state(state, year, sg, model, scaler)
                except Exception as e:
                    logging.error(f"Processing failed: {state}-{year}-{sg} | {e}")

if __name__ == '__main__':
    main()
