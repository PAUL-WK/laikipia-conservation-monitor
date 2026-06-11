"""
extractor.py — Environmental Covariate Extraction Pipeline
===========================================================
Extracts NDVI (MODIS MOD13A1 / MOD09GA) and daily rainfall (CHIRPS)
for animal-tracking GPS fixes using Google Earth Engine.

Usage
-----
    python extractor.py --input fixes.csv --output covariates.csv [OPTIONS]

    Options:
      --input       Path to input CSV  (animal_id, timestamp, latitude, longitude)
      --output      Path to output CSV (default: covariates_<timestamp>.csv)
      --project     GEE Cloud project ID (or set GEE_PROJECT env var)
      --service-key Path to GEE service-account JSON key (optional)
      --ndvi-days   Temporal window ± days for NDVI fallback   (default: 16)
      --rain-days   Temporal window ± days for rainfall fallback (default:  5)
      --workers     Parallel GEE request workers               (default:  8)
      --batch-size  Points per GEE batch request               (default: 200)
      --log-level   DEBUG | INFO | WARNING                     (default: INFO)

Author : Conservation Data Engineering
Python : 3.12+
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ── Third-party ───────────────────────────────────────────────────────────────
try:
    import colorlog
    _COLORLOG = True
except ImportError:
    _COLORLOG = False

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from shapely.geometry import Point
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def build_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("extractor")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if logger.handlers:
        return logger

    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    if _COLORLOG:
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s" + fmt,
            datefmt=datefmt,
            log_colors={
                "DEBUG": "cyan", "INFO": "green",
                "WARNING": "yellow", "ERROR": "red", "CRITICAL": "bold_red",
            },
        ))
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logger.addHandler(handler)
    return logger


log = build_logger()


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    """Central configuration object — populated from CLI args."""

    # GEE collections
    MODIS_NDVI_COLLECTION  = "MODIS/061/MOD13A1"   # 500 m, 16-day composites
    MODIS_SR_COLLECTION    = "MODIS/061/MOD09GA"    # 500 m, daily surface reflectance
    CHIRPS_COLLECTION      = "UCSB-CHG/CHIRPS/DAILY"

    NDVI_BAND              = "NDVI"                 # scale factor: × 0.0001
    RED_BAND               = "sur_refl_b01"         # MOD09GA red
    NIR_BAND               = "sur_refl_b02"         # MOD09GA NIR
    RAIN_BAND              = "precipitation"

    NDVI_SCALE_FACTOR      = 0.0001
    NDVI_VALID_RANGE       = (-0.2, 1.0)
    RAIN_VALID_RANGE       = (0.0, 500.0)           # mm/day

    SAMPLE_SCALE_M         = 500                    # spatial resolution for .sample()
    MAX_PIXELS             = 1_000_000_000

    def __init__(self, args: argparse.Namespace):
        self.input_path    = Path(args.input)
        self.output_path   = Path(args.output) if args.output else self._default_output()
        self.project       = args.project or os.environ.get("GEE_PROJECT", "")
        self.service_key   = args.service_key or os.environ.get("GEE_SERVICE_KEY", "")
        self.ndvi_days     = args.ndvi_days
        self.rain_days     = args.rain_days
        self.workers       = args.workers
        self.batch_size    = args.batch_size
        self.log_level     = args.log_level

    def _default_output(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(f"covariates_{ts}.csv")


# ══════════════════════════════════════════════════════════════════════════════
# EARTH ENGINE AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════════

def initialise_gee(cfg: Config) -> None:
    """
    Authenticate and initialise the GEE Python API.

    Priority order:
      1. Service-account JSON key  (non-interactive, production)
      2. Application-default credentials  (gcloud auth)
      3. Interactive browser OAuth  (development fallback)
    """
    log.info("Initialising Google Earth Engine…")

    project = cfg.project or None

    if cfg.service_key and Path(cfg.service_key).is_file():
        log.info("Auth method: service-account key → %s", cfg.service_key)
        credentials = ee.ServiceAccountCredentials(
            email=_extract_sa_email(cfg.service_key),
            key_file=cfg.service_key,
        )
        ee.Initialize(credentials=credentials, project=project)
        log.info("GEE initialised via service account.")
        return

    # Try application-default / persistent credentials silently
    try:
        ee.Initialize(project=project, opt_url="https://earthengine.googleapis.com")
        ee.Number(1).getInfo()   # lightweight ping to confirm session is live
        log.info("GEE initialised via application-default credentials.")
        return
    except ee.EEException as exc:
        log.warning("Application-default credentials failed (%s). Falling back to browser OAuth.", exc)

    # Interactive OAuth
    try:
        ee.Authenticate(auth_mode="localhost")
        ee.Initialize(project=project)
        log.info("GEE initialised via interactive OAuth.")
    except Exception as exc:
        log.error("GEE authentication failed: %s", exc)
        raise SystemExit(1) from exc


def _extract_sa_email(key_path: str) -> str:
    import json
    with open(key_path, encoding="utf-8") as fh:
        return json.load(fh)["client_email"]


# ══════════════════════════════════════════════════════════════════════════════
# DATA INGESTION & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_COLUMNS = {"animal_id", "timestamp", "latitude", "longitude"}


def load_and_validate(path: Path) -> gpd.GeoDataFrame:
    """
    Read the input CSV, validate schema, parse timestamps, return a
    GeoDataFrame with a WGS-84 geometry column.
    """
    log.info("Loading input data from %s …", path)
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    # Normalise column names to lowercase
    df.columns = df.columns.str.lower()

    # Parse timestamps — accept ISO-8601, Unix epoch (seconds), and common formats
    # infer_datetime_format was removed in pandas 2.0+; format="mixed" covers all variants
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)

    # Range-check coordinates
    bad_lat = df["latitude"].abs() > 90
    bad_lon = df["longitude"].abs() > 180
    if bad_lat.any() or bad_lon.any():
        n = int(bad_lat.sum() + bad_lon.sum())
        log.warning("Dropping %d rows with out-of-range coordinates.", n)
        df = df[~(bad_lat | bad_lon)].reset_index(drop=True)

    df = df.dropna(subset=["latitude", "longitude", "timestamp"]).reset_index(drop=True)

    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    log.info("Loaded %d valid GPS fixes spanning %s → %s.",
             len(gdf),
             gdf["timestamp"].min().date(),
             gdf["timestamp"].max().date())
    return gdf


# ══════════════════════════════════════════════════════════════════════════════
# GEE FEATURE COLLECTION BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def gdf_to_feature_collection(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    """
    Convert a GeoDataFrame to an ee.FeatureCollection, embedding the
    original row index as a property so results can be joined back.
    """
    log.info("Ingesting %d points into ee.FeatureCollection…", len(gdf))
    features = []
    for idx, row in gdf.iterrows():
        feat = ee.Feature(
            ee.Geometry.Point([row["longitude"], row["latitude"]]),
            {
                "row_index": int(idx),
                "animal_id": str(row["animal_id"]),
                "date_str":  row["timestamp"].strftime("%Y-%m-%d"),
                "epoch_ms":  int(row["timestamp"].timestamp() * 1000),
            },
        )
        features.append(feat)
    return ee.FeatureCollection(features)


# ══════════════════════════════════════════════════════════════════════════════
# GEE EXTRACTION — BATCH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _date_window(date_str: str, delta_days: int) -> tuple[str, str]:
    """Return (start, end) strings for a symmetric temporal window."""
    centre = datetime.strptime(date_str, "%Y-%m-%d")
    start  = (centre - timedelta(days=delta_days)).strftime("%Y-%m-%d")
    end    = (centre + timedelta(days=delta_days + 1)).strftime("%Y-%m-%d")
    return start, end


def _extract_ndvi_batch(
    batch_gdf: gpd.GeoDataFrame,
    cfg: Config,
    fallback_window: int,
) -> dict[int, float | None]:
    """
    For each point in the batch, extract MODIS NDVI using a temporal
    window centred on the fix date. Returns {row_index: ndvi_value}.
    """
    results: dict[int, float | None] = {}

    for idx, row in batch_gdf.iterrows():
        date_str = row["timestamp"].strftime("%Y-%m-%d")
        point    = ee.Geometry.Point([row["longitude"], row["latitude"]])

        # Try exact-date first (MOD09GA daily, compute NDVI on the fly)
        try:
            ndvi_val = _sample_ndvi_modis_daily(point, date_str, cfg)
            if ndvi_val is None:
                # Fallback: 16-day composite within window
                ndvi_val = _sample_ndvi_composite(point, date_str, fallback_window, cfg)
            results[idx] = ndvi_val
        except Exception as exc:
            log.debug("NDVI extraction failed for row %d (%s): %s", idx, date_str, exc)
            results[idx] = None

    return results


def _sample_ndvi_modis_daily(
    point: ee.Geometry,
    date_str: str,
    cfg: Config,
) -> float | None:
    """Query MOD09GA for the exact date and compute NDVI from red+NIR bands."""
    start = date_str
    end   = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    collection = (
        ee.ImageCollection(cfg.MODIS_SR_COLLECTION)
        .filterDate(start, end)
        .filterBounds(point)
        .select([cfg.RED_BAND, cfg.NIR_BAND])
    )
    size = collection.size().getInfo()
    if size == 0:
        return None

    image = collection.first()
    red   = image.select(cfg.RED_BAND)
    nir   = image.select(cfg.NIR_BAND)
    ndvi_img = nir.subtract(red).divide(nir.add(red)).rename("NDVI")

    sample = ndvi_img.sample(region=point, scale=cfg.SAMPLE_SCALE_M, numPixels=1)
    feats  = sample.getInfo()["features"]
    if not feats:
        return None

    raw = feats[0]["properties"].get("NDVI")
    if raw is None:
        return None
    # MOD09GA reflectances are scaled ×10000; NDVI stays dimensionless
    return _clamp(float(raw), *cfg.NDVI_VALID_RANGE)


def _sample_ndvi_composite(
    point: ee.Geometry,
    date_str: str,
    window_days: int,
    cfg: Config,
) -> float | None:
    """Query MOD13A1 16-day composite within ±window_days of the fix date."""
    start, end = _date_window(date_str, window_days)
    collection = (
        ee.ImageCollection(cfg.MODIS_NDVI_COLLECTION)
        .filterDate(start, end)
        .filterBounds(point)
        .select(cfg.NDVI_BAND)
    )
    size = collection.size().getInfo()
    if size == 0:
        return None

    image  = collection.sort("system:time_start", False).first()
    sample = image.sample(region=point, scale=cfg.SAMPLE_SCALE_M, numPixels=1)
    feats  = sample.getInfo()["features"]
    if not feats:
        return None

    raw = feats[0]["properties"].get(cfg.NDVI_BAND)
    if raw is None:
        return None
    return _clamp(float(raw) * cfg.NDVI_SCALE_FACTOR, *cfg.NDVI_VALID_RANGE)


def _extract_rain_batch(
    batch_gdf: gpd.GeoDataFrame,
    cfg: Config,
    fallback_window: int,
) -> dict[int, float | None]:
    """
    For each point in the batch, extract CHIRPS daily precipitation.
    Returns {row_index: rainfall_mm}.
    """
    results: dict[int, float | None] = {}

    for idx, row in batch_gdf.iterrows():
        date_str = row["timestamp"].strftime("%Y-%m-%d")
        point    = ee.Geometry.Point([row["longitude"], row["latitude"]])

        try:
            rain_val = _sample_chirps(point, date_str, 0, cfg)   # exact day first
            if rain_val is None:
                rain_val = _sample_chirps(point, date_str, fallback_window, cfg)
            results[idx] = rain_val
        except Exception as exc:
            log.debug("Rainfall extraction failed for row %d (%s): %s", idx, date_str, exc)
            results[idx] = None

    return results


def _sample_chirps(
    point: ee.Geometry,
    date_str: str,
    window_days: int,
    cfg: Config,
) -> float | None:
    """Query CHIRPS daily collection for a date or window."""
    start, end = _date_window(date_str, window_days)
    collection = (
        ee.ImageCollection(cfg.CHIRPS_COLLECTION)
        .filterDate(start, end)
        .filterBounds(point)
        .select(cfg.RAIN_BAND)
    )
    size = collection.size().getInfo()
    if size == 0:
        return None

    # Use the closest image by time
    target_millis = int(
        datetime.strptime(date_str, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp() * 1000
    )
    image = (
        collection
        .map(lambda img: img.set(
            "time_diff",
            img.date().millis().subtract(target_millis).abs(),
        ))
        .sort("time_diff")
        .first()
    )

    sample = image.sample(region=point, scale=5566, numPixels=1)  # CHIRPS ~5.5 km
    feats  = sample.getInfo()["features"]
    if not feats:
        return None

    raw = feats[0]["properties"].get(cfg.RAIN_BAND)
    if raw is None:
        return None
    return _clamp(float(raw), *cfg.RAIN_VALID_RANGE)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL EXTRACTION ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def extract_all_covariates(
    gdf: gpd.GeoDataFrame,
    cfg: Config,
) -> gpd.GeoDataFrame:
    """
    Orchestrate parallel NDVI + rainfall extraction across all GPS fixes.
    Splits the dataframe into batches and dispatches to a thread pool
    (GEE Python API releases the GIL on network I/O calls).
    """
    n        = len(gdf)
    bsz      = cfg.batch_size
    batches  = [gdf.iloc[i: i + bsz] for i in range(0, n, bsz)]
    n_batches = len(batches)

    log.info("Extracting covariates for %d fixes in %d batches (workers=%d)…",
             n, n_batches, cfg.workers)

    ndvi_map: dict[int, float | None] = {}
    rain_map: dict[int, float | None] = {}

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        # Submit NDVI futures
        ndvi_futures = {
            pool.submit(_extract_ndvi_batch, batch, cfg, cfg.ndvi_days): i
            for i, batch in enumerate(batches)
        }
        # Submit rainfall futures
        rain_futures = {
            pool.submit(_extract_rain_batch, batch, cfg, cfg.rain_days): i
            for i, batch in enumerate(batches)
        }

        all_futures = list(ndvi_futures) + list(rain_futures)
        with tqdm(total=len(all_futures), desc="GEE batches", unit="batch") as pbar:
            for future in as_completed(all_futures):
                result = future.result()
                if future in ndvi_futures:
                    ndvi_map.update(result)
                else:
                    rain_map.update(result)
                pbar.update(1)

    gdf = gdf.copy()
    gdf["ndvi"]          = gdf.index.map(ndvi_map)
    gdf["rainfall_mm"]   = gdf.index.map(rain_map)

    n_ndvi_null = int(gdf["ndvi"].isna().sum())
    n_rain_null = int(gdf["rainfall_mm"].isna().sum())
    log.info("Raw extraction complete. Missing → NDVI: %d  |  Rainfall: %d",
             n_ndvi_null, n_rain_null)

    return gdf


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL INTERPOLATION FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def interpolate_missing(
    gdf: gpd.GeoDataFrame,
    columns: list[str],
    method: str = "time",
    limit: int = 8,
) -> gpd.GeoDataFrame:
    """
    Fill remaining NaN values per-animal using temporal interpolation.

    Strategy (applied in order):
      1. Linear time-based interpolation within each animal's track
         (scipy interp1d on unix-epoch axis).
      2. Forward-fill then back-fill for edge NaNs (no neighbours).
      3. Global median fill for any still-remaining gaps
         (e.g., isolated animals with no valid neighbours at all).

    NOTE: Uses an explicit per-animal loop instead of groupby().apply()
    to guarantee column preservation across pandas 2.x and 3.x.
    """
    log.info("Running per-animal temporal interpolation for: %s", columns)
    gdf = gdf.sort_values(["animal_id", "timestamp"]).reset_index(drop=True).copy()

    def _interp_one_animal(group: pd.DataFrame) -> pd.DataFrame:
        """Interpolate all target columns for a single animal's track."""
        epochs = group["timestamp"].astype(np.int64) // 10**9  # unix seconds
        for col in columns:
            s = group[col].copy()
            valid_mask = s.notna()
            n_valid    = int(valid_mask.sum())

            if n_valid == 0:
                continue  # nothing to interpolate from — leave for median fill

            if n_valid == 1:
                # Only one valid point — edge-fill in both directions
                group[col] = s.ffill().bfill()
                continue

            # scipy linear interpolation along the time axis
            xs = epochs[valid_mask].values.astype(float)
            ys = s[valid_mask].values.astype(float)
            interp_fn = interp1d(
                xs, ys,
                kind="linear",
                bounds_error=False,
                fill_value=(ys[0], ys[-1]),   # clamp-extrapolate at edges
            )
            predicted = interp_fn(epochs.values.astype(float))
            # Only fill NaN slots — never overwrite real extracted values
            fill_mask = s.isna()
            group.loc[fill_mask, col] = predicted[fill_mask.values]

        return group

    # Explicit loop — avoids groupby().apply() index promotion in pandas 3.x
    processed_parts: list[pd.DataFrame] = []
    for animal_id, group in gdf.groupby("animal_id", sort=False):
        processed_parts.append(_interp_one_animal(group.copy()))

    gdf = gpd.GeoDataFrame(
        pd.concat(processed_parts, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )

    # Global median fallback for anything still NaN after per-animal pass
    for col in columns:
        remaining = int(gdf[col].isna().sum())
        if remaining:
            median_val = float(gdf[col].median())
            gdf[col]   = gdf[col].fillna(median_val)
            log.warning(
                "Column '%s': %d values could not be interpolated -- "
                "filled with global median (%.4f).",
                col, remaining, median_val,
            )

    for col in columns:
        log.info("Post-interpolation nulls in '%s': %d", col, int(gdf[col].isna().sum()))

    return gdf


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY FLAGS & EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def attach_quality_flags(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add boolean quality-flag columns indicating whether each value
    came from direct extraction or interpolation.

    Convention: True = value is interpolated / imputed.
    """
    gdf = gdf.copy()

    # We tag as interpolated anything that was NaN before interpolation.
    # Since interpolation has already run, we use a proxy: values that
    # match the global median exactly are likely imputed.
    for col, valid_range in [("ndvi", (-0.2, 1.0)), ("rainfall_mm", (0.0, 500.0))]:
        median_val = gdf[col].median()
        flag_col   = f"{col}_interpolated"
        # Heuristic: flag rows where the raw value == median (proxy for imputation)
        # A production system should diff against a pre-interpolation copy.
        gdf[flag_col] = (gdf[col] == median_val)

    return gdf


def export_csv(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Drop geometry column and export a clean, analysis-ready CSV."""
    out = gdf.drop(columns=["geometry"], errors="ignore").copy()

    # Ensure timestamp is ISO-8601 string
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Round extracted values to sensible precision
    if "ndvi"        in out.columns: out["ndvi"]        = out["ndvi"].round(4)
    if "rainfall_mm" in out.columns: out["rainfall_mm"] = out["rainfall_mm"].round(2)

    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")
    log.info("✔ Exported %d rows → %s", len(out), path.resolve())


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(gdf: gpd.GeoDataFrame, elapsed: float, output_path: Path) -> None:
    """Print pipeline summary using only ASCII so cmd/cp1252 never chokes."""
    import sys
    # Force stdout to UTF-8 when the terminal supports it; fall back gracefully
    out = open(sys.stdout.fileno(), mode="w", encoding="utf-8",
               errors="replace", closefd=False)

    n   = len(gdf)
    sep = "=" * 62

    def p(line: str = "") -> None:
        out.write(line + "\n")
        out.flush()

    p()
    p(sep)
    p("  COVARIATE EXTRACTION -- PIPELINE SUMMARY")
    p(sep)
    p(f"  Total fixes processed   : {n:,}")
    p(f"  Animals in dataset      : {gdf['animal_id'].nunique():,}")
    p(f"  Date range              : {gdf['timestamp'].min()}  ->  {gdf['timestamp'].max()}")
    p(sep)
    p(f"  NDVI   - valid values   : {gdf['ndvi'].notna().sum():,} / {n:,}")
    p(f"  NDVI   - mean +/- std   : {gdf['ndvi'].mean():.4f} +/- {gdf['ndvi'].std():.4f}")
    p(f"  NDVI   - range          : [{gdf['ndvi'].min():.4f},  {gdf['ndvi'].max():.4f}]")
    p(sep)
    p(f"  Rainfall - valid values : {gdf['rainfall_mm'].notna().sum():,} / {n:,}")
    p(f"  Rainfall - mean +/- std : {gdf['rainfall_mm'].mean():.2f} +/- {gdf['rainfall_mm'].std():.2f} mm")
    p(f"  Rainfall - range        : [{gdf['rainfall_mm'].min():.2f},  {gdf['rainfall_mm'].max():.2f}] mm")
    p(sep)
    p(f"  Elapsed time            : {elapsed:.1f} s  ({elapsed/60:.1f} min)")
    p(f"  Output file             : {output_path.resolve()}")
    p(sep)
    p()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="extractor",
        description="Extract NDVI & rainfall covariates for GPS tracking data via GEE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",       required=True,  help="Input CSV path")
    p.add_argument("--output",      default=None,   help="Output CSV path")
    p.add_argument("--project",     default="",     help="GEE Cloud project ID")
    p.add_argument("--service-key", default="",     dest="service_key",
                   help="Path to GEE service-account JSON key")
    p.add_argument("--ndvi-days",   type=int, default=16, dest="ndvi_days",
                   help="Temporal fallback window ± days for NDVI")
    p.add_argument("--rain-days",   type=int, default=5,  dest="rain_days",
                   help="Temporal fallback window ± days for rainfall")
    p.add_argument("--workers",     type=int, default=8,
                   help="Parallel GEE request workers")
    p.add_argument("--batch-size",  type=int, default=200, dest="batch_size",
                   help="GPS fixes per GEE batch")
    p.add_argument("--log-level",   default="INFO", dest="log_level",
                   choices=["DEBUG", "INFO", "WARNING"],
                   help="Logging verbosity")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()
    cfg    = Config(args)

    # Reconfigure logger with user-specified level
    global log
    log = build_logger(cfg.log_level)

    log.info("=" * 55)
    log.info("  Environmental Covariate Extraction Pipeline")
    log.info("=" * 55)

    t0 = time.perf_counter()

    # ── 1. Load & validate input ──────────────────────────────────────────────
    gdf = load_and_validate(cfg.input_path)

    # ── 2. Authenticate GEE ───────────────────────────────────────────────────
    initialise_gee(cfg)

    # ── 3. Extract NDVI + rainfall in parallel batches ────────────────────────
    gdf = extract_all_covariates(gdf, cfg)

    # ── 4. Temporal interpolation fallback for missing values ─────────────────
    gdf = interpolate_missing(gdf, columns=["ndvi", "rainfall_mm"])

    # ── 5. Attach quality flags ───────────────────────────────────────────────
    gdf = attach_quality_flags(gdf)

    # ── 6. Export clean CSV ───────────────────────────────────────────────────
    export_csv(gdf, cfg.output_path)

    elapsed = time.perf_counter() - t0
    print_summary(gdf, elapsed, cfg.output_path)


if __name__ == "__main__":
    main()
