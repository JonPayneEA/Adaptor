#!/usr/bin/env python3
"""
pre_adapter.py - FEWS Pre-Adapter for NeuralHydrology

Converts FEWS PI-XML timeseries data into the file/directory structure
that NeuralHydrology expects for inference. This is Phase 1 of the
FEWS General Adapter → Model Adapter pipeline.

Usage (called by FEWS General Adapter):
    python pre_adapter.py <work_dir> <adapter_config_path>

What it does:
  1. Reads the FEWS run_info.xml to determine the run period.
  2. Reads exported PI-XML timeseries (forcing data).
  3. Maps FEWS locationIds → NH basin IDs and parameterIds → NH features.
  4. Writes per-basin CSV files in the NeuralHydrology format.
  5. Writes a temporary NH config override for the inference period.
  6. Optionally reads LSTM state from PI state XML.
  7. Writes PI diagnostics XML.

Exit codes:
  0 = success
  1 = graceful failure (diagnostics file written)
"""

import sys
import os
import shutil
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

from pi_xml import (
    parse_run_info,
    read_pi_timeseries,
    DiagnosticsWriter,
)


def main():
    if len(sys.argv) < 3:
        print("Usage: python pre_adapter.py <work_dir> <adapter_config_path>")
        sys.exit(1)

    work_dir = Path(sys.argv[1])
    config_path = Path(sys.argv[2])

    diag = DiagnosticsWriter()
    diag.info("NeuralHydrology pre-adapter started")
    diag.debug(f"Work directory: {work_dir}")
    diag.debug(f"Adapter config: {config_path}")

    try:
        _run_pre_adapter(work_dir, config_path, diag)
    except Exception as e:
        diag.fatal(f"Pre-adapter crashed: {e}")
        diag.fatal(traceback.format_exc())
        _write_diag_and_exit(work_dir, config_path, diag, exit_code=1)

    if diag.has_errors():
        _write_diag_and_exit(work_dir, config_path, diag, exit_code=1)
    else:
        diag.info("Pre-adapter completed successfully")
        _write_diag_and_exit(work_dir, config_path, diag, exit_code=0)


def _write_diag_and_exit(work_dir, config_path, diag, exit_code):
    """Write diagnostics and exit."""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        diag_file = cfg.get("files", {}).get("diagnostics", "diag_neuralhydrology.xml")
    except Exception:
        diag_file = "diag_neuralhydrology.xml"

    diag.write(str(work_dir / diag_file))
    sys.exit(exit_code)


def _run_pre_adapter(work_dir: Path, config_path: Path, diag: DiagnosticsWriter):
    """Core pre-adapter logic."""

    # --- Load adapter configuration ---
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    nh_cfg = cfg["neuralhydrology"]
    param_map = cfg["parameter_mapping"]
    loc_map = cfg["location_mapping"]
    static_attrs = cfg.get("static_attributes", {})
    files_cfg = cfg["files"]
    miss_val = cfg.get("missing_value", -999.0)

    # --- Read FEWS run info ---
    run_info_path = work_dir / "input" / files_cfg.get("run_info", "run_info.xml")
    if not run_info_path.exists():
        # Try alternative locations
        for alt in [work_dir / files_cfg.get("run_info", "run_info.xml"),
                    work_dir / "run_info.xml"]:
            if alt.exists():
                run_info_path = alt
                break

    if run_info_path.exists():
        run_info = parse_run_info(str(run_info_path))
        diag.info(f"Run period: {run_info.start_time} to {run_info.end_time}")
        diag.info(f"Time zero (T0): {run_info.time_zero}")
    else:
        diag.error(f"Run info file not found: {run_info_path}")
        return

    # --- Read PI-XML input timeseries ---
    input_ts_path = work_dir / "input" / files_cfg["input_timeseries"]
    if not input_ts_path.exists():
        input_ts_path = work_dir / files_cfg["input_timeseries"]

    if not input_ts_path.exists():
        diag.error(f"Input timeseries not found: {input_ts_path}")
        return

    diag.info(f"Reading input timeseries: {input_ts_path}")
    pi_data = read_pi_timeseries(str(input_ts_path), missing_value=miss_val)
    diag.info(f"Found {len(pi_data)} timeseries in input file")

    # --- Map to input parameter names ---
    input_param_map = param_map.get("input", {})

    # --- Create NH-compatible data structure per basin ---
    nh_data_dir = work_dir / "nh_data" / "timeseries"
    nh_data_dir.mkdir(parents=True, exist_ok=True)

    basins_processed = []

    for fews_loc_id, nh_basin_id in loc_map.items():
        diag.info(f"Processing location {fews_loc_id} → basin {nh_basin_id}")

        # Collect all input features for this location
        feature_dfs = {}
        for fews_param_id, nh_feature_name in input_param_map.items():
            key = (fews_loc_id, fews_param_id)
            if key in pi_data:
                feature_dfs[nh_feature_name] = pi_data[key]["value"]
                diag.debug(f"  Mapped {fews_param_id} → {nh_feature_name} "
                           f"({len(pi_data[key])} values)")
            else:
                diag.warn(f"  Parameter {fews_param_id} not found for location "
                          f"{fews_loc_id} in input timeseries")

        if not feature_dfs:
            diag.error(f"No input features found for location {fews_loc_id}")
            continue

        # Combine into single DataFrame
        basin_df = pd.DataFrame(feature_dfs)
        basin_df.index.name = "date"

        # NeuralHydrology expects the full spin-up period + forecast period.
        # The FEWS export should already include the spin-up window, but
        # we validate here.
        seq_length = nh_cfg.get("seq_length", 365)
        expected_start = run_info.end_time - timedelta(days=seq_length + 1)

        if basin_df.index[0] > expected_start:
            diag.warn(
                f"  Input starts at {basin_df.index[0]}, but seq_length={seq_length} "
                f"requires data from {expected_start}. LSTM spin-up may be incomplete."
            )

        # Write per-basin CSV (NeuralHydrology format)
        # NH expects: date column + feature columns
        basin_csv = nh_data_dir / f"{nh_basin_id}.csv"
        basin_df.to_csv(basin_csv, index=True)
        diag.debug(f"  Wrote {basin_csv} ({len(basin_df)} rows, "
                   f"{len(basin_df.columns)} features)")

        basins_processed.append(nh_basin_id)

    if not basins_processed:
        diag.error("No basins were successfully processed")
        return

    # --- Write basin list file ---
    basin_file = work_dir / "nh_data" / "basins.txt"
    with open(basin_file, "w") as f:
        for b in basins_processed:
            f.write(f"{b}\n")
    diag.debug(f"Wrote basin list: {basin_file}")

    # --- Write static attributes CSV (if used) ---
    if static_attrs:
        attrs_dir = work_dir / "nh_data" / "attributes"
        attrs_dir.mkdir(parents=True, exist_ok=True)

        # Build attributes DataFrame
        attr_rows = []
        for basin_id in basins_processed:
            if basin_id in static_attrs:
                row = {"basin_id": basin_id}
                row.update(static_attrs[basin_id])
                attr_rows.append(row)

        if attr_rows:
            attr_df = pd.DataFrame(attr_rows)
            attr_df.to_csv(attrs_dir / "attributes.csv", index=False)
            diag.debug(f"Wrote static attributes for {len(attr_rows)} basins")

    # --- Write NH inference config override ---
    # This tells the run_model.py step what period to evaluate
    inference_cfg = {
        "basins": basins_processed,
        "run_dir": str(Path(nh_cfg["run_dir"]).resolve())
            if Path(nh_cfg["run_dir"]).is_absolute()
            else nh_cfg["run_dir"],
        "epoch": nh_cfg.get("epoch"),
        "device": nh_cfg.get("device", "cpu"),
        "data_dir": str(nh_data_dir.parent),
        "start_time": run_info.start_time.strftime("%Y-%m-%d"),
        "end_time": run_info.end_time.strftime("%Y-%m-%d"),
        "time_zero": run_info.time_zero.strftime("%Y-%m-%d %H:%M:%S"),
        "forecast_time": run_info.time_zero.strftime("%Y-%m-%d %H:%M:%S"),
    }

    inference_cfg_path = work_dir / "nh_inference_config.yml"
    with open(inference_cfg_path, "w") as f:
        yaml.dump(inference_cfg, f, default_flow_style=False)
    diag.info(f"Wrote NH inference config: {inference_cfg_path}")

    diag.info(f"Pre-adapter: {len(basins_processed)} basins ready for inference")


if __name__ == "__main__":
    main()
