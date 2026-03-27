#!/usr/bin/env python3
"""
post_adapter.py - FEWS Post-Adapter for NeuralHydrology

Converts NeuralHydrology LSTM predictions back into FEWS PI-XML
timeseries format. This is Phase 3 of the FEWS General Adapter →
Model Adapter pipeline.

Usage (called by FEWS General Adapter):
    python post_adapter.py <work_dir> <adapter_config_path>

What it does:
  1. Reads predictions from run_model.py output (pickle).
  2. Maps NH target variables → FEWS parameterIds.
  3. Maps NH basin IDs → FEWS locationIds.
  4. Trims predictions to the FEWS forecast period (T0 → end).
  5. Writes PI-XML timeseries to the FEWS importDir.
  6. Optionally writes LSTM state as PI state XML.
  7. Writes PI diagnostics XML.

Exit codes:
  0 = success
  1 = graceful failure (diagnostics file written)
"""

import sys
import pickle
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from pi_xml import (
    parse_run_info,
    write_pi_timeseries,
    DiagnosticsWriter,
)


def main():
    if len(sys.argv) < 3:
        print("Usage: python post_adapter.py <work_dir> <adapter_config_path>")
        sys.exit(1)

    work_dir = Path(sys.argv[1])
    config_path = Path(sys.argv[2])

    diag = DiagnosticsWriter()
    diag.info("NeuralHydrology post-adapter started")

    try:
        _run_post_adapter(work_dir, config_path, diag)
    except Exception as e:
        diag.fatal(f"Post-adapter crashed: {e}")
        diag.fatal(traceback.format_exc())
        _write_diag_and_exit(work_dir, config_path, diag, exit_code=1)

    if diag.has_errors():
        _write_diag_and_exit(work_dir, config_path, diag, exit_code=1)
    else:
        diag.info("Post-adapter completed successfully")
        _write_diag_and_exit(work_dir, config_path, diag, exit_code=0)


def _write_diag_and_exit(work_dir, config_path, diag, exit_code):
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        diag_file = cfg.get("files", {}).get("diagnostics", "diag_neuralhydrology.xml")
    except Exception:
        diag_file = "diag_neuralhydrology.xml"
    diag.write(str(work_dir / diag_file))
    sys.exit(exit_code)


def _run_post_adapter(work_dir: Path, config_path: Path, diag: DiagnosticsWriter):
    """Core post-adapter logic."""

    # --- Load adapter config ---
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_param_map = cfg["parameter_mapping"].get("output", {})
    loc_map = cfg["location_mapping"]
    files_cfg = cfg["files"]
    miss_val = cfg.get("missing_value", -999.0)
    time_zone = cfg.get("time_zone", 0.0)

    # Reverse location map: NH basin_id → FEWS locationId
    reverse_loc_map = {v: k for k, v in loc_map.items()}

    # --- Load inference config (for period info) ---
    inference_cfg_path = work_dir / "nh_inference_config.yml"
    if inference_cfg_path.exists():
        with open(inference_cfg_path) as f:
            inf_cfg = yaml.safe_load(f)
        forecast_time = datetime.strptime(inf_cfg["time_zero"], "%Y-%m-%d %H:%M:%S")
        start_time = datetime.strptime(inf_cfg["start_time"], "%Y-%m-%d")
        end_time = datetime.strptime(inf_cfg["end_time"], "%Y-%m-%d")
    else:
        diag.warn("Inference config not found, using run_info.xml for period")
        run_info_path = work_dir / "input" / files_cfg.get("run_info", "run_info.xml")
        if not run_info_path.exists():
            run_info_path = work_dir / "run_info.xml"
        run_info = parse_run_info(str(run_info_path))
        forecast_time = run_info.time_zero
        start_time = run_info.start_time
        end_time = run_info.end_time

    diag.info(f"Forecast T0: {forecast_time}")
    diag.info(f"Output period: {start_time} to {end_time}")

    # --- Load predictions ---
    pred_pkl = work_dir / "nh_predictions.pkl"
    if not pred_pkl.exists():
        diag.error(f"Predictions file not found: {pred_pkl}. "
                   "Did run_model.py complete successfully?")
        return

    with open(pred_pkl, "rb") as f:
        all_predictions = pickle.load(f)

    diag.info(f"Loaded predictions for {len(all_predictions)} basins")

    # --- Build PI-XML output ---
    pi_output = {}

    for basin_id, pred_df in all_predictions.items():
        # Map basin → FEWS location
        fews_loc_id = reverse_loc_map.get(basin_id)
        if fews_loc_id is None:
            diag.warn(f"Basin {basin_id} has no FEWS location mapping, skipping")
            continue

        # Trim to forecast period only (T0 onward)
        # FEWS typically only wants the forecast, not the spin-up
        forecast_mask = pred_df.index >= forecast_time
        forecast_df = pred_df[forecast_mask]

        if len(forecast_df) == 0:
            # Fall back: use predictions from start_time
            forecast_mask = pred_df.index >= start_time
            forecast_df = pred_df[forecast_mask]

        if len(forecast_df) == 0:
            diag.warn(f"No predictions in forecast window for basin {basin_id}")
            continue

        # Also trim to end_time
        forecast_df = forecast_df[forecast_df.index <= end_time]

        diag.info(f"Basin {basin_id} → location {fews_loc_id}: "
                  f"{len(forecast_df)} forecast values")

        # Create PI-XML series for each target variable
        for nh_target, fews_param_id in output_param_map.items():
            if nh_target not in forecast_df.columns:
                # Try matching by position if column names differ
                if len(forecast_df.columns) == 1:
                    col = forecast_df.columns[0]
                else:
                    diag.warn(f"Target {nh_target} not in predictions columns "
                              f"{list(forecast_df.columns)}")
                    continue
            else:
                col = nh_target

            # Build the output DataFrame FEWS expects
            out_df = pd.DataFrame({
                "value": forecast_df[col].values,
                "flag": 0,  # unreliable=0, reliable=2 etc.
            }, index=forecast_df.index)

            # Clip negative discharge to zero (physical constraint)
            if "Q" in fews_param_id.upper():
                neg_count = (out_df["value"] < 0).sum()
                if neg_count > 0:
                    diag.warn(f"Clipping {neg_count} negative discharge values "
                              f"to zero for {fews_loc_id}/{fews_param_id}")
                    out_df.loc[out_df["value"] < 0, "value"] = 0.0

            pi_output[(fews_loc_id, fews_param_id)] = out_df

    if not pi_output:
        diag.error("No output timeseries were generated")
        return

    # --- Determine timestep ---
    # Infer from the prediction index
    sample_df = list(pi_output.values())[0]
    if len(sample_df) > 1:
        dt_diff = (sample_df.index[1] - sample_df.index[0]).total_seconds()
        ts_multiplier = int(dt_diff)
    else:
        ts_multiplier = 86400  # default daily

    # --- Write PI-XML output ---
    output_dir = work_dir / "output"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / files_cfg["output_timeseries"]
    write_pi_timeseries(
        data=pi_output,
        output_path=str(output_path),
        missing_value=miss_val,
        time_zone=time_zone,
        time_step_unit="second",
        time_step_multiplier=ts_multiplier,
        forecast_time=forecast_time,
    )
    diag.info(f"Wrote output PI-XML: {output_path}")
    diag.info(f"  {len(pi_output)} series, timestep={ts_multiplier}s")

    # --- Write state file (optional) ---
    state_dir = work_dir / "nh_state"
    if state_dir.exists():
        state_files = list(state_dir.glob("state_*.pkl"))
        if state_files:
            _write_state_xml(state_files, work_dir, files_cfg, diag)

    diag.info(f"Post-adapter: {len(pi_output)} output series written")


def _write_state_xml(state_files, work_dir, files_cfg, diag):
    """Write LSTM state as a PI state XML for FEWS state management.

    The LSTM hidden/cell state is serialized as a binary companion
    alongside a minimal PI state XML wrapper. This allows FEWS to
    manage warm-start states through its state import/export mechanism.
    """
    import xml.etree.ElementTree as ET

    PI_NS = "http://www.wldelft.nl/fews/PI"

    state_out_path = work_dir / "output" / files_cfg.get("state_out", "state_out.xml")

    root = ET.Element("States")
    root.set("xmlns", PI_NS)
    root.set("version", "1.2")

    for sf in state_files:
        with open(sf, "rb") as f:
            state_data = pickle.load(f)

        state_el = ET.SubElement(root, "state")
        basin_el = ET.SubElement(state_el, "basinId")
        basin_el.text = state_data.get("basin_id", "unknown")

        date_el = ET.SubElement(state_el, "dateTime")
        date_el.set("date", state_data.get("last_date", "")[:10])
        date_el.set("time", "00:00:00")

        # State binary file reference
        bin_name = sf.name
        file_el = ET.SubElement(state_el, "stateFile")
        file_el.text = bin_name

        # Copy the pickle to the output dir as the state binary
        import shutil
        shutil.copy2(sf, work_dir / "output" / bin_name)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(state_out_path), encoding="UTF-8", xml_declaration=True)
    diag.info(f"Wrote state XML: {state_out_path}")


if __name__ == "__main__":
    main()
