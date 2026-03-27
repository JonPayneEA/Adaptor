#!/usr/bin/env python3
"""
run_model.py - FEWS Module for NeuralHydrology LSTM Inference

This is Phase 2 of the FEWS General Adapter → Model Adapter pipeline.
It loads a trained NeuralHydrology model and runs inference on the
data prepared by pre_adapter.py.

Usage (called by FEWS General Adapter):
    python run_model.py <work_dir> <adapter_config_path>

What it does:
  1. Reads the NH inference config written by the pre-adapter.
  2. Loads the trained NeuralHydrology model checkpoint.
  3. Prepares input tensors from the per-basin CSVs.
  4. Runs LSTM forward pass for each basin.
  5. Saves raw predictions to a pickle/CSV in work_dir.
  6. Optionally saves LSTM hidden state for warm-start.
  7. Writes PI diagnostics XML.

Exit codes:
  0 = success
  1 = graceful failure (diagnostics file written)

Dependencies:
  - torch
  - neuralhydrology
  - pandas, numpy, yaml
"""

import sys
import os
import pickle
import traceback
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd

from pi_xml import DiagnosticsWriter

# NeuralHydrology imports are deferred to allow graceful error handling
# if the package is not installed.


def main():
    if len(sys.argv) < 3:
        print("Usage: python run_model.py <work_dir> <adapter_config_path>")
        sys.exit(1)

    work_dir = Path(sys.argv[1])
    config_path = Path(sys.argv[2])

    diag = DiagnosticsWriter()
    diag.info("NeuralHydrology model runner started")

    try:
        _run_model(work_dir, config_path, diag)
    except Exception as e:
        diag.fatal(f"Model runner crashed: {e}")
        diag.fatal(traceback.format_exc())
        _write_diag_and_exit(work_dir, config_path, diag, exit_code=1)

    if diag.has_errors():
        _write_diag_and_exit(work_dir, config_path, diag, exit_code=1)
    else:
        diag.info("Model runner completed successfully")
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


def _run_model(work_dir: Path, config_path: Path, diag: DiagnosticsWriter):
    """Core model execution logic."""

    # --- Import NeuralHydrology (deferred for graceful error) ---
    try:
        import torch
        from neuralhydrology.nh_run import eval_run
        from neuralhydrology.utils.config import Config
        from neuralhydrology.modelzoo import get_model
        from neuralhydrology.datautils.utils import load_scaler
    except ImportError as e:
        diag.error(f"Failed to import required packages: {e}")
        diag.error("Ensure neuralhydrology and torch are installed in the "
                    "Python environment used by FEWS.")
        return

    # --- Load adapter config ---
    with open(config_path) as f:
        adapter_cfg = yaml.safe_load(f)

    # --- Load inference config written by pre-adapter ---
    inference_cfg_path = work_dir / "nh_inference_config.yml"
    if not inference_cfg_path.exists():
        diag.error(f"Inference config not found: {inference_cfg_path}. "
                   "Did the pre-adapter run successfully?")
        return

    with open(inference_cfg_path) as f:
        inf_cfg = yaml.safe_load(f)

    run_dir = Path(inf_cfg["run_dir"])
    epoch = inf_cfg.get("epoch")
    device = inf_cfg.get("device", "cpu")
    basins = inf_cfg["basins"]

    diag.info(f"NH run directory: {run_dir}")
    diag.info(f"Device: {device}")
    diag.info(f"Basins: {basins}")
    diag.info(f"Inference period: {inf_cfg['start_time']} to {inf_cfg['end_time']}")

    # --- Validate run directory ---
    if not run_dir.exists():
        diag.error(f"NeuralHydrology run directory does not exist: {run_dir}")
        return

    nh_config_path = run_dir / "config.yml"
    if not nh_config_path.exists():
        diag.error(f"NH config.yml not found in: {run_dir}")
        return

    # --- Load the NH config ---
    try:
        nh_cfg = Config(nh_config_path)
        diag.info(f"Loaded NH config: model={nh_cfg.model}, "
                  f"hidden_size={nh_cfg.hidden_size}, "
                  f"seq_length={nh_cfg.seq_length}")
    except Exception as e:
        diag.error(f"Failed to load NH config: {e}")
        return

    # --- Determine checkpoint ---
    if epoch is None:
        # Find the last checkpoint
        ckpt_dir = run_dir / "model_epoch"  # NH convention
        if not ckpt_dir.exists():
            # Try alternative: checkpoints stored directly in run_dir
            ckpts = list(run_dir.glob("model_epoch*.pt"))
            if not ckpts:
                diag.error("No model checkpoints found")
                return
            ckpt_path = sorted(ckpts)[-1]
        else:
            ckpts = sorted(ckpt_dir.glob("*.pt"))
            if not ckpts:
                diag.error("No .pt files in model_epoch directory")
                return
            ckpt_path = ckpts[-1]
    else:
        ckpt_path = run_dir / f"model_epoch{epoch:03d}.pt"
        if not ckpt_path.exists():
            # Alternative naming
            ckpt_path = run_dir / f"model_epoch{epoch}.pt"

    if not ckpt_path.exists():
        diag.error(f"Checkpoint not found: {ckpt_path}")
        return

    diag.info(f"Loading checkpoint: {ckpt_path}")

    # --- Load model ---
    try:
        model = get_model(nh_cfg)
        checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        diag.info("Model loaded successfully")
    except Exception as e:
        diag.error(f"Failed to load model: {e}")
        return

    # --- Load scaler (for input normalisation) ---
    try:
        scaler_path = run_dir / "train_data" / "train_data_scaler.p"
        if not scaler_path.exists():
            scaler_path = run_dir / "train_data_scaler.p"

        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            diag.info("Loaded training data scaler")
        else:
            diag.warn("No scaler found — inputs will NOT be normalised. "
                      "This will likely produce incorrect predictions.")
            scaler = None
    except Exception as e:
        diag.warn(f"Failed to load scaler: {e}")
        scaler = None

    # --- Get feature names from the NH config ---
    dynamic_inputs = nh_cfg.dynamic_inputs
    if isinstance(dynamic_inputs, dict):
        # Multi-frequency: flatten to single list for the primary frequency
        freq = list(dynamic_inputs.keys())[0]
        dynamic_inputs = dynamic_inputs[freq]

    static_inputs = getattr(nh_cfg, "static_attributes", []) or []
    target_vars = nh_cfg.target_variables

    diag.debug(f"Dynamic inputs: {dynamic_inputs}")
    diag.debug(f"Static inputs: {static_inputs}")
    diag.debug(f"Target variables: {target_vars}")

    # --- Run inference per basin ---
    all_predictions = {}

    for basin_id in basins:
        diag.info(f"Running inference for basin: {basin_id}")

        # Read prepared CSV
        basin_csv = work_dir / "nh_data" / "timeseries" / f"{basin_id}.csv"
        if not basin_csv.exists():
            diag.error(f"Basin CSV not found: {basin_csv}")
            continue

        try:
            basin_df = pd.read_csv(basin_csv, index_col=0, parse_dates=True)
        except Exception as e:
            diag.error(f"Failed to read basin CSV: {e}")
            continue

        # Extract dynamic features in the correct order
        missing_features = [f for f in dynamic_inputs if f not in basin_df.columns]
        if missing_features:
            diag.error(f"Missing dynamic features for basin {basin_id}: "
                       f"{missing_features}")
            continue

        x_d = basin_df[dynamic_inputs].values  # shape: (T, n_features)

        # Handle NaN in inputs (replace with 0 or interpolate)
        nan_count = np.isnan(x_d).sum()
        if nan_count > 0:
            diag.warn(f"Basin {basin_id}: {nan_count} NaN values in input, "
                      f"filling with interpolation/zero")
            x_d_df = pd.DataFrame(x_d, columns=dynamic_inputs)
            x_d_df = x_d_df.interpolate(limit_direction="both").fillna(0)
            x_d = x_d_df.values

        # Normalise using training scaler
        if scaler is not None:
            try:
                centers = scaler.get("xd_center", {})
                scales = scaler.get("xd_scale", {})
                for i, feat in enumerate(dynamic_inputs):
                    if feat in centers and feat in scales:
                        x_d[:, i] = (x_d[:, i] - centers[feat]) / (scales[feat] + 1e-10)
            except Exception as e:
                diag.warn(f"Scaler application failed: {e}. Using raw values.")

        # Static attributes
        x_s = None
        if static_inputs:
            attr_csv = work_dir / "nh_data" / "attributes" / "attributes.csv"
            if attr_csv.exists():
                attr_df = pd.read_csv(attr_csv)
                basin_attrs = attr_df[attr_df["basin_id"] == basin_id]
                if len(basin_attrs) > 0:
                    x_s = basin_attrs[static_inputs].values.astype(np.float32)
                    if scaler is not None:
                        try:
                            s_centers = scaler.get("xs_center", {})
                            s_scales = scaler.get("xs_scale", {})
                            for i, feat in enumerate(static_inputs):
                                if feat in s_centers and feat in s_scales:
                                    x_s[0, i] = ((x_s[0, i] - s_centers[feat])
                                                 / (s_scales[feat] + 1e-10))
                        except Exception as e:
                            diag.warn(f"Static scaler failed: {e}")

        # Build input tensor
        # x_d shape: (seq_len, n_features) → (1, seq_len, n_features)
        x_d_tensor = torch.from_numpy(x_d).float().unsqueeze(0).to(device)

        input_dict = {"x_d": x_d_tensor}

        if x_s is not None:
            x_s_tensor = torch.from_numpy(x_s).float().unsqueeze(0).to(device)
            input_dict["x_s"] = x_s_tensor

        # Forward pass
        try:
            with torch.no_grad():
                output = model(input_dict)

            # output["y_hat"] shape: (1, seq_len, n_targets)
            y_hat = output["y_hat"].cpu().numpy().squeeze(0)  # (seq_len, n_targets)

            # Denormalise predictions
            if scaler is not None:
                try:
                    y_centers = scaler.get("y_center", {})
                    y_scales = scaler.get("y_scale", {})
                    for i, target in enumerate(target_vars):
                        if target in y_centers and target in y_scales:
                            y_hat[:, i] = (y_hat[:, i] * y_scales[target]
                                           + y_centers[target])
                except Exception as e:
                    diag.warn(f"Output denormalisation failed: {e}")

            # Create output DataFrame
            pred_df = pd.DataFrame(
                y_hat,
                index=basin_df.index,
                columns=target_vars,
            )

            all_predictions[basin_id] = pred_df
            diag.info(f"Basin {basin_id}: prediction shape = {y_hat.shape}")

            # Optionally save LSTM hidden state for warm-starting
            if "h_n" in output:
                state_dir = work_dir / "nh_state"
                state_dir.mkdir(exist_ok=True)
                state = {
                    "h_n": output["h_n"].cpu().numpy(),
                    "c_n": output.get("c_n", output["h_n"]).cpu().numpy(),
                    "basin_id": basin_id,
                    "last_date": str(basin_df.index[-1]),
                }
                with open(state_dir / f"state_{basin_id}.pkl", "wb") as f:
                    pickle.dump(state, f)
                diag.debug(f"Saved LSTM state for basin {basin_id}")

        except Exception as e:
            diag.error(f"Inference failed for basin {basin_id}: {e}")
            diag.error(traceback.format_exc())
            continue

    if not all_predictions:
        diag.error("No predictions were generated for any basin")
        return

    # --- Save all predictions ---
    output_pkl = work_dir / "nh_predictions.pkl"
    with open(output_pkl, "wb") as f:
        pickle.dump(all_predictions, f)
    diag.info(f"Saved predictions to {output_pkl}")

    # Also save as CSV for debugging
    for basin_id, pred_df in all_predictions.items():
        csv_path = work_dir / f"nh_pred_{basin_id}.csv"
        pred_df.to_csv(csv_path)

    diag.info(f"Model runner: generated predictions for "
              f"{len(all_predictions)} basins")


if __name__ == "__main__":
    main()
