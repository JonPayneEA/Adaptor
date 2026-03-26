# FEWS-Compliant Adapter for NeuralHydrology LSTM Models

A Delft-FEWS model adapter that runs trained [NeuralHydrology](https://neuralhydrology.readthedocs.io/) 
LSTM models within the FEWS General Adapter framework.

## Architecture

The adapter follows the standard FEWS three-phase pattern where FEWS and the 
model communicate exclusively via PI-XML files:

```
┌──────────────────────────────────────────────────────────────┐
│                     DELFT-FEWS                               │
│                                                              │
│  ┌────────────┐                        ┌────────────┐        │
│  │  Export    │   PI-XML timeseries    │  Import    │        │
│  │  Activities├───────────────────┐    │  Activities│        │
│  └────────────┘                   │    └─────▲──────┘        │
│                                   │          │               │
└───────────────────────────────────┼──────────┼───────────────┘
                                    │          │
                                    ▼          │
                    ┌───────────────────────────────────┐
                    │        MODEL ADAPTER              │
                    │                                   │
                    │  ┌──────────────┐                 │
                    │  │ pre_adapter  │  PI-XML → CSV   │
                    │  │   .py        │  + config.yml   │
                    │  └──────┬───────┘                 │
                    │         │                         │
                    │         ▼                         │
                    │  ┌──────────────┐                 │
                    │  │ run_model    │  Load LSTM      │
                    │  │   .py        │  → inference    │
                    │  └──────┬───────┘  → predictions  │
                    │         │                         │
                    │         ▼                         │
                    │  ┌──────────────┐                 │
                    │  │ post_adapter │  predictions    │
                    │  │   .py        │  → PI-XML       │
                    │  └──────────────┘                 │
                    └───────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `adapter_config.yml` | Maps FEWS parameters/locations to NeuralHydrology features/basins |
| `pi_xml.py` | PI-XML reader/writer utilities (timeseries, diagnostics, run info) |
| `pre_adapter.py` | Phase 1: converts exported PI-XML → NH per-basin CSVs |
| `run_model.py` | Phase 2: loads trained LSTM checkpoint, runs inference |
| `post_adapter.py` | Phase 3: converts predictions → PI-XML for FEWS import |
| `GeneralAdapterRun_NeuralHydrology.xml` | Example FEWS General Adapter config |

## Requirements

Python 3.9+ with:

```
torch
neuralhydrology
pandas
numpy
pyyaml
```

## Installation

1. Install the Python environment accessible to FEWS:
   ```bash
   pip install neuralhydrology torch pandas numpy pyyaml
   ```

2. Copy the adapter files to your FEWS module directory:
   ```
   Modules/
     neuralhydrology/
       bin/
         adapter_config.yml
         pi_xml.py
         pre_adapter.py
         run_model.py
         post_adapter.py
       model/
         runs/
           my_lstm_run/       ← trained NH run directory
             config.yml
             model_epoch030.pt
             train_data/
               train_data_scaler.p
   ```

3. Copy `GeneralAdapterRun_NeuralHydrology.xml` to your FEWS 
   `ModuleConfigFiles` directory and adjust the paths.

4. Edit `adapter_config.yml` to match your FEWS parameter IDs, 
   location IDs, and NeuralHydrology model configuration.

## Configuration

### adapter_config.yml

**Parameter mapping** — maps FEWS PI-XML `parameterId` values to the 
dynamic input feature names used when training the NH model:

```yaml
parameter_mapping:
  input:
    P.obs: "total_precipitation"    # FEWS → NH
    T.obs: "temperature"
  output:
    QObs_mm_per_day: "Q.simulated"  # NH target → FEWS
```

**Location mapping** — maps FEWS `locationId` to NH basin IDs:

```yaml
location_mapping:
  "gauge_01234": "01234500"
```

**Static attributes** — catchment properties used by the LSTM 
(must match the features used during training):

```yaml
static_attributes:
  "01234500":
    area_gages2: 2050.0
    elev_mean: 450.0
```

### FEWS General Adapter

Key points for the GA configuration:

- The `relativeViewPeriod` start must go back far enough to cover 
  the LSTM `seq_length` (e.g. -730 days for a 365-day sequence).
- Each adapter phase writes/reads the same `diag_neuralhydrology.xml` 
  diagnostics file.
- The `executeActivity` timeout should account for model loading time 
  (first run loads PyTorch).

## FEWS Compliance

The adapter conforms to FEWS adapter requirements:

- **PI-XML I/O**: All data exchange uses the Published Interface XML format
- **Diagnostics**: Each phase writes `pi_diag.xsd`-compliant diagnostics 
  with levels 0 (fatal) through 4 (debug)
- **Return codes**: Exit 0 on success, exit 1 on graceful failure
- **Black box**: FEWS has no knowledge of NeuralHydrology internals; 
  the adapter handles all translation
- **State management**: LSTM hidden/cell state can be exported via 
  PI state XML for warm-starting subsequent runs
- **Configurable**: All mappings are in `adapter_config.yml`, not hardcoded

## Diagnostics Levels

| Level | Name    | Meaning |
|-------|---------|---------|
| 4     | debug   | Verbose trace (file paths, tensor shapes) |
| 3     | info    | Normal progress (basin count, period) |
| 2     | warning | Non-fatal issues (missing features, NaN fill) |
| 1     | error   | Critical problems (missing files, failed inference) |
| 0     | fatal   | Unrecoverable crash |

## Extending

- **Multi-frequency models** (MTS-LSTM): adjust `pre_adapter.py` to write 
  separate CSVs per frequency and update the input tensor construction 
  in `run_model.py`.
- **Ensemble predictions**: loop over ensemble members in `run_model.py` 
  and write PI-XML with `ensembleId`/`ensembleMemberIndex` in the header.
- **Uncertainty (GMM/CMAL)**: the NH model output includes distribution 
  parameters; `post_adapter.py` can be extended to write quantile 
  timeseries as separate FEWS parameters (e.g. `Q.sim.p10`, `Q.sim.p90`).
