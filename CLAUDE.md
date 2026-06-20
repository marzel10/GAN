# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MATLAB deep learning project for structural health monitoring of PZT (Piezoelectric) sensor panels. Networks learn to reconstruct or compress time-series signals from 4 panels (103, 104, 105, 109) across 28 propagation paths and multiple frequencies. The latent space encodes a health index (HI) that tracks structural damage progression.

## Running Training

All training scripts are in `matlab_project/src/training/`. **Run them directly from their location** — each script sets up MATLAB paths automatically at the top. Do not run from the project root without the path setup block.

Key training scripts:
- `training_one_path_one_freq.m` — FC autoencoder for a single path + frequency
- `training_allp_sHI.m` / `training_1p_sHI.m` — trains with structural health index (sHI) loss (MSE + monotonicity + proxy labels)
- `Bayesian_optimization_1p1f_fc.m` — Bayesian hyperparameter search over FC architectures, loops over all 28 paths

To run Bayesian optimization across all paths: open `Bayesian_optimization_1p1f_fc.m` and run; it iterates `p_i = 2:28` internally and spawns a `parpool('local', 2)`.

Results are saved to `matlab_project/results/<net_name>/` with a timestamped `.mat` containing `trained_net`, `training_info`, and `model_metadata`.

## Path Setup (Critical)

Every training script must run the path-setup block at the top before anything else. The block:
1. Sets `projectRoot` two levels up from `src/training/`
2. Removes the old `C:\Users\Maria\Documents\Honours Programme\PZT` path to avoid class conflicts
3. Adds `src/models`, `src/models/layers`, `src/data`, `src/utils` with `-begin` priority

If `which('GAN')` points outside `GAN\src\models`, clear MATLAB's class cache (`clear classes`) and re-run path setup.

## Data Format — The Core Constraint

**Never split the 4000 time steps.** Each sample is one complete measurement cycle.

| Context | Format string | Tensor shape |
|---|---|---|
| CNN / 2D-conv networks | `"SSCB"` | `[4000, 2, 6, batch]` |
| Fully-connected networks | `"BC"` | `[4000, batch]` |

`trainingOptions` must set `InputDataFormats` and `TargetDataFormats` to match. MiniBatchSize ≤ 4 for CNN; up to 16 for FC.

Raw data files: `matlab_project/data/States_<panel>.mat` (panels 103, 104, 105, 109). Downsampled/lowpass variants are in `data/Not_needed_hopefully/`.

## Architecture

### Network classes (`src/models/`)
- `architectures_container` — static factory for all production networks: `buildOptimizedNetwork_compressed` (CNN autoencoder), `deep_fully_connected_network` (FC autoencoder with k-sparse bottleneck)
- `tiny_architectures_container` — small HI networks; also `network_connecter` to chain a frozen pre-trained AE with a new HI head
- `GAN.m` (layer) — custom Graph Attention Network layer; uses a learned adjacency matrix; `type=1` outputs two ports (`graph_out`, `latent_out`)

### Custom layers (`src/models/layers/`)
`kSparseLayer`, `amplitudeScaleLayer`, `removeMeanLayer`, `deconcatenation`, `eluBSSC`, `zeroPadding2dBSSC`, `Scaled_tanh`, `Polynomial_activation`, `change_format`

### Custom datastores (`src/data/`)
- `CyclemultiInputDatastore_separate_sin_freq_fc` — main datastore for FC training; implements `Datastore`, `MiniBatchable`, `Shuffleable`, `Subsettable`; extracts a single path and frequency from `States_<panel>` structs
- `one_path_sHI_sin_freq_fc_datastore` / `all_path_sHI_sin_freq_fc_datastore` — include proxy health-index labels as targets
- `latent_space_datastore` — feeds pre-computed latent vectors to the downstream HI network

Datastores are combined with `combine(..., ReadOrder="sequential")` and split 80/20 with `subset()`.

## Loss Functions

Loss functions are defined as local functions inside each training script:
- `multiOutputMSE` — standard MSE averaged over all outputs
- `mse_l1_loss` — amplitude-weighted MSE+L1
- `amplitude_aware_loss` — separate treatment of high- vs low-amplitude regions
- `multiOutputMSE_monotonicity` — MSE + monotonicity penalty + proxy-label loss (for sHI training)
- `peak_preserving_noise_suppressing_loss` — separates peak and noise regions

Switch the active loss by changing the `lossFcn = @(varargin) ...` line.

## Debugging Tensor Shapes

Add `fprintf` with `mat2str(size(X))` at every new data transformation. The `testing_datastore.m` utility script reads one batch and prints shapes. Use `size()` and `class()` checks liberally — MATLAB silently transposes or reorders dimensions in some operations.
