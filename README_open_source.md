# Conditional Diffusion for Structural Seismic Ground Motion Inversion

This repository provides the implementation of a conditional diffusion model for **structural seismic ground motion inversion**. The goal is to reconstruct bidirectional NS/EW ground motion time histories from multi-floor structural responses, such as displacement or acceleration responses, without requiring prior knowledge of structural dynamic parameters.

Structural seismic ground motion inversion is inherently ill-posed in the Hadamard sense because structural systems act as low-pass filters and the inverse mapping from floor responses to ground motion is generally non-injective. Conventional inversion methods usually depend on accurate structural dynamic parameters, which limits their applicability in practical engineering scenarios. This project addresses these limitations by learning a conditional generative mapping from structural floor responses to ground motion inputs.

## Overview

The proposed framework introduces a **conditional diffusion generative model** for structural seismic ground motion inversion. Given multi-floor displacement or acceleration responses of a single building, the model directly reconstructs the corresponding bidirectional ground motion time histories in the NS and EW directions.

The framework learns the conditional probability distribution between floor responses and ground motions. This enables not only direct generative reconstruction, but also uncertainty characterization through repeated stochastic sampling.

A frequency-weighted training loss is also included to emphasize low-frequency components and response-spectrum-related information that are important for earthquake engineering assessment.

## Key Features

- Conditional diffusion model for seismic ground motion inversion.
- Direct reconstruction of bidirectional NS/EW ground motion time histories.
- Supports both acceleration-response and displacement-response inputs.
- No prior structural dynamic parameters are required.
- Multi-floor response conditioning with flexible layer combinations.
- Frequency-weighted loss for improved recovery of engineering-relevant spectral features.
- DDIM sampling for efficient inference.
- Ensemble sampling for uncertainty quantification.
- Full-record reconstruction with Hann overlap-add stitching.
- Evaluation metrics including RMSE, NRMSE, waveform correlation, peak error, peak timing error, and response-spectrum error for acceleration mode.
- Deterministic U-Net baseline for comparison.

## Method Summary

The inversion task is formulated as a conditional generative modeling problem. Let the structural floor response be the condition and the ground motion be the target signal. During training, the diffusion model learns to denoise corrupted ground motion samples conditioned on the observed floor responses. During inference, the model starts from random noise and progressively generates ground motion time histories under the given structural response condition.

For uncertainty quantification, the same condition can be sampled multiple times with stochastic DDIM sampling. The ensemble mean is used as the final prediction, while the pointwise standard deviation provides an uncertainty estimate.

## Repository Structure

```text
.
├── dataset.py              # Dataset loading, window slicing, normalization, and layer selection
├── trainer.py              # Main training pipeline for the conditional diffusion model
├── test.py                 # Deterministic forward U-Net baseline training script
├── invert.py               # Single-window ground motion inversion example
├── reconstruct_full.py     # Full-record reconstruction, metrics, visualization, and summary plots
├── utils.py                # Random seed setup and basic plotting utilities
└── net/
    ├── component.py        # Sinusoidal time embedding and basic 1D convolution block
    └── Diffusion.py        # Conditional diffusion U-Net, diffusion manager, DDIM sampler, and loss
```

## Installation

Create a Python environment and install the required dependencies.

```bash
conda create -n seismic-diffusion python=3.10
conda activate seismic-diffusion

pip install torch numpy pandas scipy matplotlib tqdm
```

Install the PyTorch version that matches your CUDA environment. Please refer to the official PyTorch installation instructions for the correct command for your system.

## Data Preparation

The dataset is expected to contain `.npz` files and an annotation CSV file.

Each `.npz` file should contain:

```text
features    # Floor response features, shape: [T, 12]
labels      # Ground motion labels, shape: [T, 4]
```

The expected feature-channel convention is:

```text
features columns:
  disp:
    Layer 1: [0, 1]
    Layer 2: [4, 5]
    Layer 3: [8, 9]

  acc:
    Layer 1: [2, 3]
    Layer 2: [6, 7]
    Layer 3: [10, 11]
```

The expected label-channel convention is:

```text
labels columns:
  acc:  [0, 1]    # NS, EW acceleration
  disp: [2, 3]    # NS, EW displacement
```

The annotation CSV file should include at least the following columns:

```text
filename
source_folder
A
D
```

where:

- `filename` is the name of the `.npz` file.
- `source_folder` is the subfolder containing the file.
- `A = 1` indicates that the record is available for acceleration-mode training.
- `D = 1` indicates that the record is available for displacement-mode training.

A typical data directory may look like:

```text
<DATASET_ROOT>/
├── data_annotation.csv
├── folder_001/
│   ├── record_001.npz
│   └── record_002.npz
└── folder_002/
    ├── record_003.npz
    └── record_004.npz
```

Before running the code, replace the dataset placeholders in the scripts with your local paths, for example:

```python
DATA_ROOT = "<DATASET_ROOT>"
CSV_PATH = os.path.join(DATA_ROOT, "<DATASET_ANNOTATION_CSV>")
```

## Training the Conditional Diffusion Model

The main training entry point is `trainer.py`.

Example configuration:

```python
cfg = Config(
    DATA_TYPE="acc",
    LAYER_COMBO=[0, 2],
    SAVE_DIR="experiments/run_acc02",
    FREQ_ALPHA=0.5,
    FREQ_FCUT=10.0,
    UQ_SAMPLES=10,
)
```

Run training:

```bash
python trainer.py
```

Important configuration fields:

| Field | Description |
|---|---|
| `DATA_ROOT` | Root directory of the dataset |
| `DATA_TYPE` | Signal mode: `"acc"` or `"disp"` |
| `LAYER_COMBO` | Floor response layers used as condition, e.g. `[0, 1, 2]` |
| `WINDOW_SIZE` | Window length for training samples |
| `STRIDE` | Sliding-window stride |
| `TIMESTEPS` | Number of diffusion timesteps |
| `DDIM_STEPS` | Number of DDIM sampling steps |
| `FREQ_ALPHA` | Weight of the time-domain loss |
| `FREQ_FCUT` | Cutoff frequency for frequency-weighted loss |
| `UQ_SAMPLES` | Number of ensemble samples for uncertainty quantification |

Training outputs are saved under `SAVE_DIR`, including:

```text
experiments/run_xxx/
├── config.json
├── loss_curve.png
├── uq_visualization.png
├── visualizations/
├── checkpoints/
│   ├── best_model.pth
│   └── epoch_xxx.pth
└── full_record_results/
```

## Frequency-Weighted Loss

The frequency-weighted loss combines time-domain MSE and frequency-domain weighted error:

```text
Loss = freq_alpha * Loss_time + (1 - freq_alpha) * Loss_freq
```

The frequency-domain term assigns higher weights to components below the cutoff frequency `FREQ_FCUT`. This encourages the model to recover low-frequency components that are especially important for structural response and engineering assessment.

## Full-Record Reconstruction

After training, `trainer.py` automatically loads the best checkpoint and calls full-record reconstruction.

The reconstruction pipeline:

1. Groups validation windows by record.
2. Performs DDIM-based inversion for each window.
3. Reconstructs the full time history using Hann overlap-add.
4. Removes mean bias and applies spike filtering.
5. Computes evaluation metrics.
6. Saves plots and a CSV summary.

The summary CSV is saved as:

```text
<SAVE_DIR>/full_record_results/metrics_summary.csv
```

## Single-Window Inversion

`invert.py` provides a simple example for reconstructing one validation sample.

```bash
python invert.py
```

Before running, set:

```python
DATA_ROOT = "<DATASET_ROOT>"
CSV_PATH = "<DATASET_ANNOTATION_CSV>"
MODEL_PATH = "<MODEL_CHECKPOINT_PATH>"
```

The script saves a comparison figure:

```text
inversion_result.png
```

## Uncertainty Quantification

The DDIM sampler supports ensemble sampling:

```python
mean, std, all_samples = sampler.sample_ensemble(
    model,
    condition,
    device,
    n_samples=10,
    eta=1.0,
)
```

The ensemble mean is used as the reconstructed ground motion, and the pointwise standard deviation is used as an uncertainty estimate. A 95% confidence interval can be approximated as:

```text
mean ± 1.96 * std
```

## Deterministic Baseline

`test.py` implements a deterministic 1D U-Net baseline that maps ground motion to floor responses. It can be used as a comparison model or as a forward modeling reference.

Run:

```bash
python test.py
```

## Evaluation Metrics

The reconstruction pipeline computes the following metrics:

| Metric | Description |
|---|---|
| `RMSE` | Root mean square error |
| `NRMSE` | Normalized RMSE |
| `CorrNS` | Pearson correlation coefficient in the NS direction |
| `CorrEW` | Pearson correlation coefficient in the EW direction |
| `ePeak_%` | Relative peak ground motion error |
| `dt_peak_s` | Peak timing error |
| `eSa_ns_%` | Response-spectrum error in the NS direction, acceleration mode only |
| `eSa_ew_%` | Response-spectrum error in the EW direction, acceleration mode only |

## Experimental Findings

Validation using full-scale shaking table test data of a three-story steel frame structure shows that the conditional diffusion framework can effectively reconstruct bidirectional ground motion time histories from structural floor responses.

Compared with a deterministic U-Net baseline, the conditional diffusion model with the frequency-weighted loss achieves improved waveform reconstruction and substantially reduced spectral error. In acceleration-response mode, the spectral errors in the NS and EW directions are reduced by 40.7% and 29.5%, respectively. In displacement-response mode, the peak ground displacement error is reduced from 20.21% to 11.76%, with improved NS-direction waveform correlation.

## Notes

- The dataset is not included in this repository.
- Replace all path placeholders before running the scripts.
- The code assumes two output channels corresponding to NS and EW directions.
- The default input length is 1024 samples.
- Layer indices follow zero-based indexing:
  - `0`: Layer 1
  - `1`: Layer 2
  - `2`: Layer 3

  
## License

This project is released for academic and research purposes. Please add an appropriate open-source license, such as MIT, Apache-2.0, or GPL-3.0, before public release.

## Contact

For questions, issues, or collaboration, please open an issue in this repository or contact the authors.
