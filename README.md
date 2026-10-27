# MicroFlow: Domain-Specific Optical Flow for Ground Deformation Estimation in Seismic Events

[![arXiv](https://img.shields.io/badge/arXiv-2504.13452-b31b1b.svg)](https://arxiv.org/abs/2504.13452)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

This repository contains the official code for **MicroFlow**, a deep learning framework designed to estimate surface ground deformation fields (East-West and North-South displacements) from optical satellite image pairs across seismic events.

The repository includes:
* **MicroFlow**: Iterative GeoFlowNet architecture with Total Variation (TV) and Local Total Variation (LTV) regularization.
* **Iterative unregularized flow**: Fast recurrent flow estimation without TV regularization.
* **SeaRAFT baseline**: Fine-tuned Sea-RAFT optical flow baseline adapted for high-resolution satellite imagery.
* **FaultDeform synthetic benchmark**: Dataloading, training, and evaluation scripts on semi-synthetic seismic rupture imagery.
* **High-Throughput & Batch Inference (v2.0)**: Streaming on-the-fly dataloader for large GeoTIFF imagery, seam boundary trimming, and multi-folder batch tile processing.
* **Comprehensive Cluster Guides (v2.0)**: Step-by-step documentation for HPC dataset extraction, inverse flow pre-computation, and distributed training.

---

## Detailed Documentation Guides

* [**Inference & Paper Reproduction Guide**](readme_inference.md): Instructions for validation, reproducing paper metrics (XS/S/M displacement, near-fault vs away-fault EPE), and tile processing.
* [**HPC Dataset Setup Guide**](README_databases.md): Storage hierarchy, cluster extraction, and inverse flow generation.
* [**Training & Execution Guide**](README_training.md): Guide for launching training runs, YAML configuration options, and TensorBoard monitoring.

---

## Environment Setup

### 1. Prerequisites (LAPACK for Prox-TV)
The TV/LTV regularization uses `prox-tv`, which requires LAPACK development headers:
```bash
# Ubuntu / Debian
sudo apt-get install liblapacke-dev

# Fedora / CentOS / RHEL
sudo yum install lapacke lapacke-devel
```

### 2. Python Environment
Create a virtual environment and install the required packages:

```bash
python -m venv microflow_env
source microflow_env/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## Pretrained Model Weights

Pretrained model weights can be downloaded from:
* **Google Drive**: [MicroFlow & SeaRAFT Weights](https://drive.google.com/drive/folders/1PB2yfOJH3e9VcFZySrmymd-4xDj2NzhP?usp=sharing)
  * `microflow_seed10.pt`: Full MicroFlow model with iterative refinement.
  * `searaft_s10.pt`: Fine-tuned SeaRAFT model.
  * `irseparated_GeoFlowNet_intermediatel1_0.8_sf012_e40.pt`: Pretrained baseline weights.

Place your downloaded `.pt` weight files into a local folder (e.g. `pretrained_models/`).

---

## Quick Start: Inference on Real Examples

A sample pair of real satellite GeoTIFF crops is provided in `data/example_real_data/` (`pl_01_pre.tif` and `pl_01_post.tif`).

### 1. Running MicroFlow
```bash
python inference_real_examples.py \
  --config_name microflow \
  --pretrained_model_filename pretrained_models/microflow_seed10.pt \
  --dataset_dir data/example_real_data/ \
  --save_dir results/microflow/ \
  --window_size 1024 \
  --stride 0 \
  --offset 0 \
  --device cuda
```
*(Use `--device cpu` if running without an Nvidia GPU).*

### 2. Running SeaRAFT
```bash
python inference_real_examples.py \
  --config_name searaft \
  --pretrained_model_filename pretrained_models/searaft_s10.pt \
  --dataset_dir data/example_real_data/ \
  --save_dir results/searaft/ \
  --window_size 1024 \
  --stride 0 \
  --offset 0 \
  --device cuda
```

### 3. Sliding Window Parameters for Large Images
For very large satellite image pairs:
* `--window_size`: Size of sliding window patches fed to the model (default: `1024`).
* `--stride`: Overlap step between patches (e.g., `512` for 50% overlap, `0` for non-overlapping tiles).
* `--offset`: Boundary pixels trimmed from each patch before blending (e.g., `128`) to eliminate edge artifacts.
* `--save_dir`: Directory where the georeferenced output GeoTIFFs (`_ew.tif` and `_ns.tif`) are saved.

---

## FaultDeform Semi-Synthetic Benchmark

### 1. Downloading the Dataset
The FaultDeform synthetic dataset is available on Recherche Data Gouv:
* **Download link**: [FaultDeform Dataset (DOI: 10.57745/G02ZXZ)](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/G02ZXZ&version=1.0)

Extract the archive to a directory (`FaultDeform_root_dir/`) containing `train/`, `val/`, and `test/` folders.

### 2. Evaluating on FaultDeform
```bash
python inference_fault_deform.py \
  --train_config_name irseparated_geoflownet_intermediatel1_noreg \
  --pretrained_model_filename pretrained_models/microflow_seed10.pt \
  --dataset_name faultdeform \
  --dataset_dir <FaultDeform_root_dir> \
  --split_dir data/splits/ \
  --split_name test \
  --split_scaling_factors 1 \
  --save_metrics \
  --metric_filename results/faultdeform_metrics.txt \
  --device cuda
```

### 3. Training on FaultDeform

MicroFlow models (`irseparatedinverse_GeoFlowNet`) predict reverse displacement fields to perform backward warping. Training them requires pre-computed inverse flow ground-truth labels.

#### Step 1: Generate the Inverse Flow Dataset
Run `generate_reverse_flow_dataset.py` on your extracted FaultDeform directory:
```bash
# Generate inverse labels for train split
python generate_reverse_flow_dataset.py \
  --dataset_dir <FaultDeform_root_dir>/train \
  --split_dir data/splits/ \
  --split_name train \
  --split_count -1 \
  --split_scaling_factors 0 1 2 \
  --save_dir <FaultDeform_inverse_dir>/train/

# Generate inverse labels for validation split
python generate_reverse_flow_dataset.py \
  --dataset_dir <FaultDeform_root_dir>/val \
  --split_dir data/splits/ \
  --split_name validation \
  --split_count -1 \
  --split_scaling_factors 0 1 2 \
  --save_dir <FaultDeform_inverse_dir>/val/
```

#### Step 2: Launch Training
Train MicroFlow with your generated `--inverse_dataset_dir`:
```bash
python train_fault_deform.py \
  --train_config_name train_microflow \
  --dataset_dir <FaultDeform_root_dir> \
  --inverse_dataset_dir <FaultDeform_inverse_dir> \
  --split_dir data/splits/ \
  --checkpoints_dir results/checkpoints_microflow/ \
  --device cuda \
  --seed 1
```

*(Note: If training a forward-only model such as `irseparated_geoflownet_intermediatel1_noreg`, you can omit `--inverse_dataset_dir`).*

Pre-saved training configuration files can be found in `data/configs/train_fault_deform/`.

---

## Citation

If you find this code or our pretrained models useful in your research, please cite our preprint:

```bibtex
@article{bertrand2025microflow,
  title={MicroFlow: Domain-Specific Optical Flow for Ground Deformation Estimation in Seismic Events},
  author={Bertrand, Juliette and Giffard-Roisin, Sophie and Hollingsworth, James and Mairal, Julien},
  journal={arXiv preprint arXiv:2504.13452},
  year={2025}
}
```

---

## Authors & Credits

* **Core Architecture & Modeling (v1.0):** Juliette Bertrand (developed during her research contract at Inria Thoth & ISTerre).
* **Maintenance, v2 Extensions & Scientific Supervision:** Sophie Giffard-Roisin (ISTerre / Université Grenoble Alpes), James Hollingsworth (ISTerre), Julien Mairal (Inria Thoth).

---

## Acknowledgments and License

This software was developed at **Inria** (Thoth team) in collaboration with **ISTerre** (Université Grenoble Alpes).

This project is licensed under the terms of the **GNU Affero General Public License v3 (AGPLv3)**. See the [LICENSE](LICENSE) file for details.
