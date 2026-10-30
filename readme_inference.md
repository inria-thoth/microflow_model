# Inference Guide

This document describes how to run inference using the Microflow model, including standard evaluation on the FaultDeform dataset and the specific setup needed to reproduce the metrics reported in the paper.

## Standard Inference (Validation Set)

To run a standard evaluation on the validation set of the FaultDeform dataset, you can use the `inference_fault_deform.py` script. 
By default, the dataset contains multiple scaling factors. You can evaluate them by specifying `--split_scaling_factors`.

```bash
python inference_fault_deform.py \
    --train_config_name microflow \
    --pretrained_model_filename <path_to_your_microflow_weights.pth> \
    --dataset_name faultdeform \
    --dataset_dir <path_to_faultdeform_dataset> \
    --split_dir <path_to_splits_dir> \
    --split_name validation \
    --split_scaling_factors 0 1 2 \
    --split_count 5000 \
    --save_metrics \
    --metric_filename <path_to_save_metrics.txt>
```

## Reproducing Paper Metrics (Semi-synthetic FaultDeform)

In the paper, the semi-synthetic FaultDeform metrics are reported with the following specifics:
- Evaluated on the 10% test samples containing the highest number of fault lines.
- Evaluated on Extra Small (XS), Small (S), and Medium (M) displacements, which correspond to scaling factors `0`, `1`, and `2`.
- Errors are decoupled into EPE near the fault `EPE (F)` and EPE away from the fault `EPE (NF)`.

### How it works under the hood
1. **The 10% samples with highest fault lines**: The `--near_fault_only` flag forces the dataloader to read from `test_names_center.txt` instead of the standard `test_names.txt`. This file contains the pre-computed frame IDs for this 10% subset.
2. **XS, S, M displacement categories**: These correspond to the `--split_scaling_factors 0`, `1`, and `2` respectively.
3. **EPE (F) vs EPE (NF)**: By providing the `--fault_boundary_dir` and `--fault_boundary_disk 1`, the evaluation script computes `epe_near_fault` (F) and `epe_away_fault` (NF).

### Execution Command (Without Regularization)

To reproduce the exact metrics for the unregularized Microflow model (as currently being trained), run the script sequentially for each scaling factor. 

*Example for XS (Scaling Factor 0):*
```bash
python inference_fault_deform.py \
    --train_config_name microflow \
    --pretrained_model_filename <path_to_your_microflow_weights.pth> \
    --dataset_name faultdeform \
    --dataset_dir <path_to_faultdeform_dataset> \
    --split_dir <path_to_splits_dir_containing_test_names_center.txt> \
    --split_name test \
    --fault_boundary_dir <path_to_fault_boundaries/test> \
    --fault_boundary_disk 1 \
    --near_fault_only \
    --split_scaling_factors 0 \
    --save_metrics \
    --metric_filename <path_to_save_metrics.txt>
```

*Note: To reproduce S and M, repeat the command and change `--split_scaling_factors 0` to `1` and `2` respectively.*

## Inference on Real Examples

There are two primary ways to run inference on real-world satellite imagery, depending on whether you are processing a single massive image or a batch of small, pre-cropped tiles.

### 1. Single Large Image (`inference_real_examples.py`)

When dealing with a single massive image pair (e.g., thousands of pixels wide), the image must be processed in patches using a sliding window to fit into GPU memory. 

You can control this sliding window behavior using the following arguments:
- `--window_size` (default: 1024): The size of the sliding window patch (e.g., 1024x1024).
- `--stride` (default: 0): The stride for the sliding window.
- `--offset` (default: None): The offset to remove pixels on the boundaries of each window, minimizing border artifacts. If not provided, it defaults to half the overlap.
- `-b` / `--batch-size` (default: 1): The number of patches to process simultaneously. **Note:** Since patches are large, keep this small (e.g., 1 or 2) to prevent OOM errors.

**Example Command:**
```bash
python inference_real_examples.py \
    --config_name iterative_only \
    --pretrained_model_filename <path_to_weights.pt> \
    --dataset_name real_examples_onthefly \
    --dataset_dir <path_to_large_image_dir> \
    --save_dir <path_to_save_dir> \
    -b 2 \
    --window_size 1024 \
    --stride 512 \
    --offset 256
```

### 2. Batch Processing Small Tiles (`inference_batch_real_examples.py`)

If you have a parent directory containing multiple subdirectories of pre-cropped small tiles (e.g., `corr_*` folders which are exactly 1024x1024), you should use the batch script. This script loops through all subdirectories in a single job, loading the heavy model weights only once. 

Since these are small tiles, you **do not** need overlap or boundaries (e.g., offset and stride should be 0).

**Example Command (Single Job):**
```bash
python inference_batch_real_examples.py \
    --config_name iterative_only \
    --pretrained_model_filename <path_to_weights.pt> \
    --dataset_name real_examples_onthefly \
    --dataset_dir <path_to_parent_dir_of_corr_folders> \
    --save_dir <path_to_save_base_dir> \
    -b 16 \
    --window_size 1024 \
    --stride 0 \
    --offset 0
```

---

## Regularization Workflows (MicroFlow v2)

### 1. Saving Both Raw and Regularized Fields During Inference
When running with regularization enabled (e.g., `--config_name microflow`), the pipeline automatically saves:
* **Raw unregularized field:** `<image_name>_<config>_raw_ew_i0.tif` and `<image_name>_<config>_raw_ns_i0.tif`
* **Regularized field:** `<image_name>_<config>_ew_i0.tif` and `<image_name>_<config>_ns_i0.tif`

You can control this behavior using `--save_raw` / `--no-save_raw`, or toggle regularization via `--regularization` / `--no-regularization`.

### 2. Standalone Regularization on Existing GeoTIFFs (`regularize_displacement.py`)
To test or apply TV/LTV regularization to pre-computed displacement fields without reloading the neural network:

```bash
python regularize_displacement.py \
    --ew_path path/to/displacement_ew.tif \
    --ns_path path/to/displacement_ns.tif \
    --save_dir path/to/output_dir/ \
    --backend gpu \
    --penalty_function ltv \
    --reg_lambda 0.001 \
    --reg_iterations 3 \
    --suffix _reg
```
* **Backends:**
  * `--backend gpu` (Default): Native PyTorch energy minimization on GPU. Fast, scalable, and requires **no external C++ libraries** or `prox-tv`.
  * `--backend proxtv`: Legacy CPU dynamic programming solver via `prox-tv`.
* **Supported penalties:** `ltv` (Local Total Variation), `tv` (Total Variation), `l2`.
* Preserves the exact projection, CRS, and affine transform from the input GeoTIFFs.
