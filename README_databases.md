# Training Databases Overview

This document explains the structure and purpose of the different datasets used to train models (like `microflow_model2`) on the cluster. 

The training pipeline relies on several separate directories working together.

## 1. Original Dataset (`--dataset_dir`)
This is the core dataset containing the satellite imagery and the **forward** flow ground truth.
* **Format:** Uncompressed folders named `train/`, `validation/`, and `test/`.
* **File Naming:** `{frame_id}_{scaling_factor}_sample.npy.npz` (e.g., `000000_0_sample.npy.npz`).
* **Contents:** Contains the `pre` and `post` image arrays, as well as the standard forward displacement maps.
* **Notes:** On the cluster, these are currently stored as `.tar.gz` archives (e.g., `train_20000_sf0.tar.gz`). **These must be extracted** into the `train/` and `validation/` folders before the dataloader can read them.

## 2. Inverse Dataset (`--inverse_dataset_dir`)
This dataset contains mathematically reversed displacement fields, pre-computed for models that predict flow in the opposite direction.
* **Format:** Folders named `train/`, `validation/`, and `test/`.
* **File Naming:** `{frame_id}_{scaling_factor}_inverse.npy.npz` (e.g., `014299_0_inverse.npy.npz`).
* **Contents:** Contains ONLY the inverse ground truth labels (`inverse_flow_ew`, `inverse_flow_ns`).
* **Usage:** When you train a model with `"inverse"` in the name (like `irseparatedinverse_GeoFlowNet`), the dataloader automatically loads the `pre` and `post` images from the **Original Dataset**, but loads the target labels from this **Inverse Dataset**.

## 3. Split Definitions (`--split_dir`)
Instead of randomly reading files from a folder, the dataloader uses text files as an explicit "table of contents".
* **Files Required:** `train_names.txt`, `validation_names.txt`.
* **Purpose:** 
    1. It ensures the exact same subset of images is used every time. For example, if your config sets `TRAIN_COUNT: 20000`, it reads exactly the first 20,000 IDs from the text file.
    2. It guarantees reproducible shuffling and ordering across different machines.
* **Notes:** You don't need to extract these from the cluster archives, as they are already available inside your repository at `data/splits/`.

## 4. Fault Boundaries (`--fault_boundary_dir`)
Optional physical masks representing where the actual geological faults lie.
* **Format:** Flat folder containing numpy arrays.
* **File Naming:** `{frame_id}_fault_boundary.npy` (e.g., `014299_fault_boundary.npy`).
* **Contents:** Binary arrays (0 or 255 values).
* **Usage:** The dataloader loads these, dilates them slightly, and includes them in the batch dictionary, often used for specific loss masking or regularization near fault lines.

## Summary Checklist for Training
To run a successful training job, make sure you have:
1. `train/` and `validation/` folders extracted from your `original/` `.tar.gz` files.
2. The `inverse/` directory mapped properly.
3. Your local `data/splits/` directory mapped.
4. Optional: the `fault_boundaries/` mapped.



### Storage and extraction / scratch space on HPC clusters
Example workflow to extract the datasets and create splits on a high-performance cluster (e.g. Jean Zay, GRICAD, or SLURM cluster):

- Shared/long-term storage (compressed archives): `$WORK/dataset/`
- Fast scratch storage (uncompressed arrays): `$SCRATCH/FaultDeform/`

To extract and merge the different parts of the training data (e.g., scaling factors 0, 1, 2):

```bash
mkdir -p $SCRATCH/FaultDeform/original
cd $SCRATCH/FaultDeform/original

# Extract scaling factor archives
tar -xzf $WORK/dataset/original/train_20000_sf0.tar.gz
tar -xzf $WORK/dataset/original/train_20000_sf1.tar.gz
tar -xzf $WORK/dataset/original/train_20000_sf2.tar.gz

# If archives extract into separate subdirectories (e.g. train_sf0/), merge into a single train/ folder:
mkdir -p train
mv train_sf0/* train/ 2>/dev/null || true
mv train_sf1/* train/ 2>/dev/null || true
mv train_sf2/* train/ 2>/dev/null || true
```

Extract the pre-computed inverse training dataset:
```bash
mkdir -p $SCRATCH/FaultDeform/inverse
cd $SCRATCH/FaultDeform/inverse
tar -xzf $WORK/dataset/inverse/inverse_train.tar.gz
```

### Regenerating Inverse Flow Dataset
If the inverse dataset needs to be generated or regenerated from the original dataset, you can do so by running `generate_reverse_flow_dataset.py` for both the `train` and `validation` splits:

**1. Regenerate train split:**
```bash
python generate_reverse_flow_dataset.py \
    --dataset_dir $SCRATCH/FaultDeform/original/ \
    --split_dir data/splits/ \
    --split_name train \
    --split_count -1 \
    --save_dir $SCRATCH/FaultDeform/inverse/train/ \
    --split_scaling_factors 0 1 2
```

**2. Regenerate validation split:**
```bash
python generate_reverse_flow_dataset.py \
    --dataset_dir $SCRATCH/FaultDeform/original/ \
    --split_dir data/splits/ \
    --split_name validation \
    --split_count -1 \
    --save_dir $SCRATCH/FaultDeform/inverse/validation/ \
    --split_scaling_factors 0 1 2
```
