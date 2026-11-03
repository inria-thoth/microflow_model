# Training Procedure Guide

This document outlines how to execute the training pipeline for the `microflow_model2` repository, based on the current configuration and historical usage on the cluster.

## 1. The Entry Point
All training is orchestrated through the `train_fault_deform.py` script. This script acts as the main wrapper that parses your command-line arguments, loads the YAML configurations, initializes the dataset loaders, and triggers the PyTorch training loop.

## 2. Configuration Files (YAML)
Instead of passing dozens of hyperparameters via the command line, the script relies on YAML files located in `data/configs/train_fault_deform/`. 
* You select a configuration using the `--train_config_name` argument (without the `.yaml` extension).
* Example configs: `train_microflow` or `train_iterative_only`.
* These files dictate critical settings like the `MODEL_NAME` (e.g., `irseparatedinverse_GeoFlowNet`), maximum iterations, learning rate, and whether certain layers are frozen (like `searaft_freeze`).

## 3. Launching a Training Job
A complete command to launch training on the cluster requires pointing the script to your specific database directories and split files. 

### Training From Scratch (Default)
The previous developer typically trained these models completely from scratch. Here is the standard execution command:

```bash
python train_fault_deform.py \
    --train_config_name train_microflow \
    --dataset_dir /path/to/extracted/original_data/ \
    --inverse_dataset_dir /path/to/inverse/ \
    --split_dir data/splits/
```
*(Note: If you are using a model that does NOT have "inverse" in its name, you can omit the `--inverse_dataset_dir` parameter).*

### Training with Pre-Trained Backbone Weights
If you want to warm-start the iterations using pre-trained backbone weights (like the original GeoFlowNet model), you can inject them using the `--pretrained_model_filename` argument:

```bash
python train_fault_deform.py \
    --train_config_name train_microflow \
    --dataset_dir /path/to/extracted/original_data/ \
    --inverse_dataset_dir /path/to/inverse/ \
    --split_dir data/splits/ \
    --pretrained_model_filename /path/to/GeoFlowNet-A.pt
```

## 4. How the Model Selection Modifies Data Loading
The pipeline is designed to dynamically change how it loads data based on the `MODEL_NAME` specified in your YAML config:
* **Forward Models**: If you use a standard model name, it uses `FaultDeformDataset` which learns the flow from the *pre* image to the *post* image using the ground truth in `--dataset_dir`.
* **Inverse Models**: If your model name contains `"inverse"` (like `irseparatedinverse_GeoFlowNet`), the code intercepts the dataloader and switches to `InverseFaultDeformDataset`. It will continue loading the input images from `--dataset_dir`, but it will swap the target labels to the reverse flow maps found in `--inverse_dataset_dir`.

## 5. Additional Optional Arguments
* `--fault_boundary_dir /path/to/fault_boundaries/`: Use this if your model loss or regularization requires physical fault masks.
* `--save_offline`: Add this flag if you are running on a cluster node without internet access to ensure that Weights & Biases (wandb) logs are saved locally instead of attempting to upload immediately.


## 6. Tensorboard visualization
To visualize training progress with TensorBoard (or via JupyterHub on an HPC cluster), point `--logdir` to your checkpoints directory:
```bash
tensorboard --logdir results/checkpoints_microflow/
```