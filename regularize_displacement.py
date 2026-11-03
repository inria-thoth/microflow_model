"""
Standalone Regularization Script for Displacement Fields (MicroFlow v2)

Applies Total Variation (TV) or Local Total Variation (LTV) regularization
directly to existing East-West (EW) and North-South (NS) displacement GeoTIFFs,
without needing to load the neural network.

Usage Example:
    python regularize_displacement.py \
        --ew_path results/test/pl_01_microflow_raw_ew_i0.tif \
        --ns_path results/test/pl_01_microflow_raw_ns_i0.tif \
        --save_dir results/test/ \
        --penalty_function ltv \
        --reg_lambda 0.001 \
        --reg_iterations 3
"""

import argparse
import os
import time
import numpy as np
import rasterio
import torch

from src.models.model_utils.iterative_model_utils import compute_regularization
from src.dataloaders.tiff_utils import save_array_to_tiff


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Displacement Field Regularization")

    # Input / Output
    parser.add_argument("--ew_path", required=True, type=str, help="Path to East-West displacement GeoTIFF")
    parser.add_argument("--ns_path", required=True, type=str, help="Path to North-South displacement GeoTIFF")
    parser.add_argument("--save_dir", default=None, type=str, help="Directory to save regularized TIFFs (default: same as input)")
    parser.add_argument("--suffix", default="_reg", type=str, help="Suffix added to filename before extension")

    # Regularization parameters
    parser.add_argument("--penalty_function", default="ltv", choices=["ltv", "tv", "l2"], help="Regularization penalty function")
    parser.add_argument("--reg_lambda", default=0.001, type=float, help="Regularization lambda parameter")
    parser.add_argument("--reg_iterations", default=3, type=int, help="Number of regularization iterations")
    parser.add_argument("--reg_2d_max_iter", default=10, type=int, help="Maximum iterations per 2D alternance in prox-tv")
    parser.add_argument("--reg_threads", default=16, type=int, help="Number of CPU threads for prox-tv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (cuda or cpu)")
    parser.add_argument("--reg_lr", default=0.1, type=float, help="Learning rate for Adam optimizer in GPU proximal solver")
    parser.add_argument("--chunk_size", default=2048, type=int, help="Chunk size for processing large grids (default: 2048)")
    parser.add_argument("--backend", default="gpu", choices=["gpu", "proxtv"], help="Backend engine: pure PyTorch GPU (gpu) or legacy prox-tv (proxtv)")

    return parser.parse_args()


def regularize_displacement(args):
    # Verify input files exist
    if not os.path.isfile(args.ew_path):
        raise FileNotFoundError(f"EW displacement file not found: {args.ew_path}")
    if not os.path.isfile(args.ns_path):
        raise FileNotFoundError(f"NS displacement file not found: {args.ns_path}")

    device = torch.device(getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    # Set up save directory
    save_dir = args.save_dir if args.save_dir is not None else os.path.dirname(os.path.abspath(args.ew_path))
    os.makedirs(save_dir, exist_ok=True)

    # Load input GeoTIFFs
    with rasterio.open(args.ew_path) as src_ew:
        ew_data = src_ew.read(1).astype(np.float32)
        crs = src_ew.crs
        transform = src_ew.transform

    with rasterio.open(args.ns_path) as src_ns:
        ns_data = src_ns.read(1).astype(np.float32)

    assert ew_data.shape == ns_data.shape, f"EW shape {ew_data.shape} does not match NS shape {ns_data.shape}"
    h, w = ew_data.shape
    print(f"Loaded displacement fields: {w} × {h} pixels (device: {device}, backend: {args.backend})")
    print(f"  EW: {args.ew_path}")
    print(f"  NS: {args.ns_path}")

    # Convert from QGIS convention (-1 for EW) back to internal flow convention
    of = np.stack([-ew_data, ns_data], axis=0)  # Shape: (2, H, W)
    nan_mask = np.isnan(of)
    of[nan_mask] = 0

    t0 = time.time()
    ptv = None
    if args.backend == "proxtv":
        try:
            import prox_tv as ptv
        except ImportError:
            raise ImportError("prox-tv is not installed. Use --backend gpu to run the pure PyTorch regularizer.")

    of_tensor = torch.tensor(of[None, :], dtype=torch.float32, device=device if args.backend == "gpu" else "cpu")
    chunk_size = args.chunk_size

    for i_h in range(h // chunk_size + 1):
        for i_w in range(w // chunk_size + 1):
            start_h = i_h * chunk_size
            end_h = min((i_h + 1) * chunk_size, h)
            start_w = i_w * chunk_size
            end_w = min((i_w + 1) * chunk_size, w)
            if start_h >= end_h or start_w >= end_w:
                continue

            sub_tensor = of_tensor[:, :, start_h:end_h, start_w:end_w].contiguous()
            reg_sub = compute_regularization(sub_tensor, args, ptv)
            of[:, start_h:end_h, start_w:end_w] = reg_sub[0].cpu().numpy()

    t1 = time.time()
    print(f"Regularization ({args.penalty_function}, λ={args.reg_lambda}, iters={args.reg_iterations}, backend={args.backend}) completed in {t1 - t0:.2f}s")

    # Restore NaN values where original displacement had no data
    of[nan_mask] = np.nan

    # Convert EW back to QGIS convention
    of[0] = of[0] * -1

    # Build output filenames
    ew_base = os.path.splitext(os.path.basename(args.ew_path))[0]
    ns_base = os.path.splitext(os.path.basename(args.ns_path))[0]
    out_ew_filename = f"{ew_base}{args.suffix}.tif"
    out_ns_filename = f"{ns_base}{args.suffix}.tif"

    out_ew_path = os.path.join(save_dir, out_ew_filename)
    out_ns_path = os.path.join(save_dir, out_ns_filename)

    # Save regularized GeoTIFFs preserving geo-metadata
    save_array_to_tiff(of[0].astype(np.float32), out_ew_path, transform=transform, crs=crs)
    save_array_to_tiff(of[1].astype(np.float32), out_ns_path, transform=transform, crs=crs)

    print(f"Saved regularized GeoTIFFs:")
    print(f"  → {out_ew_path}")
    print(f"  → {out_ns_path}")


if __name__ == "__main__":
    args = parse_args()
    regularize_displacement(args)
