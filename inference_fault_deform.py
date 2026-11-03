import os
import argparse
import torch
import numpy as np
import time
import random


from src.models.model_loader import load_model
from src.training_inference.evaluate import evaluate_inference
from src.dataloaders.data_loader import get_inference_dataloader

from src.metrics_and_losses.photometric_metrics import realEPE
from src.metrics_and_losses.gradient_metrics import L2Smoothness
from src.configs.load_config import ConfigArgumentParser, YamlConfigAction


def fix_seed(seed):
    """
    Fix the random seed for reproducibility across PyTorch, NumPy, and Python's random module.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    """
    Parse command line arguments for the inference script.
    """
    parser = argparse.ArgumentParser()

    parser = ConfigArgumentParser('inference fault deform')
    parser.add_argument('--train_config_name', default='irseparated_geoflownet_intermediatel1_noreg')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (e.g. cuda or cpu)')
    
    # Inference specific parameters
    parser.add_argument('--seed', default=1, type=int, help='Seed used for the random number generators')
    parser.add_argument('--pretrained_model_filename', help='Filename containing the pretrained weights associated witht the config')  
    
    # data loader parameters
    parser.add_argument('--split_name', type=str, default="test", help='name of the split')
    parser.add_argument('--split_start_idx', type=int, default=0, help='id of the first example')
    parser.add_argument('--split_count', type=int, default=10000, help='number of split examples')   
    parser.add_argument('--split_scaling_factors',  nargs='+', default=[0, 1, 2, 3], help="scaling factors")
    parser.add_argument('--split_dir', type=str, help='directory containing the dataset splits')
    parser.add_argument('--dataset_dir', help='path to dataset dir') 
    parser.add_argument('--dataset_name', default='faultdeform', help='dataset name')
    parser.add_argument('--image_size', default=512, type=int, help='Image size')  
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size')  

    # edge parameters
    parser.add_argument('--near_fault_only', action='store_true', help='')
    parser.add_argument('--fault_boundary_dir', default=None, help="dir containing the boundary faults images")
    parser.add_argument('--fault_boundary_disk', type=int, default=1, help='number of val examples')

    # saving parameters
    parser.add_argument('--save_metrics', action='store_true', help='')
    parser.add_argument('--eval_crop_size', type=int, default=0, help='when > 0 it is mainly for aligning with cosi-corr')
    parser.add_argument('--save_images', action='store_true', help='')
    parser.add_argument('--metric_filename', default=None, help="filename where to save the metric results") 
    parser.add_argument('--save_dir', default=None, help="dir where to save the estimated images")

    namespace, _ = parser.parse_known_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_filename = os.path.join(current_dir, "data/configs/train_fault_deform", f'{namespace.train_config_name.lower()}.yaml')
    parser.add_argument('--config', action=YamlConfigAction, default=[config_filename])

    return parser.parse_args()


def run_inference(args):
    """
    Run the inference pipeline.
    """
    # Fix the seed
    fix_seed(args.seed)

    # Define device
    device = torch.device(args.device)

    # Load models
    optical_flow_model = load_model(args, device)

    # Define metrics
    photometric_metrics = [realEPE]
    smoothness_metrics = [L2Smoothness()]

    t_start = time.time()   
    
    original_scaling_factors = args.split_scaling_factors
    for scaling_factor in original_scaling_factors:
        print(f"\nEvaluating scaling factor: {scaling_factor}")
        # Overwrite the scaling factors with just the current one so get_inference_dataloader uses only it
        args.split_scaling_factors = [scaling_factor]
        
        # Get dataloaders
        loader, frame_names, crs_meta_datas, transform_meta_datas = get_inference_dataloader(args)
    
        evaluate_inference(
            args=args,
            device=device,
            model=optical_flow_model,
            loader=loader,
            photometric_metrics=photometric_metrics,
            smoothness_metrics=smoothness_metrics,
            frame_names=frame_names,
            crs_meta_datas=crs_meta_datas,
            transform_meta_datas=transform_meta_datas
        )
    t_end = time.time()

    print(f'\n## Inference on {args.split_name}##')
    print(f"total time {t_end - t_start}")


if __name__ == '__main__':
    # Load parameters
    args = parse_args()

    run_inference(args)

