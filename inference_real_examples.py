import argparse
import torch
import numpy as np
import os
import random

from src.models.model_loader import load_model
from src.training_inference.evaluate import evaluate_inference_large_image
from src.dataloaders.data_loader import get_real_examples_dataloader
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
    Parse command line arguments for estimating the displacement fields for large images.
    """
    parser = ConfigArgumentParser('inference real examples')
    parser.add_argument('--config_name', default='microflow')

    # Inference specific parameters
    parser.add_argument('--seed', default=1, type=int, help='Seed used for the random number generators')
    parser.add_argument('--pretrained_model_filename', help='Filename containing the pretrained weights associated with the config')
    parser.add_argument('--dataset_dir', help='path to the dir containing the real examples')
    parser.add_argument('--save_dir', default=None, help="dir where to save the flow estimates")
    parser.add_argument('--dataset_name', default='real_examples_onthefly', type=str,
                        help='Dataset type to use for inference')

    parser.add_argument('--window_size', default=1024, type=int, help='Sliding window size (useful when the image is large)')
    parser.add_argument('--stride', default=0, type=int, help='Stride for the sliding window')
    parser.add_argument('--offset', default=None, type=int, help='Offset to remove pixels on the boundaries of each window.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run inference on (cuda or cpu)')

    parser.add_argument('-b', '--batch-size', default=1, type=int, help='batch size for estimating on the sliding windows')

    # ---- ITERATIVE_MODEL params ----
    parser.add_argument('--normalization', default='minmax', type=str,
                        help='Normalization method: minmax or standard')
    parser.add_argument('--apply_noise', default=False, action=argparse.BooleanOptionalAction,
                        help='Apply noise during inference (test feature, keep False)')

    # ---- BASELINE_MODELS params ----
    parser.add_argument('--renormalize', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--detach_gradients', default=True, action=argparse.BooleanOptionalAction,
                        help='Detach gradients between iterative model steps')
    parser.add_argument('--searaft_freeze', default=False, action=argparse.BooleanOptionalAction,
                        help='Freeze SEA-RAFT backbone weights')

    # ---- SEA-RAFT specific params ----
    parser.add_argument('--searaft_block_dims', nargs='+', type=int, default=[64, 128, 256])
    parser.add_argument('--searaft_dim', default=128, type=int)
    parser.add_argument('--searaft_initial_dim', default=64, type=int)
    parser.add_argument('--searaft_max_iterations', default=4, type=int)
    parser.add_argument('--searaft_num_blocks', default=2, type=int)
    parser.add_argument('--searaft_pretrain', default='resnet34', type=str)
    parser.add_argument('--searaft_radius', default=4, type=int)
    parser.add_argument('--searaft_use_var', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--searaft_var_max', default=10, type=float)
    parser.add_argument('--searaft_var_min', default=0, type=float)
    parser.add_argument('--upsample_learned', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--repeat_first_iteration', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--repeat_model', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--resnet_imagenet_pretrained', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--resnet_input_dim', default=6, type=int)
    parser.add_argument('--gamma_scale', default=0.85, type=float)
    parser.add_argument('--max_scale', default=1, type=int)

    namespace, _ = parser.parse_known_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_filename = os.path.join(current_dir, "data/configs/inference_real_examples/", f'{namespace.config_name.lower()}.yaml')
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

    print(args)

    # Load models
    optical_flow_model = load_model(args, device)

    # Get dataloaders
    loader, crs_meta_datas, transform_meta_datas = get_real_examples_dataloader(args)

    evaluate_inference_large_image(
        args=args,
        device=device,
        model=optical_flow_model,
        loader=loader,
        crs_meta_datas=crs_meta_datas,
        transform_meta_datas=transform_meta_datas,
        )


if __name__ == '__main__':

    # Load parameters
    args = parse_args()

    run_inference(args)
