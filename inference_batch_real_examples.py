import torch
import os
import glob
from inference_real_examples import parse_args, fix_seed
from src.models.model_loader import load_model
from src.training_inference.evaluate import evaluate_inference_large_image
from src.dataloaders.data_loader import get_real_examples_dataloader

def run_batch_inference(args):
    """
    Run the inference pipeline on multiple subdirectories starting with 'corr_'.
    Loads the model only once for efficiency.
    """
    # Fix the seed
    fix_seed(args.seed)

    # Define device
    device = torch.device(args.device)

    print(args)

    # Load model ONCE for all folders
    print(f"Loading model from {args.pretrained_model_filename}...")
    optical_flow_model = load_model(args, device)
    print("Model loaded successfully.")

    parent_dir = args.dataset_dir
    base_save_dir = args.save_dir
    
    # Find all corr_ folders
    corr_folders = glob.glob(os.path.join(parent_dir, "corr_*"))
    corr_folders = [f for f in corr_folders if os.path.isdir(f)]
    corr_folders.sort()
    
    if not corr_folders:
        print(f"No 'corr_*' folders found in {parent_dir}")
        return
        
    print(f"Found {len(corr_folders)} folders to process.")
    
    for folder in corr_folders:
        folder_name = os.path.basename(folder)
        print(f"\n{'='*50}")
        print(f"Processing folder: {folder_name}")
        print(f"{'='*50}")
        
        # Override args for this specific folder
        args.dataset_dir = folder
        if base_save_dir:
            args.save_dir = os.path.join(base_save_dir, folder_name)
            os.makedirs(args.save_dir, exist_ok=True)
        else:
            args.save_dir = None
            
        # Get dataloaders for this specific folder
        loader, crs_meta_datas, transform_meta_datas = get_real_examples_dataloader(args)

        # Run inference
        evaluate_inference_large_image(
            args=args,
            device=device,
            model=optical_flow_model,
            loader=loader,
            crs_meta_datas=crs_meta_datas,
            transform_meta_datas=transform_meta_datas,
        )

if __name__ == '__main__':
    # Load parameters using the same parser as inference_real_examples.py
    args = parse_args()
    
    # The user should provide the PARENT directory to --dataset_dir
    # and the BASE results directory to --save_dir
    run_batch_inference(args)
