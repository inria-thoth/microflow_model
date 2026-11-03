import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm
from src.models.model_utils.warping_utils import warp_image_torch, get_inverse_flow
import time
from src.dataloaders.data_loader import get_inference_dataloader



def parse_args():
    parser = argparse.ArgumentParser()  #
    parser.add_argument('--split_name', type=str, default=None, help='name of the split to be loaded')
    parser.add_argument('--split_start_idx', type=int, default=0, help='id of the first example')
    parser.add_argument('--split_count', type=int, default=10000, help='number of split examples')
    parser.add_argument('--split_scaling_factors', nargs='+', default=[0, 1, 2, 3], help="scaling factors")
    parser.add_argument('--split_dir', type=str, help='directory containing the dataset splits')
    parser.add_argument('--dataset_dir', help='path to dataset dir')
    parser.add_argument('--dataset_name', default='faultdeform', help='dataset name')
    parser.add_argument('--image_size', type=int, default=1024, help='size of the image')

    parser.add_argument('--num_workers', default=16, type=int, help='num workers in dataloader')
    parser.add_argument('--persistent_workers', default=True, action=argparse.BooleanOptionalAction,
                        help='activate persistent workers in dataloader')
    parser.add_argument('--pin_memory', default=True, action=argparse.BooleanOptionalAction,
                        help='activate pin memory option in dataloader')
    parser.add_argument('--non_blocking', default=True, action=argparse.BooleanOptionalAction,
                        help='activate asynchronuous GPU transfer')
    parser.add_argument('--prefetch_factor', default=3, type=int,
                        help='prefectch factor in dataloader')
    parser.add_argument('--drop_last', action='store_true', help='')

    parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size')

    # edge parameters
    parser.add_argument('--near_fault_only', action='store_true', help='')
    parser.add_argument('--fault_boundary_dir', default=None, help="dir containing the boundary faults images")
    parser.add_argument('--fault_boundary_disk', type=int, default=1, help='number of val examples')

    # saving parameters
    parser.add_argument('--save_dir', default=None, help="dir where to save the metrics")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    return args


if __name__ == '__main__':

    """ RUN:
    python generate_reverse_flow_dataset.py 
    """

    # Load parameters
    t0 = time.time()
    args = parse_args()
    device = torch.device('cuda')
    t1 = time.time()

    # Dataloader
    loader, frame_names, crs_meta_datas, transform_meta_datas = get_inference_dataloader(args)

    total_count = 0

    for i_loader, batchv in enumerate(tqdm(loader, mininterval=5.0)):
        batch_input = batchv['pre_post_image'].to(device, non_blocking=args.non_blocking)
        batch_target = batchv['target_dm'].to(device, non_blocking=args.non_blocking) if 'target_dm' in batchv else None
        if batch_target is not None:
            batch_target = torch.nan_to_num(batch_target, nan=0.0, posinf=0.0, neginf=0.0)
        batch_frame_ids = batchv['frame_id']
        batch_scaling_factors = batchv['scaling_factor'] if args.dataset_name.lower() == "faultdeform" else None

        crs_meta_data = crs_meta_datas[i_loader] if crs_meta_datas else None
        transform_meta_data = transform_meta_datas[i_loader] if transform_meta_datas else None

        inverse_flows_tensor = get_inverse_flow(batch_target)
        inverse_flows_tensor = torch.nan_to_num(inverse_flows_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        inverse_flows = inverse_flows_tensor.cpu().numpy().astype("float16")
        # warped_images = warp_image_torch(sat_images[:, 1].view(b, 1, h, w), inverse_flows)

        frame_id = batch_frame_ids[0]
        scaling_factor = batch_scaling_factors[0]
        output_filename = os.path.join(
            args.save_dir, f"{frame_id:06d}_{scaling_factor}_inverse.npy.npz")
        # print(f"output_filename {output_filename}")

        np.savez(output_filename, inverse_flow_ew=inverse_flows[0, 0], inverse_flow_ns=inverse_flows[0, 1])


    t3 = time.time()
    print(f"total_count {total_count}")



