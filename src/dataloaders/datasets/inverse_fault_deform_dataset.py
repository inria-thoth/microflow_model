import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
import os
import math
import cv2
from skimage.morphology import disk
from src.dataloaders.datasets.fault_deform_dataset import FaultDeformDataset


class InverseFaultDeformDataset(FaultDeformDataset):

    def __init__(
            self, frame_ids, root_dir, transform=None, scaling_factors=None, fault_boundary_dir=None,
            fault_boundary_disk=1,
            inverse_root_dir=None,
            clean_pair_dataset_dir=None, clean_pair_prob=0.5, clean_pair_add_noise=True
    ):
        super().__init__(
            frame_ids, root_dir, transform, scaling_factors, fault_boundary_dir, fault_boundary_disk,
            clean_pair_dataset_dir, clean_pair_prob, clean_pair_add_noise
        )
        self.inverse_root_dir = inverse_root_dir

    def load_inverse_sample(self, filename):
        arrays = np.load(filename, allow_pickle=True)

        dm_ew = arrays["inverse_flow_ew"]
        dm_ns = arrays["inverse_flow_ns"]
        dm_array = np.stack([dm_ew, dm_ns]).astype(np.float32)
        dm_array = np.nan_to_num(dm_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Filter extreme outliers (e.g. GIS NoData values like 1e10 or -99999)
        dm_array = np.where(np.abs(dm_array) > 2000, 0.0, dm_array)

        return dm_array

    def __getitem__(self, idx):
        id_scalings = idx % self.len_scaling_factor
        id_frames = idx // self.len_scaling_factor

        frame_id = self.frame_ids[id_frames]
        scaling_factor = self.scaling_factors[id_scalings]

        sample_filename = os.path.join(self.root_dir, f"{frame_id:06d}_{scaling_factor}_sample.npy.npz")
        
        loaded_from_clean_pair = False
        if scaling_factor == 0 and self.clean_pair_dataset_dir is not None:
            if np.random.rand() < self.clean_pair_prob:
                clean_pair_filename = os.path.join(self.clean_pair_dataset_dir, f"{frame_id:06d}_{scaling_factor}_sample.npy.npz")
                if os.path.exists(clean_pair_filename):
                    sample_filename = clean_pair_filename
                    loaded_from_clean_pair = True
                elif self.clean_pair_prob == 1.0 and np.random.rand() < 0.05:
                    print(f"[DEBUG] Missing clean pair file: {clean_pair_filename}")

        try:
            pre_post_image_array, dm_array = self.load_sample(sample_filename)
        except Exception as e:
            if loaded_from_clean_pair:
                if not hasattr(self, 'corrupted_clean_pairs'):
                    self.corrupted_clean_pairs = 0
                self.corrupted_clean_pairs += 1
                if self.corrupted_clean_pairs > 10:
                    raise RuntimeError(f"Too many corrupted clean pair files! Last one: {sample_filename}") from e
                
                print(f"[WARNING] Corrupted clean pair file ({sample_filename}). Falling back to noisy sample. Error: {e}")
                sample_filename = os.path.join(self.root_dir, f"{frame_id:06d}_{scaling_factor}_sample.npy.npz")
                loaded_from_clean_pair = False
                pre_post_image_array, dm_array = self.load_sample(sample_filename)
            else:
                raise
        
        if loaded_from_clean_pair and self.clean_pair_add_noise and np.random.rand() < 0.5:
            # Subtle multiplicative brightness and small noise
            brightness_scale = np.random.uniform(0.95, 1.05)
            std_dev = np.std(pre_post_image_array[1]) + 1e-6
            noise = np.random.normal(0, 0.02 * std_dev, pre_post_image_array[1].shape)
            pre_post_image_array[1] = (pre_post_image_array[1] * brightness_scale) + noise

        inverse_filename = os.path.join(self.inverse_root_dir, f"{frame_id:06d}_{scaling_factor}_inverse.npy.npz")
        inverse_dm_array = self.load_inverse_sample(inverse_filename)

        max_displacement = np.abs(np.max(dm_array))

        target_dm = torch.from_numpy(inverse_dm_array)
        sample = {
            'pre_post_image': torch.from_numpy(pre_post_image_array),
            'target_dm': target_dm,
            'valid': torch.ones_like(target_dm, dtype=torch.bool)
        }

        if self.fault_boundary_dir is not None:
            fault_filename = os.path.join(self.fault_boundary_dir, f"{frame_id:06d}_fault_boundary.npy")
            fault_boundary = np.load(fault_filename)  # values are either 0 or 255
            fault_boundary = cv2.dilate(fault_boundary.astype(np.uint8),
                                        disk(self.fault_boundary_disk).astype(np.uint8))
            sample['fault_boundary'] = torch.from_numpy(fault_boundary).unsqueeze(dim=0) / 255

        if self.transform is not None:
            sample = self.transform(sample)

        sample['frame_id'] = torch.tensor(frame_id)
        sample['scaling_factor'] = torch.tensor(int(scaling_factor))
        sample['max_displacement'] = torch.tensor(int(math.ceil(max_displacement)))

        return sample
