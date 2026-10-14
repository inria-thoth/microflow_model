import os
import torch
from src.dataloaders.datasets.fault_deform_dataset import FaultDeformDataset
from src.dataloaders.datasets.prediction_dataset import PredictionDataset
from src.dataloaders.datasets.inverse_fault_deform_dataset import InverseFaultDeformDataset
from src.dataloaders.datasets.real_example_dataset import RealExampleDataset
from src.dataloaders.datasets.real_example_large_dataset import RealExampleLargeDataset
from src.dataloaders.datasets.real_example_large_dataset_onthefly import RealExampleLargeDatasetOnTheFly
from src.dataloaders.transform import get_inference_transforms, get_train_transforms, get_inference_from_estimates_transforms


def get_inference_dataloader(args):
    """
    Create and return the inference dataloader with appropriate transforms and dataset.
    """
    inference_transform = get_inference_transforms(args.image_size)

    if args.dataset_name.lower() == "faultdeform":
        frame_ids = load_frame_ids(args, split_name=args.split_name, split_start_idx=args.split_start_idx, split_count=args.split_count)
        inference_frame_ids = [int(frame_id) for frame_id in frame_ids]
        inference_set = FaultDeformDataset(
            frame_ids=inference_frame_ids,
            root_dir=os.path.join(args.dataset_dir, args.split_name),
            transform=inference_transform,
            scaling_factors=args.split_scaling_factors,
            fault_boundary_dir=args.fault_boundary_dir,
            fault_boundary_disk=args.fault_boundary_disk
        )
        crop_names, crs_meta_datas, transform_meta_datas = None, None, None
    elif args.dataset_name.lower() == "real_examples":
        crop_names = [f for f in os.listdir(args.dataset_dir) if f.startswith("corr_landsat8_01")]
        inference_set = RealExampleDataset(
            args.dataset_dir,
            transform=inference_transform,
            pre_template="pre.tif",
            post_template="post.tif",
            crop_names=crop_names
        )
        crs_meta_datas, transform_meta_datas = inference_set.get_tiff_metadata()
    elif args.dataset_name.lower() == "estimate_faultdeform":
        inference_transform = get_inference_from_estimates_transforms(args.image_size)
        frame_ids = [f.split("_")[0] for f in os.listdir(args.estimation_root_dir)] 
        inference_frame_ids = [int(frame_id) for frame_id in frame_ids]
        inference_set = PredictionDataset(
            frame_ids=inference_frame_ids, 
            fault_deform_root_dir=os.path.join(args.dataset_dir, args.split_name), 
            estimation_root_dir=args.estimation_root_dir, 
            transform=inference_transform, 
            scaling_factors=args.split_scaling_factors,
            fault_boundary_dir=args.fault_boundary_dir,
            fault_boundary_disk=args.fault_boundary_disk
        )
        crop_names, crs_meta_datas, transform_meta_datas = None, None, None
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported")

    inference_loader = torch.utils.data.DataLoader(
        dataset=inference_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        drop_last=False)

    return inference_loader, crop_names, crs_meta_datas, transform_meta_datas


def get_real_examples_dataloader(args):
    """
    Create and return the inference dataloader for large real satellite images.
    Supports sliding-window inference on large tiffs.
    """
    normalization = getattr(args, 'normalization', 'minmax')
    print(args.dataset_name.lower())
    if args.dataset_name.lower() == "real_examples_onthefly":
        inference_transform = get_inference_transforms(args.window_size, normalization)
        inference_set = RealExampleLargeDatasetOnTheFly(
            args.dataset_dir,
            transform=inference_transform,
            pre_template="pre.tif",
            post_template="post.tif",
            top=0,
            left=0,
            window_size=args.window_size,
            window_overlap=args.stride
        )
    else:
        # Default: use the original preloaded dataset
        inference_transform = get_inference_transforms(args.window_size, normalization)
        inference_set = RealExampleLargeDataset(
            args.dataset_dir,
            transform=inference_transform,
            pre_template="pre.tif",
            post_template="post.tif",
            top=0,
            left=0,
            window_size=args.window_size,
            window_overlap=args.stride
        )

    # Metadata is loaded outside the dataloader (rasterio format, not tensor-compatible)
    crs_meta_datas, transform_meta_datas = inference_set.get_tiff_metadata()

    inference_loader = torch.utils.data.DataLoader(
        dataset=inference_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        drop_last=False)

    return inference_loader, crs_meta_datas, transform_meta_datas


def get_training_dataloaders(args):
    train_split_name = "train"
    train_frame_ids = load_frame_ids(args, split_name=train_split_name, split_start_idx=0, split_count=args.train_count)
    train_transform = get_train_transforms(args.train_image_size)
    if "inverse" in args.model_name.lower():
        train_set = InverseFaultDeformDataset(
            frame_ids=train_frame_ids,
            root_dir=os.path.join(args.dataset_dir, train_split_name),
            transform=train_transform,
            scaling_factors=args.train_scaling_factors,
            inverse_root_dir=os.path.join(args.inverse_dataset_dir, train_split_name),
            clean_pair_dataset_dir=os.path.join(args.clean_pair_dataset_dir, train_split_name) if getattr(args, 'clean_pair_dataset_dir', None) else None,
            clean_pair_prob=getattr(args, 'clean_pair_prob', 0.5),
            clean_pair_add_noise=getattr(args, 'clean_pair_add_noise', True)
        )
    else:
        train_set = FaultDeformDataset(
            frame_ids=train_frame_ids,
            root_dir=os.path.join(args.dataset_dir, train_split_name),
            transform=train_transform,
            scaling_factors=args.train_scaling_factors,
            clean_pair_dataset_dir=os.path.join(args.clean_pair_dataset_dir, train_split_name) if getattr(args, 'clean_pair_dataset_dir', None) else None,
            clean_pair_prob=getattr(args, 'clean_pair_prob', 0.5),
            clean_pair_add_noise=getattr(args, 'clean_pair_add_noise', True)
        )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        drop_last=args.drop_last)

    val_split_name = "validation"
    val_frame_ids = load_frame_ids(args, split_name=val_split_name, split_start_idx=0, split_count=args.val_count)
    validation_transform = get_inference_transforms(args.train_image_size)
    if "inverse" in args.model_name.lower():
        val_set = InverseFaultDeformDataset(
            frame_ids=val_frame_ids,
            root_dir=os.path.join(args.dataset_dir, val_split_name),
            transform=validation_transform,
            scaling_factors=args.val_scaling_factors,
            inverse_root_dir=os.path.join(args.inverse_dataset_dir, val_split_name)
        )
    else:
        val_set = FaultDeformDataset(
            frame_ids=val_frame_ids,
            root_dir=os.path.join(args.dataset_dir, val_split_name),
            transform=validation_transform,
            scaling_factors=args.val_scaling_factors,
        )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        drop_last=args.drop_last)
    clean_pair_val_loader = None
    if getattr(args, 'run_clean_pair_val', False) and getattr(args, 'clean_pair_dataset_dir', None):
        if "inverse" in args.model_name.lower():
            clean_pair_val_set = InverseFaultDeformDataset(
                frame_ids=val_frame_ids,
                root_dir=os.path.join(args.dataset_dir, val_split_name),
                transform=validation_transform,
                scaling_factors=args.val_scaling_factors,
                inverse_root_dir=os.path.join(args.inverse_dataset_dir, val_split_name),
                clean_pair_dataset_dir=os.path.join(args.clean_pair_dataset_dir, val_split_name),
                clean_pair_prob=1.0,
                clean_pair_add_noise=False
            )
        else:
            clean_pair_val_set = FaultDeformDataset(
                frame_ids=val_frame_ids,
                root_dir=os.path.join(args.dataset_dir, val_split_name),
                transform=validation_transform,
                scaling_factors=args.val_scaling_factors,
                clean_pair_dataset_dir=os.path.join(args.clean_pair_dataset_dir, val_split_name),
                clean_pair_prob=1.0,
                clean_pair_add_noise=False
            )
        clean_pair_val_loader = torch.utils.data.DataLoader(
            dataset=clean_pair_val_set,
            batch_size=args.train_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            drop_last=args.drop_last)

    print('{} train samples and {} validation samples'.format(len(train_set), len(val_set)))
    return train_loader, val_loader, clean_pair_val_loader


def load_frame_ids(args, split_name="test", split_start_idx=0, split_count=-1):
    """
    Load frame IDs corresponding to the dataset split file.
    """
    split_filename = os.path.join(args.split_dir, f"{split_name}_names_center.txt" if ("near_fault_only" in args and args.near_fault_only) else f"{split_name}_names.txt")
    with open(split_filename, "r") as reader:
        split_names = [int(name.strip().split("_")[0]) for name in reader]
   
    if split_count != -1:
        frame_ids = split_names[split_start_idx:split_start_idx + split_count]
    else:
        frame_ids = split_names[split_start_idx:]

    return frame_ids


