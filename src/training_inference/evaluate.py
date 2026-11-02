import os
from tqdm import tqdm
from collections import defaultdict
import torch
import numpy as np
import time
import gc
from src.dataloaders.tiff_utils import save_array_to_tiff
from src.models.model_utils.iterative_model_utils import compute_regularization
import matplotlib.pyplot as plt


def evaluate_training(args, device, model, val_loader=None, loss_function=None, metrics=None):
    model.eval()
    val_dataset_count = len(val_loader.dataset)
    val_batches_count = val_dataset_count / args.train_batch_size

    val_loss = 0
    val_errors = [0 for _ in metrics]
    val_errors_steps = [[0 for _ in metrics] for _ in range(model.max_iterations)]
    val_errors_by_scaling_factor = {0: defaultdict(int), 1: defaultdict(int), 2: defaultdict(int), 3: defaultdict(int)}

    for _, batchv in enumerate(tqdm(val_loader, mininterval=5.0)):
        val_input = batchv['pre_post_image'].to(device, non_blocking=args.non_blocking)
        val_target = batchv['target_dm'].to(device, non_blocking=args.non_blocking)
        images_no_normalization = batchv['pre_post_image_no_normalization'].to(device, non_blocking=args.non_blocking)
        scaling_factors = batchv['scaling_factor']
        b, c, h, w = val_input.shape
        input_model = torch.cat(
            [val_input, images_no_normalization[:, 1].view(b, 1, h, w), images_no_normalization[:, 0].view(b, 1, h, w)],
            dim=1)
        with torch.no_grad():
            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=device_type, enabled=(args.amp and device_type == 'cuda'), dtype=amp_dtype):
                optical_flow_predictions = model(input_model, args=args)

                if loss_function is not None:
                    val_outputs = optical_flow_predictions if args.loss == "raft" else optical_flow_predictions[-1]
                    loss = loss_function(val_outputs, val_target)
                    val_loss += loss.item() / val_batches_count

                for metric_id, metric in enumerate(metrics):
                    # global metric
                    update_full_image_error(
                        metric,
                        optical_flow_predictions[-1],
                        val_target,
                        val_errors,
                        metric_id,
                        val_batches_count
                    )

                    # metric at each step
                    for step in range(len(optical_flow_predictions)):
                        update_full_image_error(
                            metric,
                            optical_flow_predictions[step],
                            val_target,
                            val_errors_steps[step],
                            metric_id,
                            val_batches_count
                        )

                    # metric at each scaling factor
                    b, c, h, w = val_target.shape
                    for sf in range(4):  # For scaling factors 0,1,2,3
                        mask = (scaling_factors == sf)
                        if not mask.any():
                            continue
                        update_full_image_error(
                            metric,
                            optical_flow_predictions[-1][mask],
                            val_target[mask],
                            val_errors_by_scaling_factor[sf],
                            metric_id,
                            val_dataset_count / len(scaling_factors) / sum(mask)
                            # done per scaling factor, so the count needs to be adjusted
                        )

    return val_loss, val_errors, val_errors_steps, val_errors_by_scaling_factor


def evaluate_inference(
        args, device, model, loader=None, photometric_metrics=None, smoothness_metrics=None,
        frame_names=None, crs_meta_datas=None, transform_meta_datas=None
):
    """
    Evaluate the model using photometric (EPE) and smoothness metrics.
    """
    model.eval()

    val_dataset_count = len(loader.dataset)
    val_batches_count = val_dataset_count / args.batch_size

    epe_errors = [[0 for _ in photometric_metrics] for _ in range(model.max_iterations)]
    epe_errors_near_fault = [[0 for _ in photometric_metrics] for _ in range(model.max_iterations)]  # Not used for now
    epe_errors_away_fault = [[0 for _ in photometric_metrics] for _ in range(model.max_iterations)]  # Not used for now
    smoothness_errors = [[0 for _ in range(2 * len(smoothness_metrics))] for _ in range(model.max_iterations)]
    smoothness_errors_near_fault = [[0 for _ in range(2 * len(smoothness_metrics))] for _ in
                                    range(model.max_iterations)]
    smoothness_errors_away_fault = [[0 for _ in range(2 * len(smoothness_metrics))] for _ in
                                    range(model.max_iterations)]

    if args.regularization:
        import prox_tv as ptv
    else:
        ptv = None

    for i_loader, batchv in enumerate(tqdm(loader)):
        batch_input = batchv['pre_post_image'].to(device, non_blocking=args.non_blocking)
        batch_target = batchv['target_dm'].to(device, non_blocking=args.non_blocking) if 'target_dm' in batchv else None
        batch_near_fault = batchv['fault_boundary'].to(device,
                                                       non_blocking=args.non_blocking) if args.fault_boundary_dir is not None else None
        batch_images_no_normalization = batchv['pre_post_image_no_normalization'].to(device,
                                                                                     non_blocking=args.non_blocking)
        batch_frame_ids = batchv['frame_id']
        batch_scaling_factors = batchv['scaling_factor'] if args.dataset_name.lower() == "faultdeform" else None

        crs_meta_data = crs_meta_datas[i_loader] if crs_meta_datas else None
        transform_meta_data = transform_meta_datas[i_loader] if transform_meta_datas else None

        if args.model_name.lower() == "raft":
            input_model = batch_input
        else:
            b, c, h, w = batch_input.shape
            input_model = torch.cat([batch_input, batch_images_no_normalization[:, 1].view(b, 1, h, w),
                                     batch_images_no_normalization[:, 0].view(b, 1, h, w)], dim=1)

        frame_labels = get_frame_labels(batch_frame_ids, batch_scaling_factors, args.dataset_name,
                                        frame_names=frame_names)

        with torch.no_grad():
            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=device_type, enabled=(args.amp and device_type == 'cuda'), dtype=amp_dtype):
                optical_flow_predictions = model.forward(
                    input_model,
                    ptv=ptv,
                    args=args,
                    save=args.save_images,
                    frame_labels=frame_labels,
                    crs_meta_data=crs_meta_data,
                    transform_meta_data=transform_meta_data
                )

                if args.save_metrics:
                    if batch_target is None:
                        raise error("To compute the metrics, you need to define the groundtruth / target flow")

                    # The metrics are computed on the cropped image to avoid border effects with the cosi-corr baseline
                    crop_slice = slice(args.eval_crop_size, -args.eval_crop_size or None)
                    batch_target_cropped = batch_target[:, :, crop_slice, crop_slice]

                    for metric_id, metric in enumerate(photometric_metrics):
                        for iteration in range(model.max_iterations):
                            iteration_prediction = optical_flow_predictions[iteration]
                            iteration_prediction_cropped = iteration_prediction[:, :, crop_slice, crop_slice]
                            
                            if args.near_fault_only:
                                batch_near_faults_cropped = batch_near_fault[:, :, crop_slice, crop_slice]
                                error, error_near_fault, error_away_fault = metric(
                                    iteration_prediction_cropped, 
                                    batch_target_cropped, 
                                    edge_mask=batch_near_faults_cropped
                                )
                                epe_errors[iteration][metric_id] += error.item() / val_batches_count
                                epe_errors_near_fault[iteration][metric_id] += error_near_fault.item() / val_batches_count
                                epe_errors_away_fault[iteration][metric_id] += error_away_fault.item() / val_batches_count
                            else:
                                error = metric(iteration_prediction_cropped, batch_target_cropped)
                                epe_errors[iteration][metric_id] += error.item() / val_batches_count

                    if args.near_fault_only:
                        batch_near_faults_cropped = batch_near_fault[:, :, crop_slice, crop_slice]

                        for metric_id, metric in enumerate(smoothness_metrics):
                            for iteration in range(model.max_iterations):
                                iteration_prediction = optical_flow_predictions[iteration]
                                iteration_prediction_cropped = iteration_prediction[:, :, crop_slice, crop_slice]

                                # Smoothness errors for prediction
                                update_errors(
                                    metric=metric,
                                    prediction=iteration_prediction_cropped,
                                    target=batch_near_faults_cropped,
                                    errors=smoothness_errors[iteration],
                                    errors_near_fault=smoothness_errors_near_fault[iteration],
                                    errors_away_fault=smoothness_errors_away_fault[iteration],
                                    index=2 * metric_id,
                                    val_dataset_count=val_batches_count
                                )

                                # Smoothness errors for ground truth / reference
                                update_errors(
                                    metric=metric,
                                    prediction=batch_target_cropped,
                                    target=batch_near_faults_cropped,
                                    errors=smoothness_errors[iteration],
                                    errors_near_fault=smoothness_errors_near_fault[iteration],
                                    errors_away_fault=smoothness_errors_away_fault[iteration],
                                    index=2 * metric_id + 1,
                                    val_dataset_count=val_batches_count
                                )

    if args.save_metrics:
        model_name = args.pretrained_model_filename.split("/")[-1]
        scaling_factor_name = "_".join(args.split_scaling_factors)
        regularization_name = f"_{args.penalty_function}_l{args.reg_lambda}_i{args.reg_iterations}" if args.regularization else ""

        metrics = compute_all_metrics(
            args, model.max_iterations, epe_errors, smoothness_errors, smoothness_errors_near_fault,
            smoothness_errors_away_fault, epe_errors_near_fault, epe_errors_away_fault
        )

        save_metrics(args, model_name, scaling_factor_name, regularization_name, metrics)


def evaluate_inference_from_estimates(
        args, device, loader=None, photometric_metrics=None, smoothness_metrics=None,
        frame_names=None, crs_meta_datas=None, transform_meta_datas=None
):
    """
    Evaluate the model using photometric (EPE) and smoothness metrics.
    """
    val_dataset_count = len(loader.dataset)
    val_batches_count = val_dataset_count / args.batch_size

    epe_errors = [[0 for _ in photometric_metrics] for _ in range(1)]
    epe_errors_near_fault = [[0 for _ in photometric_metrics] for _ in range(1)]  # Not used for now
    epe_errors_away_fault = [[0 for _ in photometric_metrics] for _ in range(1)]  # Not used for now
    smoothness_errors = [[0 for _ in range(2 * len(smoothness_metrics))] for _ in range(1)]
    smoothness_errors_near_fault = [[0 for _ in range(2 * len(smoothness_metrics))] for _ in range(1)]
    smoothness_errors_away_fault = [[0 for _ in range(2 * len(smoothness_metrics))] for _ in range(1)]

    for _, batchv in enumerate(tqdm(loader)):
        optical_flow_predictions = batchv['estimated_dm'].to(device, non_blocking=args.non_blocking)
        batch_target = batchv['target_dm'].to(device, non_blocking=args.non_blocking) if 'target_dm' in batchv else None
        batch_near_fault = batchv['fault_boundary'].to(device,
                                                       non_blocking=args.non_blocking) if args.fault_boundary_dir is not None else None

        if args.save_metrics:
            if batch_target is None:
                raise error("To compute the metrics, you need to define the groundtruth / target flow")

        # The metrics are computed on the cropped image to avoid border effects with the cosi-corr baseline
        crop_slice = slice(args.eval_crop_size, -args.eval_crop_size or None)
        batch_target_cropped = batch_target[:, :, crop_slice, crop_slice]

        for metric_id, metric in enumerate(photometric_metrics):
            for iteration in range(1):
                iteration_prediction = optical_flow_predictions
                iteration_prediction_cropped = iteration_prediction[:, :, crop_slice, crop_slice]
                
                if args.near_fault_only:
                    batch_near_faults_cropped = batch_near_fault[:, :, crop_slice, crop_slice]
                    error, error_near_fault, error_away_fault = metric(
                        iteration_prediction_cropped, 
                        batch_target_cropped, 
                        edge_mask=batch_near_faults_cropped
                    )
                    epe_errors[iteration][metric_id] += error.item() / val_batches_count
                    epe_errors_near_fault[iteration][metric_id] += error_near_fault.item() / val_batches_count
                    epe_errors_away_fault[iteration][metric_id] += error_away_fault.item() / val_batches_count
                else:
                    error = metric(iteration_prediction_cropped, batch_target_cropped)
                    epe_errors[iteration][metric_id] += error.item() / val_batches_count

        if args.near_fault_only:
            batch_near_faults_cropped = batch_near_fault[:, :, crop_slice, crop_slice]

            for metric_id, metric in enumerate(smoothness_metrics):
                for iteration in range(1):
                    iteration_prediction = optical_flow_predictions
                    iteration_prediction_cropped = iteration_prediction[:, :, crop_slice, crop_slice]

                    # Smoothness errors for prediction
                    update_errors(
                        metric=metric,
                        prediction=iteration_prediction_cropped,
                        target=batch_near_faults_cropped,
                        errors=smoothness_errors[iteration],
                        errors_near_fault=smoothness_errors_near_fault[iteration],
                        errors_away_fault=smoothness_errors_away_fault[iteration],
                        index=2 * metric_id,
                        val_dataset_count=val_batches_count
                    )

                    # Smoothness errors for ground truth / reference
                    update_errors(
                        metric=metric,
                        prediction=batch_target_cropped,
                        target=batch_near_faults_cropped,
                        errors=smoothness_errors[iteration],
                        errors_near_fault=smoothness_errors_near_fault[iteration],
                        errors_away_fault=smoothness_errors_away_fault[iteration],
                        index=2 * metric_id + 1,
                        val_dataset_count=val_batches_count
                    )

    if args.save_metrics:
        model_name = args.estimate_model_name
        scaling_factor_name = "_".join(args.split_scaling_factors)
        regularization_name = f"_{args.penalty_function}_l{args.reg_lambda}_i{args.reg_iterations}" if args.regularization else ""

        metrics = compute_all_metrics(
            args, 1, epe_errors, smoothness_errors, smoothness_errors_near_fault,
            smoothness_errors_away_fault, epe_errors_near_fault, epe_errors_away_fault
        )

        save_metrics(args, model_name, scaling_factor_name, regularization_name, metrics)


def evaluate_inference_large_image(
        args, device, model, loader=None, crs_meta_datas=None, transform_meta_datas=None
):
    """
    Inference for large images and save the
    """
    model.eval()

    reg_backend = getattr(args, 'reg_backend', 'gpu')
    if args.regularization and reg_backend == 'proxtv':
        try:
            import prox_tv as ptv
        except ImportError:
            print("Warning: prox_tv not installed. Falling back to GPU regularization backend.")
            reg_backend = 'gpu'
            args.reg_backend = 'gpu'
            ptv = None
    else:
        ptv = None

    height, width, window_overlap, window_size, image_pair_name = loader.dataset.get_image_info()

    offset_max = int(np.min([window_size / 2, window_overlap / 2]))
    if args.offset is None:
        offset = offset_max
    else:
        offset = int(np.min([args.offset, offset_max]))
    print('Removing boundary pixels, with offset = ', offset)

    full_optical_flow = np.zeros((2, height, width), dtype=np.float32)
    counts_image = np.zeros((height, width), dtype=np.uint16)

    # For the large images, we disable the regularization for each patch and apply the regularization at the end
    apply_regularization = args.regularization
    args.regularization = False

    t0 = time.time()
    count = 0
    t_data, t_transfer, t_forward, t_cpu = 0.0, 0.0, 0.0, 0.0
    t_iter_start = time.time()

    for _, batchv in enumerate(tqdm(loader)):
        t_data += time.time() - t_iter_start

        ta = time.time()
        batch_input = batchv['pre_post_image'].to(device, non_blocking=args.non_blocking)
        batch_images_no_normalization = batchv['pre_post_image_no_normalization'].to(device,
                                                                                     non_blocking=args.non_blocking)
        batch_x_positions = batchv['x_position']
        batch_y_positions = batchv['y_position']
        nan_mask = batchv['nan_mask'].unsqueeze(1).expand_as(batch_input)

        if torch.sum(torch.abs(batch_input)) == 0:  # for regions where there is no data
            count += 1
            t_iter_start = time.time()
            continue

        if args.model_name.lower() == "raft":
            input_model = batch_input
        else:
            b, c, h, w = batch_input.shape
            input_model = torch.cat([batch_input, batch_images_no_normalization[:, 1].view(b, 1, h, w),
                                     batch_images_no_normalization[:, 0].view(b, 1, h, w)], dim=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_transfer += time.time() - ta

        tb = time.time()
        with torch.no_grad():
            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=device_type, enabled=(args.amp and device_type == 'cuda'), dtype=amp_dtype):
                result = model(
                    input_model,
                    ptv=ptv,
                    args=args,
                    save=False,
                    test_mode=False
                )
                # SeaRAFT returns (flow_predictions, nf_predictions); iterative models return a plain list
                if args.model_name.lower().startswith("searaft"):
                    optical_flow_predictions, _ = result
                else:
                    optical_flow_predictions = result
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_forward += time.time() - tb

        tc = time.time()
        # Cast to float32 before numpy conversion because numpy doesn't natively support bfloat16
        last_iteration_prediction = optical_flow_predictions[-1].cpu().to(torch.float32).numpy()
        last_iteration_prediction[nan_mask] = np.nan

        for i_batch in range(last_iteration_prediction.shape[0]):
            x_pos = batch_x_positions[i_batch].item()
            min_x = x_pos
            max_x = x_pos + window_size
            y_pos = batch_y_positions[i_batch].item()
            min_y = y_pos
            max_y = y_pos + window_size

            # Calculate actual bounds within the full image
            valid_min_x = max(0, min_x + offset)
            valid_max_x = min(width, max_x - offset)
            valid_min_y = max(0, min_y + offset)
            valid_max_y = min(height, max_y - offset)

            if valid_min_x >= valid_max_x or valid_min_y >= valid_max_y:
                continue

            # Calculate the relative slicing for the patch prediction
            pred_min_x = valid_min_x - min_x
            pred_max_x = valid_max_x - min_x
            pred_min_y = valid_min_y - min_y
            pred_max_y = valid_max_y - min_y

            # Add prediction to the valid area
            patch_pred = last_iteration_prediction[i_batch, :, pred_min_y:pred_max_y, pred_min_x:pred_max_x]
            
            # Ensure shape matches in case the prediction tensor was smaller (like in-memory fallback truncation)
            h_slice = min(valid_max_y - valid_min_y, patch_pred.shape[1])
            w_slice = min(valid_max_x - valid_min_x, patch_pred.shape[2])

            full_optical_flow[:, valid_min_y:valid_min_y+h_slice, valid_min_x:valid_min_x+w_slice] += patch_pred[:, :h_slice, :w_slice]
            counts_image[valid_min_y:valid_min_y+h_slice, valid_min_x:valid_min_x+w_slice] += 1
        t_cpu += time.time() - tc

        t_iter_start = time.time()

    t1 = time.time()
    print(f"load data {t1 - t0:.2f}s  count no data={count}")
    print(f"  DataLoader wait : {t_data:.2f}s")
    print(f"  GPU transfer    : {t_transfer:.2f}s")
    print(f"  Model forward   : {t_forward:.2f}s  ← GPU compute")
    print(f"  CPU copy+accumulate: {t_cpu:.2f}s")

    # In-place normalization to avoid memory spikes
    valid_mask = counts_image > 0
    counts_image[~valid_mask] = 1
    full_optical_flow[0] /= counts_image
    full_optical_flow[1] /= counts_image
    full_optical_flow[:, ~valid_mask] = np.nan
    del counts_image
    del valid_mask
    gc.collect()

    flow_to_enumerate = [full_optical_flow]

    for i_of, of in enumerate(flow_to_enumerate):
        _, h, w = of.shape

        os.makedirs(args.save_dir, exist_ok=True)
        config_name = args.config_name

        # Save raw unregularized flow if regularization is applied
        save_raw = getattr(args, 'save_raw', True)
        if apply_regularization and save_raw:
            raw_ew_filename = f"{image_pair_name}_{config_name}_raw_ew_i{i_of}.tif".replace("__", "_")
            raw_ns_filename = f"{image_pair_name}_{config_name}_raw_ns_i{i_of}.tif".replace("__", "_")
            
            # Apply QGIS convention (-1 for EW direction) on copy
            save_array_to_tiff(
                -of[0],
                os.path.join(args.save_dir, raw_ew_filename), transform=transform_meta_datas, crs=crs_meta_datas
            )
            save_array_to_tiff(
                of[1],
                os.path.join(args.save_dir, raw_ns_filename), transform=transform_meta_datas, crs=crs_meta_datas
            )
            print(f"raw_ew_filename {os.path.join(args.save_dir, raw_ew_filename)}")
            print(f"raw_ns_filename {os.path.join(args.save_dir, raw_ns_filename)}")

        t0 = time.time()
        if apply_regularization:
            reg_backend = getattr(args, 'reg_backend', 'gpu')
            chunk_size = getattr(args, 'reg_chunk_size', 2048 if reg_backend == 'gpu' else 10000)
            _, h, w = of.shape
            of[np.isnan(of)] = 0

            for i_h in range(h // chunk_size + 1):
                for i_w in range(w // chunk_size + 1):
                    start_h = i_h * chunk_size
                    end_h = min((i_h + 1) * chunk_size, h)
                    start_w = i_w * chunk_size
                    end_w = min((i_w + 1) * chunk_size, w)
                    if start_h >= end_h or start_w >= end_w:
                        continue

                    chunk = torch.tensor(of[None, :, start_h:end_h, start_w:end_w], dtype=torch.float32, device=device)
                    of[:, start_h:end_h, start_w:end_w] = compute_regularization(
                        chunk, args, ptv)[0].cpu().numpy()
        t1 = time.time()
        print(f"time regularization ({reg_backend}) {t1 - t0:.2f}s")

        # Save the full images (regularized if apply_regularization, otherwise raw)
        print(f"args.save_dir {args.save_dir}")
        of[0] *= -1  # Qgis convention for the EW direction (in-place)

        ew_filename = f"{image_pair_name}_{config_name}_ew_i{i_of}.tif".replace("__", "_")
        save_array_to_tiff(
            of[0],
            os.path.join(args.save_dir, ew_filename), transform=transform_meta_datas, crs=crs_meta_datas
        )

        ns_filename = f"{image_pair_name}_{config_name}_ns_i{i_of}.tif".replace("__",
                                                                                "_")  # replace __ in case the pre filename already contains the "_" char
        save_array_to_tiff(
            of[1].astype(np.float32),
            os.path.join(args.save_dir, ns_filename), transform=transform_meta_datas, crs=crs_meta_datas
        )

        print(f"ew_filename {os.path.join(args.save_dir, ew_filename)}")
        print(f"ns_filename {os.path.join(args.save_dir, ns_filename)}")


def update_full_image_error(metric, prediction, target, errors, index, val_dataset_count):
    """
    Update error metrics for global region.
    """
    error = metric(prediction, target)
    errors[index] += error.item() / val_dataset_count


def update_errors(metric, prediction, target, errors, errors_near_fault, errors_away_fault, index, val_dataset_count):
    """
    Update error metrics for global, near_fault and non-near_fault regions.
    """
    error, error_near_fault, error_away_fault = metric(prediction, target)
    errors[index] += error.item() / val_dataset_count
    errors_near_fault[index] += error_near_fault.item() / val_dataset_count
    errors_away_fault[index] += error_away_fault.item() / val_dataset_count


def compute_all_metrics(args, max_iterations, epe_errors, smoothness_errors, smoothness_errors_near_fault,
                        smoothness_errors_away_fault, epe_errors_near_fault, epe_errors_away_fault):
    """
    Compute evaluation metrics (EPE and smoothness) for global, near_fault and non-near_fault regions.
    """
    epe = compute_metrics(epe_errors, max_iterations, 0)

    if args.near_fault_only:
        epe_near_fault = compute_metrics(epe_errors_near_fault, max_iterations, 0)
        epe_away_fault = compute_metrics(epe_errors_away_fault, max_iterations, 0)

        l2 = compute_metrics(smoothness_errors, max_iterations, 0)
        l2_near_fault = compute_metrics(smoothness_errors_near_fault, max_iterations, 0)
        l2_away_fault = compute_metrics(smoothness_errors_away_fault, max_iterations, 0)

        l2_gt = compute_metrics(smoothness_errors, max_iterations, 1)
        l2_gt_near_fault = compute_metrics(smoothness_errors_near_fault, max_iterations, 1)
        l2_gt_away_fault = compute_metrics(smoothness_errors_away_fault, max_iterations, 1)
    else:
        # Initialize all metrics to None when not computing near_fault metrics
        epe_near_fault = epe_away_fault = None
        l2 = l2_near_fault = l2_away_fault = None
        l2_gt = l2_gt_near_fault = l2_gt_away_fault = None

    return [epe, epe_near_fault, epe_away_fault, l2, l2_near_fault, l2_away_fault, l2_gt, l2_gt_near_fault,
            l2_gt_away_fault]


def compute_metrics(metrics_list, steps, index):
    """
    Compute metrics across multiple steps and join them with underscores.
    """
    return "_".join([f"{metrics_list[step][index]}" for step in range(steps)])


def save_metrics(args, model_name, scaling_factor_name, regularization_name, metrics):
    """
    Save the computed metrics to a file.
    """
    epe, epe_near_fault, epe_away_fault, l2, l2_near_fault, l2_away_fault, l2_gt, l2_gt_near_fault, l2_gt_away_fault = metrics

    # define columns
    base_columns = [
        model_name, scaling_factor_name, str(args.image_size), regularization_name, str(args.fault_boundary_disk),
        str(args.eval_crop_size),
        str(args.split_count), epe
    ]
    
    base_headers = [
        "model_name", "scaling_factors", "image_size", "regularization", "fault_boundary_disk",
        "eval_crop_size", "split_count", "epe_global"
    ]
    
    additional_columns = []
    additional_headers = []
    if args.near_fault_only:
        additional_columns = [epe_near_fault, epe_away_fault, l2, l2_near_fault, l2_away_fault, l2_gt, l2_gt_near_fault,
                              l2_gt_away_fault]
        additional_headers = ["epe_near_fault", "epe_away_fault", "l2_global", "l2_near_fault", "l2_away_fault", "l2_gt_global", "l2_gt_near_fault", "l2_gt_away_fault"]
        
    columns = base_columns + additional_columns
    headers = base_headers + additional_headers

    # save metric file
    metric_dir = os.path.dirname(args.metric_filename)
    if metric_dir != '':
        os.makedirs(metric_dir, exist_ok=True)
        
    file_exists = os.path.exists(args.metric_filename)
    is_empty = not file_exists or os.path.getsize(args.metric_filename) == 0
    
    with open(args.metric_filename, "a+") as writer:
        if is_empty:
            writer.write(";".join(headers) + "\n")
        writer.write(";".join(columns) + "\n")


def get_frame_labels(frame_ids, scaling_factors, dataset_name, frame_names=None):
    if frame_names:
        frame_labels = ["_".join(frame_names[f].split("_")[1:]) for f in frame_ids]
    elif dataset_name.lower().startswith("faultdeform"):
        frame_labels = [f"{f:06d}_{scaling_factors[ii]:01d}" for ii, f in enumerate(frame_ids)]
    else:
        frame_labels = [f"{f:06d}" for f in frame_ids]
    return frame_labels
