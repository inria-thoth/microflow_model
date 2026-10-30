import torch
import numpy as np
from src.metrics_and_losses.gradient_metrics import compute_gradients


def normalize_image(image):
    image_mean = torch.mean(image)
    image_std = torch.std(image)
    return (image - image_mean) / image_std


def create_weights_from_inverse_grads_gpu(grad_predictions, ltv_lambda=0, epsilon=1e-8):
    """
    Create weights from inverse gradients for the LTV regularization using pure PyTorch.
    Keeps all data and computations directly on the device (GPU/CPU).
    """
    h, w = grad_predictions.shape[-2], grad_predictions.shape[-1]
    crop_size = min(100, h // 4, w // 4)
    if crop_size > 0:
        cropped = grad_predictions[..., crop_size:-crop_size, crop_size:-crop_size]
    else:
        cropped = grad_predictions

    minimum = torch.min(cropped) + epsilon
    maximum = torch.max(cropped)
    grad_clamped = torch.clamp(grad_predictions, min=minimum)

    square = (ltv_lambda * maximum) / grad_clamped
    return square


def create_weights_from_inverse_grads(grad_predictions, ltv_lambda=0, epsilon=1e-8):
    """
    Create weights from inverse gradients for the legacy prox-tv regularization (CPU).
    """
    crop_size = 100  # crop to avoid border effects
    h, w = grad_predictions.shape
    minimum = np.min(grad_predictions[crop_size:-crop_size, crop_size:-crop_size]) + epsilon  
    maximum = np.max(grad_predictions[crop_size:-crop_size, crop_size:-crop_size])
    grad_predictions = np.where(grad_predictions < minimum, minimum, grad_predictions)

    square = ltv_lambda * maximum / grad_predictions
    w_col = square[:h - 1, :w]
    w_row = square[:h, :w - 1]
    return w_col, w_row


def solve_proximal_energy(v, weight_map, penalty="ltv", num_iters=35, lr=0.1):
    """
    Pure PyTorch replacement for proxTV.
    Minimizes the energy functional:
        E(x) = 0.5 * ||x - v||_2^2 + R(x)
    using Adam gradient descent on GPU/CPU.
    """
    x = v.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)

    for _ in range(num_iters):
        optimizer.zero_grad()

        # 1. Data fidelity term: 0.5 * ||x - v||_2^2
        loss_data = 0.5 * torch.sum((x - v) ** 2)

        # 2. Spatial discrete gradients (horizontal and vertical differences)
        dx = x[..., :, 1:] - x[..., :, :-1]
        dy = x[..., 1:, :] - x[..., :-1, :]

        # 3. Regularization penalty term
        if penalty == "ltv":
            w_dx = weight_map[..., :, :-1]
            w_dy = weight_map[..., :-1, :]
            loss_reg = torch.sum(w_dx * torch.abs(dx)) + torch.sum(w_dy * torch.abs(dy))
        elif penalty == "tv":
            loss_reg = weight_map * (torch.sum(torch.abs(dx)) + torch.sum(torch.abs(dy)))
        elif penalty == "l2":
            w_dx = weight_map[..., :, :-1]
            w_dy = weight_map[..., :-1, :]
            loss_reg = torch.sum(w_dx * (dx ** 2)) + torch.sum(w_dy * (dy ** 2))
        else:
            raise ValueError(f"Unknown penalty function: {penalty}")

        loss = loss_data + loss_reg
        loss.backward()
        optimizer.step()

    return x.detach()


def compute_regularization_gpu(predicted_flow, args):
    """
    Compute regularization on GPU/CPU using native PyTorch energy minimization.
    """
    denoised_predicted = predicted_flow.clone()
    b, c, h, w = denoised_predicted.shape

    num_iters = getattr(args, 'reg_2d_max_iter', 10)
    lr = getattr(args, 'reg_lr', 0.1)
    reg_lambda = getattr(args, 'reg_lambda', 0.001)
    reg_iterations = getattr(args, 'reg_iterations', 3)
    penalty = getattr(args, 'penalty_function', 'ltv')

    for i_batch in range(b):
        for i_channel in range(c):
            v_target = denoised_predicted[i_batch, i_channel]

            if penalty == "ltv":
                for _ in range(reg_iterations):
                    grad_gpu = compute_gradients(denoised_predicted)
                    weight_map = create_weights_from_inverse_grads_gpu(
                        grad_gpu[i_batch, 0], ltv_lambda=reg_lambda
                    )
                    v_target = solve_proximal_energy(
                        v_target,
                        weight_map,
                        penalty="ltv",
                        num_iters=num_iters,
                        lr=lr
                    )
                    denoised_predicted[i_batch, i_channel] = v_target

            elif penalty == "tv":
                for _ in range(reg_iterations):
                    v_target = solve_proximal_energy(
                        v_target,
                        weight_map=reg_lambda,
                        penalty="tv",
                        num_iters=num_iters,
                        lr=lr
                    )
                denoised_predicted[i_batch, i_channel] = v_target

            elif penalty == "l2":
                for _ in range(reg_iterations):
                    v_target = solve_proximal_energy(
                        v_target,
                        weight_map=reg_lambda,
                        penalty="l2",
                        num_iters=num_iters,
                        lr=lr
                    )
                denoised_predicted[i_batch, i_channel] = v_target

    return denoised_predicted


def compute_regularization_proxtv(predicted_flow, args, ptv=None):
    """
    Legacy CPU regularization computed using the proxTV library (albarji/proxTV).
    """
    if ptv is None:
        try:
            import prox_tv as ptv
        except ImportError:
            raise ImportError(
                "prox-tv is not installed. To use GPU regularization without C++ dependencies, "
                "set --reg_backend gpu."
            )

    denoised_predicted = predicted_flow.clone()
    predicted_cpu = predicted_flow.detach().cpu().numpy()

    b, c, _, _ = denoised_predicted.shape
    for i_batch in range(b):
        for i_channel in range(c):
            denoised_cpu = predicted_cpu[i_batch, i_channel].copy()
            if args.penalty_function == "ltv":
                for _ in range(args.reg_iterations):
                    grad_cpu = compute_gradients(denoised_predicted).detach().cpu().numpy()
                    w_col, w_row = create_weights_from_inverse_grads(
                        grad_cpu[i_batch, 0], ltv_lambda=args.reg_lambda)
                    denoised_cpu = ptv.tv1w_2d(
                        denoised_cpu, w_col, w_row, max_iters=args.reg_2d_max_iter, n_threads=args.reg_threads)
                    denoised_predicted[i_batch, i_channel] = torch.tensor(denoised_cpu)
            elif args.penalty_function == "tv":
                for _ in range(args.reg_iterations):
                    denoised_cpu = ptv.tv1_2d(
                        denoised_cpu, args.reg_lambda, n_threads=args.reg_threads, max_iters=args.reg_2d_max_iter)
                denoised_predicted[i_batch, i_channel] = torch.tensor(denoised_cpu)
            elif args.penalty_function == "l2":
                for _ in range(args.reg_iterations):
                    denoised_cpu = ptv.tvp_2d(
                        denoised_cpu,
                        w_col=args.reg_lambda,
                        w_row=args.reg_lambda,
                        p_col=2,
                        p_row=2,
                        n_threads=args.reg_threads,
                        max_iters=args.reg_2d_max_iter)
                denoised_predicted[i_batch, i_channel] = torch.tensor(denoised_cpu)

    return denoised_predicted


def compute_regularization(predicted_flow, args, ptv=None):
    """
    Main entry point for regularization. Dispatches to GPU PyTorch solver or CPU prox-tv.
    """
    backend = getattr(args, 'reg_backend', getattr(args, 'backend', 'gpu'))
    if backend == 'gpu':
        return compute_regularization_gpu(predicted_flow, args)
    else:
        return compute_regularization_proxtv(predicted_flow, args, ptv=ptv)

