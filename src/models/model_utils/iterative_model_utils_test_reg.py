import torch
import numpy as np
from src.metrics_and_losses.gradient_metrics import compute_gradients


def normalize_image(image):
    image_mean = torch.mean(image)
    image_std = torch.std(image)
    return (image - image_mean) / image_std


def create_weights_from_inverse_grads(grad_predictions, ltv_lambda=0, epsilon=1e-6):
    """
    Create weights from inverse gradients for the ltv regularization
    """
    crop_size=100  # crop to avoid border effects
    h, w = grad_predictions.shape
    minimum = np.min(grad_predictions[crop_size:-crop_size, crop_size:-crop_size]) + epsilon
    maximum = np.max(grad_predictions[crop_size:-crop_size, crop_size:-crop_size])
    #print(maximum)
    ltv_lambda = ltv_lambda * 0.7
    #grad_predictions = np.where(grad_predictions < minimum, minimum, grad_predictions)
    grad_predictions = grad_predictions + epsilon #minimum
    square = ltv_lambda  / grad_predictions #* maximum
    w_col = square[:h - 1, :w]
    w_row = square[:h, :w - 1]
    return w_col, w_row


def compute_regularization(predicted_flow, args, ptv):
    """
    The regularization is computed using the proxtv library 
    https://github.com/albarji/proxTV

    The flow in each direction (represented by channel) is regularized independently.
    """
    denoised_predicted = predicted_flow.clone()
    predicted_cpu = predicted_flow.detach().cpu().numpy()       

    b, c, _, _ = denoised_predicted.shape
    for i_batch in range(b):
        for i_channel in range(c):
            denoised_cpu = predicted_cpu[i_batch, i_channel].copy() 
            if args.penalty_function=="ltv":                           
                for _ in range(args.reg_iterations):           
                    # max_iters is the number of alternances between processing columns and processing rows (different from the number of iterations k in the paper)
                    # by default, when max_iters=0, it is replaced by 35 in the code. 
                    # when set to 1, it gives "stripped" results
                    # https://github.com/albarji/proxTV/blob/master/src/TV2DWopt.cpp
                    
                    grad_cpu = compute_gradients(denoised_predicted).detach().cpu().numpy()
                    w_col, w_row = create_weights_from_inverse_grads(
                        grad_cpu[i_batch, 0], ltv_lambda=args.reg_lambda)

                    denoised_cpu = ptv.tv1w_2d(
                        denoised_cpu, w_col, w_row, max_iters=args.reg_2d_max_iter, n_threads=args.reg_threads)  
                    denoised_predicted[i_batch, i_channel] = torch.tensor(denoised_cpu)
            elif args.penalty_function == "tv":
                for _ in range(args.reg_iterations): 
                    # max_iters is the number of alternances between processing columns and processing rows (different from the number of iterations k in the paper)
                    denoised_cpu = ptv.tv1_2d(
                        denoised_cpu, args.reg_lambda, n_threads=args.reg_threads, max_iters=args.reg_2d_max_iter)  
                denoised_predicted[i_batch, i_channel] = torch.tensor(denoised_cpu)
            elif args.penalty_function == "l2":
                for iii in range(args.reg_iterations): 
                    # max_iters is the number of alternances between processing columns and processing rows (different from the number of iterations k in the paper)
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
