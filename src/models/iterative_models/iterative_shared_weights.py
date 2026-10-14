import torch
import torch.nn as nn
import numpy as np

from src.models.model_utils.backbone_model_utils import load_backbone
from src.models.model_utils.iterative_model_utils import normalize_image, compute_regularization
from src.models.model_utils.warping_utils import warp_image_torch, get_inverse_flow
from src.models.model_utils.saving_utils import apply_convention_and_save


def get_iterative_shared_weights_model(
        device, model_name, batch_norm, max_iterations, filenames, trained_end2end=False):
    model = IterativeSharedWeights(device, model_name, batch_norm, max_iterations)
    if trained_end2end and filenames[0] is not None:
        model.load_state_dict(torch.load(filenames[0], map_location=device)['model_state_dict'])

    return model


class IterativeSharedWeights(nn.Module):
    expansion = 1

    def __init__(self, device, model_name, batch_norm, max_iterations):
        super(IterativeSharedWeights, self).__init__()
        self.max_iterations = max_iterations
        self.iteration_model = load_backbone(model_name, batch_norm, device)

    def forward(self, x, ptv=None, args=None, save=False, frame_labels=False, crs_meta_data=None, transform_meta_data=None):
        b, _, h, w = x.shape
        post_not_norm = x[:, 2].view(b, 1, h, w)

        iteration_inputs, iteration_sum_optical_flows = [], []
        iteration_optical_flows, iteration_inverse_flows, iteration_warped_images, iteration_warped_normalized_images = [], [], [], []

        for iteration in range(self.max_iterations):
            # inputs for the iterations
            iteration_inputs.append(x[:, 0:2].clone())
            if iteration > 0:
                iteration_inputs[iteration][:, 1] = iteration_warped_normalized_images[iteration - 1][:, 0]

            # predicts the residual flow
            iteration_optical_flows.append(self.iteration_model(iteration_inputs[iteration])[0])

            # sums residual flows to get the predicted flow
            if iteration == 0:
                iteration_sum_optical_flows.append(iteration_optical_flows[iteration])
            else:
                iteration_sum_optical_flows.append(iteration_sum_optical_flows[iteration - 1] + iteration_optical_flows[iteration])

            # inverses the flow
            iteration_inverse_flows.append(get_inverse_flow(iteration_sum_optical_flows[iteration]))

            # warps the x2 image with the flow
            iteration_warped_images.append(warp_image_torch(post_not_norm, iteration_inverse_flows[iteration]))

            # renormalizes the image
            iteration_warped_normalized_images.append(normalize_image(iteration_warped_images[iteration]))

        if save:
            apply_convention_and_save(
                frame_labels, 
                iteration_sum_optical_flows[iteration], 
                args.save_dir, 
                f"noisy_flow_iteration{iteration}", 
                crs_meta_data, 
                transform_meta_data
                )

            # For tiff visualization, save the denoised version at each iteration, though only the last iteration is denoised in practice
            denoised = compute_regularization(iteration_sum_optical_flows[iteration], args, ptv)
            apply_convention_and_save(
                frame_labels, 
                denoised, 
                args.save_dir, 
                f"denoised_flow_iteration{iteration}", 
                crs_meta_data, 
                transform_meta_data
                )

        if args.regularization:
            iteration_sum_optical_flows[iteration] = compute_regularization(iteration_sum_optical_flows[iteration], args, ptv)

        return iteration_sum_optical_flows


    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

