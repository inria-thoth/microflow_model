import torch
import torch.nn.functional as F


def create_regular_grid(h, w, device):
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack([grid_x, grid_y], dim=0).float().unsqueeze(0)
    return grid.to(device)


def add_flow(grid, flow_field, h, w):
    new_grid = grid + flow_field
    # Normalize grid to [-1, 1]
    new_grid[:, 0, :, :] = (2 * new_grid[:, 0, :, :].clone() / max(w - 1, 1) - 1) * (w - 1) / w
    new_grid[:, 1, :, :] = (2 * new_grid[:, 1, :, :].clone() / max(w - 1, 1) - 1) * (w - 1) / w
    return new_grid.permute(0, 2, 3, 1)


def warp_image_torch(image, flow_field):
    """
    Warp an image using a flow field.
    """
    with torch.cuda.amp.autocast(enabled=False):
        image = image.float()
        flow_field = flow_field.float()
        _, _, h, w = image.size()
        grid = create_regular_grid(h, w, image.device)

        new_grid = add_flow(grid, flow_field, h, w)

        # note that bicubic can be less stable -> it might be worth testing with bilinear instead
        warped_image = F.grid_sample(image, new_grid, mode='bicubic', padding_mode='border', align_corners=False)

    return warped_image


def reverse_flow(flow_field, max_iter=2):
    """
    Reverse a flow field using the iterative approach described in the paper: 
    A simple fixed-point approach to invert a deformation field    
    """
    _, _, h, w = flow_field.size()
    grid = create_regular_grid(h, w, flow_field.device)

    flow_iters = [None] * (max_iter + 1)
    flow_iters[0] = flow_field.clone()
    new_grid_iters = [None] * (max_iter + 1)
    for i in range(max_iter):
        new_grid_iters[i + 1] = add_flow(grid, -flow_iters[i], h, w)
        flow_iter_warped = torch.nn.functional.grid_sample(
            flow_field, new_grid_iters[i + 1], mode='bilinear', padding_mode='border', align_corners=False)
        flow_iters[i + 1] = flow_iter_warped

    return -flow_iters[max_iter]


def get_inverse_flow(predicted_flow):
    """ With Qgis convention"""
    flow = predicted_flow.clone()
    inverse_flow = reverse_flow(flow, max_iter=2)
    inverse_flow[:, 0] *= -1  # Qgis convention
    return inverse_flow


