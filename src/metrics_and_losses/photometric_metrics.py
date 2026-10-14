import torch
import torch.nn.functional as F


def EPE(input_flow, target_flow, sparse=False, edge_mask=None):
    error_map = target_flow - input_flow
    EPE_map = torch.norm(error_map, p=2, dim=1)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)
        EPE_map = EPE_map[~mask]

    EPE_2_norm = EPE_map.mean()
    
    if edge_mask is not None:
        edge_count = torch.sum(edge_mask)
        non_edge_count = torch.sum(1 - edge_mask)
        
        if len(edge_mask.shape) == 4:
            edge_mask = edge_mask[:, 0]
            
        edge_EPE = torch.sum(EPE_map * edge_mask) / max(edge_count, 1)
        non_edge_EPE = torch.sum(EPE_map * (1 - edge_mask)) / max(non_edge_count, 1)
        return EPE_2_norm, edge_EPE, non_edge_EPE

    return EPE_2_norm


def realEPE(output, target, sparse=False, edge_mask=None):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
    return EPE(upsampled_output, target, sparse, edge_mask)
