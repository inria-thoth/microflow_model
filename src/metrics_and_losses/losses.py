import torch.nn as nn
import torch

MAX_FLOW = 400

class IntermediateL1Loss(nn.Module):
    def __init__(self, gamma=1):
        super(IntermediateL1Loss, self).__init__()
        self.gamma = gamma

    def forward(self, predictions, target, valid=None, max_flow=MAX_FLOW):
        n_predictions = len(predictions)
        flow_loss = 0.0

        mag = torch.sum(target ** 2, dim=1).sqrt()

        if valid is None:
            valid = (mag < max_flow)
        else:
            valid = (valid >= 0.5) & (mag < max_flow)

        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            i_loss = (predictions[i] - target).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        return flow_loss
