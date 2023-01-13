import torch
import torch.nn as nn

class BCEMask(nn.Module):
    """binary cross entropy loss with mask
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss(reduction="none")

    def forward(self, input, y_true, mask):
        l = self.loss(input, y_true[:, :input.shape[1]])
        l = torch.mean(l[mask])
        return l

class BCEMaskWeighted(nn.Module):
    """weighted binary cross entropy loss (with mask), 
    this is used when fine-tuning LELA on unbalanced datasets 
    """
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward_no_mask(self, output, target):
        if self.weights is not None:
            assert len(self.weights) == 2
            
            loss = self.weights[1] * (target * torch.log(output)) + \
                self.weights[0] * ((1 - target) * torch.log(1 - output))
        else:
            loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

        return torch.neg(loss)

    def forward(self, input, y_true, mask):
        l = self.forward_no_mask(input, y_true[:, :input.shape[1]])
        l = torch.mean(l[mask])
        return l
