import torch.nn as nn

class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)
