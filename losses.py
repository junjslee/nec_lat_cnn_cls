import torch.nn as nn
import torch

class ClassificationLoss(nn.Module):
    def __init__(self, classification_weight=1.0):
        super().__init__()
        self.classification_weight = classification_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, cls_pred: torch.Tensor, cls_gt: torch.Tensor) -> torch.Tensor:
        loss = self.bce_loss(cls_pred, cls_gt) * self.classification_weight
        loss_dict = {'CLS_Loss': loss.item()}
        return loss, loss_dict
