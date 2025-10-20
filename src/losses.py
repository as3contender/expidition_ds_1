import torch
import segmentation_models_pytorch as smp


class ComboLoss(torch.nn.Module):
    def __init__(self, class_weights=None, gamma=2.0):
        super().__init__()
        # Проблемы с broadcast alpha при multilabel → используем alpha=None
        # (при необходимости веса классов можно применить поверх суммарного лосса)
        self.focal = smp.losses.FocalLoss(mode="multilabel", gamma=gamma, alpha=None)
        self.dice = smp.losses.DiceLoss(mode="multilabel")

    def forward(self, logits, targets):
        return self.focal(logits, targets) + self.dice(logits, targets)
