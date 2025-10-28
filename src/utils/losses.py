# losses.py
import torch
import torch.nn.functional as F


@torch.no_grad()
def _build_class_weights_tensor(class_names, weights_by_name, device):
    """class_names: [C] в том же порядке, что и у модели/датасета.
    weights_by_name: dict {'dorogi': 2.0, ...} из конфига."""
    w = [float(weights_by_name.get(name, 1.0)) for name in class_names]
    t = torch.tensor(w, dtype=torch.float32, device=device)
    # нормализуем, чтобы масштаб лосса был стабильным
    t = t / (t.mean().clamp_min(1e-6))
    return t


class WeightedBCEDiceLoss(torch.nn.Module):
    """
    Взвешенный BCE + SoftDice по каналам:
      loss = bce_w * BCE_w + dice_w * Dice_w
    где BCE_w и Dice_w — per-class, умноженные на веса.

    Параметры:
      class_weights: 1D-тензор [C] на device (или None)
      skip_empty_dice: если True — каналы без поз. пикселей в батче в Dice почти игнорируем
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        bce_w: float = 0.5,
        dice_w: float = 0.5,
        smooth: float = 1e-6,
        skip_empty_dice: bool = True,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights if class_weights is not None else None, persistent=False)
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.smooth = smooth
        self.skip_empty_dice = skip_empty_dice

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits, targets: [B,C,H,W]; targets ∈ {0,1}
        """
        B, C, H, W = logits.shape
        # --- BCE per-class
        bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # [B,C,H,W]
        bce_c = bce_map.mean(dim=(0, 2, 3))  # [C]

        # --- Dice per-class
        probs = logits.sigmoid()
        num = 2.0 * (probs * targets).sum(dim=(2, 3))  # [B,C]
        den = (probs.pow(2) + targets.pow(2)).sum(dim=(2, 3)) + self.smooth
        dice_bc = 1.0 - (num + self.smooth) / den  # [B,C]
        dice_c = dice_bc.mean(dim=0)  # [C]

        # --- веса
        if self.class_weights is not None:
            cw = self.class_weights
        else:
            cw = torch.ones(C, dtype=torch.float32, device=logits.device)

        # опционально ослабим Dice для каналов без позитивов (чтобы не тянули вниз)
        if self.skip_empty_dice:
            present = (targets.sum(dim=(0, 2, 3)) > 0).float()  # [C]
            # для пустых классов уменьшим вклад Dice в 5 раз
            dice_weight = torch.lerp(0.2 * cw, cw, present)  # [C]
        else:
            dice_weight = cw

        bce_w = (bce_c * cw).sum() / cw.sum().clamp_min(1e-6)
        dice_w = (dice_c * dice_weight).sum() / dice_weight.sum().clamp_min(1e-6)

        return self.bce_w * bce_w + self.dice_w * dice_w


# Удобная обёртка: построить лосс прямо из списка классов и словаря весов
def make_weighted_bce_dice(class_names, weights_by_name, device, bce_w=0.5, dice_w=0.5, skip_empty_dice=True):
    cw = _build_class_weights_tensor(class_names, weights_by_name, device)
    return WeightedBCEDiceLoss(cw, bce_w=bce_w, dice_w=dice_w, skip_empty_dice=skip_empty_dice)
