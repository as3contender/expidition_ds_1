# src/train.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
import math
import random

import albumentations as A
import cv2
import yaml
import segmentation_models_pytorch as smp

# --- наши модули ---
from .dataset_sliding import SlidingGeoDataset
from .build_splits import build_splits
from .utils.losses import make_weighted_bce_dice


# ===========================
#        УТИЛИТЫ
# ===========================
def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_cfg(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def class_name_to_idx(classes: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(classes)}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------- Метрики ----------
def f2_score(pred: np.ndarray, gt: np.ndarray, beta2: float = 4.0) -> float:
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    denom = (1 + beta2) * tp + beta2 * fn + fp + 1e-9
    return (1 + beta2) * tp / denom


def compute_f2_per_class(y_pred_bin: np.ndarray, y_true: np.ndarray) -> Tuple[List[float], float]:
    C = y_true.shape[0]
    scores = []
    for ci in range(C):
        scores.append(f2_score(y_pred_bin[ci], y_true[ci]))
    return scores, float(np.mean(scores))


# ---------- Постобработка ----------
def postproc_per_class(
    prob_stack: np.ndarray, thresholds_idx: Dict[int, float], thin_class_idx: List[int] | None = None
) -> np.ndarray:
    """
    prob_stack: [C,H,W] float[0..1]
    возвращает бинарные маски [C,H,W] uint8
    """
    C, H, W = prob_stack.shape
    out = np.zeros((C, H, W), dtype=np.uint8)

    k = np.ones((3, 3), np.uint8)
    for ci in range(C):
        thr = float(thresholds_idx.get(ci, 0.35))
        binm = (prob_stack[ci] >= thr).astype(np.uint8)

        if thin_class_idx and (ci in thin_class_idx):
            # более агрессивно закрываем разрывы для тонких классов
            binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k, iterations=2)
            binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, k, iterations=1)

        out[ci] = binm
    return out


# ---------- Оверлеи ----------
def save_overlay(img_hwc: np.ndarray, pred_bin: np.ndarray, gt_bin: np.ndarray, out_path: Path):
    """
    img_hwc: [H,W,C] float[0..1]
    pred_bin, gt_bin: [C,H,W] uint8
    Красный=Pred, Зелёный=GT (суммарно по классам).
    """
    H, W, C = img_hwc.shape

    # Преобразуем в RGB для визуализации
    if C == 1:
        # Grayscale -> RGB
        rgb = np.repeat(img_hwc, 3, axis=2)
    elif C == 3:
        # Уже RGB
        rgb = img_hwc
    elif C == 4:
        # 4 канала -> берем первые 3 или используем один канал как grayscale
        # Вариант: используем канал ch (hillshade) как grayscale
        rgb = np.repeat(img_hwc[:, :, 1:2], 3, axis=2)  # канал ch (индекс 1)
    else:
        # Fallback: первые 3 канала
        rgb = img_hwc[:, :, :3]

    base = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    pred_any = (pred_bin.max(axis=0) > 0).astype(np.uint8)
    gt_any = (gt_bin.max(axis=0) > 0).astype(np.uint8)

    overlay = base.copy()
    overlay[pred_any == 1] = (0.6 * overlay[pred_any == 1] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)
    overlay[gt_any == 1] = (0.6 * overlay[gt_any == 1] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# ===========================
#   Балансированный Sampler
# ===========================
class PosBalancedSampler(Sampler):
    """
    Возвращает индексы так, чтобы не меньше frac_pos батча были с метками.
    dataset должен иметь атрибут non_empty_indices: List[int].
    Если его нет или он пуст — используем равномерную выборку.
    """

    def __init__(self, dataset, batch_size: int, frac_pos: float = 0.5, length: int | None = None, seed: int = 42):
        self.ds = dataset
        self.bs = max(1, batch_size)
        self.frac = float(frac_pos)
        self.length = length or len(dataset)
        self.rng = random.Random(seed)

        pos = list(getattr(dataset, "non_empty_indices", []))
        neg = [i for i in range(len(dataset)) if i not in set(pos)]
        self.pos = pos
        self.neg = neg
        self.enabled = len(self.pos) > 0 and len(self.neg) > 0

    def __len__(self):
        return self.length

    def __iter__(self):
        if not self.enabled:
            # fallback: просто случайные индексы
            idx = list(range(len(self.ds)))
            self.rng.shuffle(idx)
            for i in idx[: self.length]:
                yield i
            return

        pos = self.pos.copy()
        neg = self.neg.copy()
        self.rng.shuffle(pos)
        self.rng.shuffle(neg)
        pi = ni = 0

        need_pos = max(1, int(round(self.bs * self.frac)))
        need_neg = max(0, self.bs - need_pos)

        steps = math.ceil(self.length / self.bs)
        for _ in range(steps):
            batch = []
            for _ in range(need_pos):
                if pi >= len(pos):
                    self.rng.shuffle(self.pos)
                    pos = self.pos.copy()
                    pi = 0
                batch.append(pos[pi])
                pi += 1
            for _ in range(need_neg):
                if ni >= len(neg):
                    self.rng.shuffle(self.neg)
                    neg = self.neg.copy()
                    ni = 0
                batch.append(neg[ni])
                ni += 1
            self.rng.shuffle(batch)
            for idx in batch:
                yield idx


# ===========================
#         ОБУЧЕНИЕ
# ===========================
def train_one_epoch(model, loader, criterion, optimizer, device, accum_steps: int = 1):
    model.train()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="Train", leave=False)
    for it, (imgs, masks, meta) in enumerate(pbar):
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)
        loss = criterion(logits, masks) / accum_steps
        loss.backward()

        if (it + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running += loss.item() * accum_steps
        pbar.set_postfix(loss=f"{running/(it+1):.4f}")
    return running / max(1, len(loader))


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    classes: List[str],
    thresholds: Dict[str, float],
    save_dir: Path | None = None,
    max_overlays: int = 8,
):
    model.eval()
    name2idx = class_name_to_idx(classes)
    thr_idx = {name2idx[k]: v for k, v in thresholds.items() if k in name2idx}

    thin_names = [n for n in ["dorogi", "karavannye_puti"] if n in name2idx]
    thin_idx = [name2idx[n] for n in thin_names]

    per_class_scores = []
    mean_scores = []

    prob_means = np.zeros(len(classes), dtype=np.float64)
    count_b = 0

    saved = 0
    pbar = tqdm(loader, desc="Val", leave=False)
    for imgs, masks, meta in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy()  # [B,C,H,W]
        gts = masks.cpu().numpy().astype(np.uint8)  # [B,C,H,W]
        ims = imgs.cpu().numpy().transpose(0, 2, 3, 1)  # [B,H,W,C]

        prob_means += probs.mean(axis=(0, 2, 3))
        count_b += probs.shape[0]

        for b in range(probs.shape[0]):
            prob_stack = probs[b]
            gt_stack = gts[b]
            img_hwc = ims[b].copy()

            bin_stack = postproc_per_class(prob_stack, thr_idx, thin_idx)  # [C,H,W]

            scores_c, mean_c = compute_f2_per_class(bin_stack, gt_stack)
            per_class_scores.append(scores_c)
            mean_scores.append(mean_c)

            if save_dir is not None and saved < max_overlays:
                out_path = save_dir / f"val_overlay_{saved:03d}.png"
                save_overlay(img_hwc, bin_stack, gt_stack, out_path)
                saved += 1

    if not per_class_scores:
        return 0.0, {c: 0.0 for c in classes}

    per_class_scores = np.array(per_class_scores)  # [N,C]
    mean_scores = float(np.mean(mean_scores))
    per_class_mean = {classes[i]: float(per_class_scores[:, i].mean()) for i in range(len(classes))}

    # Диагностика: средние вероятности по классам
    if count_b > 0:
        prob_means /= count_b
        stats = ", ".join([f"{classes[i]}={prob_means[i]:.3f}" for i in range(len(classes))])
        print(f"[val-probs] mean(sigmoid) per class: {stats}")

    return mean_scores, per_class_mean


# ===========================
#            MAIN
# ===========================
def main():
    cfg = load_cfg(Path("configs/default.yaml"))
    set_seed(cfg.get("seed", 42))

    # ========= Девайс =========
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print(f"[device] {device}")

    # ========= Классы/веса =========
    classes = list(cfg["data"]["classes"].keys())
    class_weights_by_name = cfg["data"]["classes"]

    # ========= Индекс-таблицы =========
    mini_train_csv = os.environ.get("MINI_TRAIN_CSV")
    mini_val_csv = os.environ.get("MINI_VAL_CSV")

    if mini_train_csv and mini_val_csv:
        idx_tr = pd.read_csv(mini_train_csv)
        idx_val = pd.read_csv(mini_val_csv)
        print(f"[mini-index] loaded train={len(idx_tr)} val={len(idx_val)} from env")
    else:
        idx_tr, idx_val = build_splits(
            regions_root=cfg["data"]["regions_root"],
            hillshade_globs=cfg["data"]["hillshade_channels"],
            labels_glob=cfg["data"]["labels_subpath"],
            limit_regions=cfg["data"]["train"].get("limit_regions", None),
            limit_val_regions=cfg["data"]["train"].get("limit_val_regions", None),
        )

    # ========= Датасеты/лоадеры =========
    tile = int(cfg["data"]["tile_size"])
    stride = int(cfg["data"]["tile_stride"])
    class_buffers = cfg["data"].get("class_buffer_m", {})

    boundary_mode = cfg["data"].get("boundary_mode", {})  # {"gorodishcha": {"enabled": True, "ring_width_m": 3.0}, ...}
    ds_tr = SlidingGeoDataset(
        idx_tr, classes, tile, stride, augment=True, class_buffers_m=class_buffers, boundary_mode=boundary_mode
    )
    ds_val = SlidingGeoDataset(
        idx_val, classes, tile, stride, augment=False, class_buffers_m=class_buffers, boundary_mode=boundary_mode
    )

    bs = int(cfg["data"]["train"]["batch_size"])
    nw = int(cfg["data"]["train"]["num_workers"])

    # pin_memory только для CUDA (MPS не поддерживает)
    use_pin = device.type == "cuda"

    # Балансированный sampler (если датасет предоставляет non_empty_indices)
    use_balanced = hasattr(ds_tr, "non_empty_indices") and len(getattr(ds_tr, "non_empty_indices", [])) > 0
    if use_balanced:
        sampler_tr = PosBalancedSampler(ds_tr, batch_size=bs, frac_pos=0.5)
        loader_tr = DataLoader(
            ds_tr, batch_size=bs, sampler=sampler_tr, num_workers=nw, pin_memory=use_pin, drop_last=True
        )
        print(f"[sampler] PosBalancedSampler enabled: pos={len(ds_tr.non_empty_indices)} of {len(ds_tr)}")
    else:
        loader_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=use_pin, drop_last=True)
        print("[sampler] fallback: shuffle=True")

    loader_val = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=max(1, nw // 2), pin_memory=use_pin)

    print(f"Tiles: train={len(ds_tr)}, val={len(ds_val)}")

    # ========= Модель =========
    in_ch = int(cfg["model"].get("in_channels", 1))
    encoder_name = cfg["model"].get("encoder", "resnet18")
    encoder_weights = cfg["model"].get("encoder_weights", "imagenet")

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_ch,
        classes=len(classes),
    ).to(device)

    # ========= Лосс с весами =========
    criterion = make_weighted_bce_dice(
        class_names=classes,
        weights_by_name=class_weights_by_name,
        device=device,
        bce_w=0.4,  # слегка усилим Dice для разреженных структур
        dice_w=0.6,
        skip_empty_dice=True,
    )

    # ========= Оптимизатор/шедулер =========
    lr = float(cfg["data"]["train"]["lr"])
    wd = float(cfg["data"]["train"].get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(cfg["data"]["train"]["epochs"])
    total_steps = max(1, len(loader_tr) * epochs)
    sched_cfg = cfg["data"]["train"].get("scheduler", {"name": "cosine", "warmup_epochs": 0, "min_lr": 1e-5})
    warmup_epochs = int(sched_cfg.get("warmup_epochs", 0))
    min_lr = float(sched_cfg.get("min_lr", 1e-5))

    cosine_steps = max(1, total_steps - warmup_epochs * len(loader_tr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=min_lr)

    accum_steps = int(cfg["data"]["train"].get("grad_accum_steps", 1))

    # ========= Логи/выходы =========
    out_dir = make_dirs(Path(cfg.get("log", {}).get("out_dir", "runs/default")))
    best_path = out_dir / "best.pt"
    overlays_dir = make_dirs(out_dir / "val_overlays")
    hist_path = out_dir / "history.json"

    # Боле щадящие дефолтные пороги; можно переопределить в default.yaml: thresholds:
    thresholds = {
        "dorogi": 0.25,
        "karavannye_puti": 0.22,
        "gorodishcha": 0.30,
        "fortifikatsii": 0.30,
        "pashni": 0.35,
        "inoe": 0.40,
    }
    thresholds.update(cfg.get("thresholds", {}))

    history = {"loss": [], "F2w": []}
    best_f2w = -1.0

    # ========= Эпохи =========
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        if warmup_epochs and epoch <= warmup_epochs:
            for g in optimizer.param_groups:
                g["lr"] = lr

        tr_loss = train_one_epoch(model, loader_tr, criterion, optimizer, device, accum_steps=accum_steps)

        if not warmup_epochs or epoch > warmup_epochs:
            scheduler.step()

        f2w, per_class = validate(model, loader_val, device, classes, thresholds, save_dir=overlays_dir, max_overlays=8)

        history["loss"].append(float(tr_loss))
        history["F2w"].append(float(f2w))
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        pcs = ", ".join([f"{k}:{per_class.get(k,0.0):.3f}" for k in classes])
        print(f"Epoch {epoch} | loss={tr_loss:.4f} | F2w={f2w:.4f} | {pcs}")

        if f2w > best_f2w:
            best_f2w = f2w
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "f2w": best_f2w, "classes": classes}, best_path
            )
            print(f"Saved best: {best_path}")

    print(f"Done. Best F2w: {best_f2w:.6f}")


if __name__ == "__main__":
    # Отключаем проверку версий albumentations (убирает SSL warnings)
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    main()
