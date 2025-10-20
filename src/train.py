import yaml
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import amp
from tqdm import tqdm

from .dataset_sliding import SlidingGeoDataset
from .model_unet import build_unet
from .losses import ComboLoss
from .metrics_local import f2_weighted
import json, collections

import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2


def _save_curve(out_dir, history):
    # history: list of dict(epoch, loss, f2w)
    xs = [h["epoch"] for h in history]
    ls = [h["loss"] for h in history]
    fs = [h["f2w"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ls, label="train loss")
    plt.plot(xs, fs, label="val F2w")
    plt.xlabel("epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    (out_dir / "learning_curve.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "learning_curve.png", dpi=150)
    plt.close()


def _append_metrics_csv(out_dir, row):
    csv_path = out_dir / "metrics.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_f2w"])
        if write_header:
            w.writeheader()
        w.writerow({"epoch": row["epoch"], "train_loss": f"{row['loss']:.6f}", "val_f2w": f"{row['f2w']:.6f}"})


# простая раскраска много-классовой маски поверх 1-канального изображения
_PALETTE = [
    (255, 0, 0),  # class 0
    (0, 255, 0),  # class 1
    (0, 0, 255),  # class 2
    (255, 255, 0),  # class 3
    (255, 0, 255),  # class 4
    (0, 255, 255),  # class 5
    (255, 128, 0),  # class 6
    (128, 0, 255),  # class 7
]


def _save_val_overlays(out_dir, imgs, probs, gts, class_names, thresholds, max_samples=6, alpha=0.4):
    """imgs: [B,1,H,W] float; probs/gts: [B,C,H,W]."""
    b = min(len(imgs), max_samples)
    out = out_dir / "overlays"
    out.mkdir(parents=True, exist_ok=True)
    imgs = imgs[:b].cpu().numpy()
    probs = probs[:b].cpu().numpy()
    gts = gts[:b].cpu().numpy()

    for i in range(b):
        img = (imgs[i, 0] * 255).astype(np.uint8)
        img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # GT контуры
        gt_rgb = img3.copy()
        for k, cname in enumerate(class_names):
            gt_bin = (gts[i, k] > 0.5).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(gt_rgb, cnts, -1, _PALETTE[k % len(_PALETTE)], 1)

        # PRED маска
        pred_rgb = img3.copy()
        for k, cname in enumerate(class_names):
            thr = thresholds.get(cname, 0.35)
            pr_bin = (probs[i, k] >= thr).astype(np.uint8)
            color = np.array(_PALETTE[k % len(_PALETTE)], dtype=np.uint8)
            overlay = np.zeros_like(pred_rgb)
            overlay[pr_bin == 1] = color
            pred_rgb = cv2.addWeighted(pred_rgb, 1.0, overlay, alpha, 0)

        cv2.imwrite(str(out / f"val_{i:02d}_img.png"), img3)
        cv2.imwrite(str(out / f"val_{i:02d}_gt.png"), gt_rgb)
        cv2.imwrite(str(out / f"val_{i:02d}_pred.png"), pred_rgb)


def main(cfg_path="configs/default.yaml"):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    out_dir = Path(cfg["log"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = list(cfg["data"]["classes"].keys())
    class_weights = [cfg["data"]["classes"][c] for c in classes]
    class_buffers = cfg["data"].get("class_buffer_m", {})
    tile, stride = cfg["data"]["tile_size"], cfg["data"]["tile_stride"]

    train_regions = pd.read_csv(out_dir / "train_regions.csv")
    val_regions = pd.read_csv(out_dir / "val_regions.csv")

    # Опционально ограничиваем число регионов для быстрого теста
    lim_tr = int(cfg["train"].get("limit_regions", 0) or 0)
    lim_va = int(cfg["train"].get("limit_val_regions", 0) or 0)
    if lim_tr > 0:
        train_regions = train_regions.head(lim_tr)
    if lim_va > 0:
        val_regions = val_regions.head(lim_va)

    def _decode(col):
        return col.apply(lambda s: json.loads(s) if isinstance(s, str) else s)

    train_regions["raster_paths"] = _decode(train_regions["raster_paths"])
    val_regions["raster_paths"] = _decode(val_regions["raster_paths"])

    train_regions["labels_paths"] = _decode(train_regions["labels_paths"])
    val_regions["labels_paths"] = _decode(val_regions["labels_paths"])

    counter = collections.Counter()
    for arr in train_regions["raster_paths"]:
        for i, p in enumerate(arr):
            if p:
                counter[i] += 1
    print("Channel coverage (train):", dict(counter))

    ds_tr = SlidingGeoDataset(train_regions, classes, tile, stride, augment=True, class_buffers_m=class_buffers)
    ds_va = SlidingGeoDataset(val_regions, classes, tile, stride, augment=False, class_buffers_m=class_buffers)
    print(f"Tiles: train={len(ds_tr)}, val={len(ds_va)}")

    # Быстрый sanity-check: есть ли вообще положительные пиксели?
    from torch.utils.data import Subset
    import numpy as np

    probe = Subset(ds_tr, list(range(min(50, len(ds_tr)))))
    pos_pix = np.zeros(len(classes), dtype=np.int64)
    for i in range(len(probe)):
        _, m, _ = probe[i]
        # m: [C,H,W]
        pos_pix += m.long().sum(dim=(1, 2)).numpy()
    print("Pos pixels per class (probe):", dict(zip(classes, map(int, pos_pix))))
    exit()
    # ============================================

    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=max(1, cfg["train"]["batch_size"] // 2),
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    num_ch = len(cfg["data"].get("hillshade_channels", [None]))
    model = build_unet(
        cfg["model"]["encoder"], cfg["model"]["encoder_weights"], in_channels=num_ch, classes=len(classes)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = amp.GradScaler(device, enabled=(torch.cuda.is_available() and bool(cfg["train"]["amp"])))
    criterion = ComboLoss(class_weights=class_weights, gamma=2.0)

    best = -1.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        for img, mask, _meta in tqdm(dl_tr, desc=f"Train {epoch}"):
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad(set_to_none=True)
            with amp.autocast(device, enabled=(torch.cuda.is_available() and bool(cfg["train"]["amp"]))):
                logits = model(img)
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item()
        # validation
        model.eval()
        probs_all, gts_all, imgs_all = [], [], []
        with torch.no_grad():
            for img, mask, _meta in tqdm(dl_va, desc=f"Val {epoch}"):
                img = img.to(device)
                logits = model(img)
                p = torch.sigmoid(logits).cpu()
                probs_all.append(p)
                gts_all.append(mask)
                imgs_all.append(img.cpu())
        probs_all = torch.cat(probs_all, 0)
        gts_all = torch.cat(gts_all, 0)
        imgs_all = torch.cat(imgs_all, 0)

        f2w, per = f2_weighted(gts_all, probs_all, cfg["thresholds"], classes, cfg["data"]["classes"])
        cur_loss = loss_sum / len(dl_tr)
        print(
            f"Epoch {epoch} | loss={cur_loss:.4f} | F2w={f2w:.4f} | "
            + ", ".join(f"{c}:{per[i]:.3f}" for i, c in enumerate(classes))
        )

        # ===== NEW: история/графики/CSV/оверлеи =====
        if "history" not in locals():
            history = []
        history.append({"epoch": epoch, "loss": cur_loss, "f2w": float(f2w)})
        _save_curve(out_dir, history)
        _append_metrics_csv(out_dir, {"epoch": epoch, "loss": cur_loss, "f2w": float(f2w)})

        # Сохраняем несколько валидационных оверлеев на каждой эпохе
        _save_val_overlays(out_dir, imgs_all, probs_all, gts_all, classes, cfg["thresholds"], max_samples=6, alpha=0.45)
        # ============================================

        if f2w > best:
            best = f2w
            torch.save({"state_dict": model.state_dict(), "classes": classes, "cfg": cfg}, out_dir / "best.pt")
            print("Saved best:", out_dir / "best.pt")
    print("Done. Best F2w:", best)


if __name__ == "__main__":
    main()
