#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from shapely.geometry import box


# Allow importing project src modules
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from raster_utils import sliding_windows as sliding_windows  # noqa: E402
from class_mapping import normalize_class  # noqa: E402


def robust_contrast01(img: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    v = img[valid] if valid is not None else img.reshape(-1)
    if v.size == 0:
        return np.zeros_like(img, dtype=np.float32), 0.0, 0.0
    lo, hi = np.percentile(v, (2, 98))
    if hi > lo:
        out = np.clip((img - lo) / float(hi - lo), 0.0, 1.0)
        return out.astype(np.float32), float(lo), float(hi)
    return np.zeros_like(img, dtype=np.float32), float(lo), float(hi)


def pick_bg_raster(rasters: List[Path]) -> Path:
    # preference: *_ch.tif then *_g.tif else first
    for p in rasters:
        s = p.stem.lower()
        if "_ch" in s or "lidar_ch" in s:
            return p
    for p in rasters:
        if p.stem.lower().endswith("_g"):
            return p
    return rasters[0]


UTM_BY_PREFIX = {
    "002": "36N",
    "003": "36N",
    "004": "36N",
    "006": "36N",
    "009": "36N",
    "010": "36N",
    "012": "36N",
    "013": "36N",
    "014": "37N",
    "015": "36N",
    "019": "36N",
    "020": "36N",
    "022": "37N",
    "025": "37N",
    "026": "37N",
    "027": "38N",
    "030": "37N",
    "031": "36N",
    "032": "36N",
    "034": "37N",
    "037": "37N",
    "038": "37N",
    "039": "35N",
    "040": "36N",
    "041": "36N",
    "046": "36N",
    "047": "36N",
    "048": "37N",
    "052": "38N",
    "053": "36N",
    "054": "37N",
    "056": "37N",
    "057": "37N",
    "058": "37N",
    "060": "37N",
    "061": "37N",
    "062": "37N",
    "064": "37N",
    "070": "37N",
    "073": "36N",
    "075": "35N",
    "078": "36N",
    "081": "38N",
    "082": "36N",
    "084": "46N",
    "085": "38N",
    "086": "37N",
    "087": "44N",
    "089": "38N",
    "091": "43N",
    "095": "36N",
    "098": "36N",
    "099": "36N",
}


def _extract_prefix(name: str) -> str:
    base = Path(name).name
    tok = base.split("_")[0]
    return tok if tok.isdigit() and len(tok) == 3 else tok


def _zone_to_epsg(zone_str: str) -> int:
    z = int(zone_str[:-1])
    hemi = zone_str[-1].upper()
    return (32600 if hemi == "N" else 32700) + z


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-9
    return inter / union


def load_labels_all_to_common_crs(paths: List[Path]) -> tuple[gpd.GeoDataFrame | None, CRS | None]:
    frames = []
    crs_seen: list[CRS] = []
    for p in paths:
        g = None
        try:
            g = gpd.read_file(p)
        except Exception:
            try:
                g = gpd.read_file(p, engine="fiona")
            except Exception:
                g = None
        if g is None or g.empty:
            continue
        frames.append(g)
        if g.crs is not None:
            try:
                crs_seen.append(CRS.from_user_input(g.crs))
            except Exception:
                pass
    if not frames:
        return None, None
    # prefer 3857
    target: CRS | None = None
    for g in frames:
        try:
            c = CRS.from_user_input(g.crs) if g.crs is not None else None
            if c and c.to_epsg() == 3857:
                target = CRS.from_epsg(3857)
                break
        except Exception:
            pass
    if target is None:
        if crs_seen:
            counts = pd.Series([str(c) for c in crs_seen]).value_counts()
            target = CRS.from_user_input(counts.index[0])
        else:
            target = CRS.from_epsg(3857)
    aligned = []
    for g in frames:
        try:
            if g.crs is None:
                g = g.set_crs(target)
            elif CRS.from_user_input(g.crs) != target:
                g = g.to_crs(target)
            aligned.append(g)
        except Exception:
            pass
    if not aligned:
        return None, None
    return gpd.GeoDataFrame(pd.concat(aligned, ignore_index=True), crs=target), target


def ensure_src_crs_with_labels(
    ref_path: Path, region_name: str, labels_gdf: gpd.GeoDataFrame | None, labels_crs: CRS | None
) -> CRS:
    # choose UTM EPSG by IoU
    prefix = _extract_prefix(region_name)
    base_zone = UTM_BY_PREFIX.get(prefix, "36N")
    cand = [_zone_to_epsg(base_zone)]
    try:
        z = int(base_zone[:-1])
        hemi = base_zone[-1]
        for dz in (-1, +1):
            nz = z + dz
            if 1 <= nz <= 60:
                cand.append(_zone_to_epsg(f"{nz}{hemi}"))
    except Exception:
        pass
    with rasterio.open(ref_path) as src:
        bounds_native = src.bounds
    best_epsg = cand[0]
    if labels_gdf is not None and labels_crs is not None:
        best_iou = -1.0
        lb = tuple(labels_gdf.total_bounds)
        for e in cand:
            try:
                tb = transform_bounds(CRS.from_epsg(e), labels_crs, *bounds_native, densify_pts=16)
                i = _bbox_iou(tb, lb)
                if i > best_iou:
                    best_iou, best_epsg = i, e
            except Exception:
                continue
    return CRS.from_epsg(best_epsg)


def save_png_gray(img01: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = (np.clip(img01, 0, 1) * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), im)


def save_mask_indices(mask_multi: np.ndarray, path: Path) -> None:
    # mask_multi: [C,H,W] uint8 -> single channel indices 0..C
    path.parent.mkdir(parents=True, exist_ok=True)
    C, H, W = mask_multi.shape
    idx = np.zeros((H, W), dtype=np.uint8)
    for ci in range(C):
        idx[mask_multi[ci] > 0] = max(1, min(255, ci + 1))
    cv2.imwrite(str(path), idx)


def run_export(
    train_csv: Path,
    val_csv: Path,
    out_dir: Path,
    tile: int,
    stride: int,
    classes: List[str] | None,
    pos_min_pixels: int,
    neg_per_pos: float,
    max_images: int | None,
    prefer_bg: List[str] | None,
):
    out_dir = out_dir.resolve()
    (out_dir / "images/train").mkdir(parents=True, exist_ok=True)
    (out_dir / "images/val").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks/train").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks/val").mkdir(parents=True, exist_ok=True)

    def _parse_list(s):
        if isinstance(s, list):
            return s
        return json.loads(s) if isinstance(s, str) and s.strip().startswith("[") else [s]

    def _pick_bg_from_list(paths: List[str]) -> Path:
        cands = [Path(p) for p in paths if p]
        if not cands:
            raise RuntimeError("No raster paths")
        # prefer based on substrings
        if prefer_bg:
            for pref in prefer_bg:
                for p in cands:
                    if pref.lower() in p.stem.lower():
                        return p
        return pick_bg_raster(cands)

    def _collect(df: pd.DataFrame):
        items = []
        for _, r in df.iterrows():
            region = r.get("region_name") or Path(r.get("region_dir", ".")).name
            rasters = _parse_list(r["raster_paths"]) if "raster_paths" in r and pd.notna(r["raster_paths"]) else []
            labels = _parse_list(r["labels_paths"]) if "labels_paths" in r and pd.notna(r["labels_paths"]) else []
            if len(rasters) == 0 or len(labels) == 0:
                continue
            items.append((region, rasters, labels))
        return items

    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)
    tr_items = _collect(df_tr)
    va_items = _collect(df_va)

    # discover classes if not provided; fallback to single-class 'object'
    single_class_mode = False
    if classes is None:
        single_class_mode = True
        classes = ["object"]
    else:
        # normalize provided class names
        classes = [normalize_class(c) or str(c) for c in classes]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    def _rasterize(gdf: gpd.GeoDataFrame, bounds, shape, transform) -> np.ndarray:
        C = len(classes)
        mask = np.zeros((C, shape[0], shape[1]), dtype=np.uint8)
        if gdf is None or gdf.empty:
            return mask
        bbox_poly = box(*bounds)
        sub = gdf[gdf.geometry.intersects(bbox_poly)].copy()
        if sub.empty:
            return mask
        sub["geometry"] = sub.geometry.intersection(bbox_poly)
        if single_class_mode:
            # rasterize all geometries into channel 0
            rast = rasterize(
                ((geom, 1) for geom in sub.geometry if geom is not None and not geom.is_empty),
                out_shape=shape,
                transform=transform,
                fill=0,
                all_touched=True,
            ).astype(np.uint8)
            mask[0] = rast
            return mask

        # multi-class: normalize feature class names
        def _infer_cname(row):
            if "class_name" in row and pd.notna(row["class_name"]):
                return normalize_class(str(row["class_name"])) or str(row["class_name"])
            if "label" in row and pd.notna(row["label"]):
                return normalize_class(str(row["label"])) or str(row["label"])
            if "class" in row and pd.notna(row["class"]):
                return normalize_class(str(row["class"])) or str(row["class"])
            if "name" in row and pd.notna(row["name"]):
                raw = str(row["name"]).split("_")[-1]
                return normalize_class(raw) or raw
            return "unknown"

        sub["__cname__"] = sub.apply(_infer_cname, axis=1)
        for cname in classes:
            part = sub[sub["__cname__"] == cname]
            if part.empty:
                continue
            rast = rasterize(
                ((geom, 1) for geom in part.geometry if geom is not None and not geom.is_empty),
                out_shape=shape,
                transform=transform,
                fill=0,
                all_touched=True,
            ).astype(np.uint8)
            mask[class_to_idx[cname]] |= rast
        return mask

    def _export_split(items, split_name: str) -> pd.DataFrame:
        rows = []
        saved_pos = 0
        saved_neg = 0
        for region, rasters, labels in items:
            bg = _pick_bg_from_list(rasters)
            label_paths = [Path(p) for p in labels]
            labels_gdf, labels_crs = load_labels_all_to_common_crs(label_paths)
            src_crs = ensure_src_crs_with_labels(bg, region, labels_gdf, labels_crs) if labels_gdf is not None else None
            if labels_gdf is not None and src_crs is not None:
                try:
                    if labels_gdf.crs != src_crs:
                        labels_gdf = labels_gdf.to_crs(src_crs)
                except Exception:
                    pass

            with rasterio.open(bg) as src:
                H, W = src.height, src.width
                tr = src.transform
                for y, x, h, w in sliding_windows(H, W, tile, stride):
                    if max_images is not None and (saved_pos + saved_neg) >= max_images:
                        break
                    win = Window(x, y, w, h)
                    bounds = rasterio.windows.bounds(win, tr)
                    mask = _rasterize(labels_gdf, bounds, (h, w), rasterio.windows.transform(win, tr))
                    pos_pixels = int(mask.sum())
                    is_pos = pos_pixels >= pos_min_pixels
                    if not is_pos:
                        # sample negatives with probability based on ratio and current counts
                        if saved_pos == 0:
                            take_neg = False
                        else:
                            target_neg = int(round(saved_pos * neg_per_pos))
                            take_neg = saved_neg < target_neg
                        if not take_neg:
                            continue
                    # read image
                    img = src.read(1, window=win, boundless=True).astype(np.float32)
                    # valid mask
                    try:
                        vm = src.read_masks(1, window=win) > 0
                    except Exception:
                        nod = src.nodata
                        vm = np.ones_like(img, dtype=bool) if nod is None else (img != nod)
                    img01, lo, hi = robust_contrast01(img, valid=vm)
                    # save files
                    stem = f"{region}_y{y}_x{x}"
                    img_rel = Path("images") / split_name / f"{stem}.png"
                    msk_rel = Path("masks") / split_name / f"{stem}.png"
                    save_png_gray(img01, out_dir / img_rel)
                    save_mask_indices(mask, out_dir / msk_rel)
                    rows.append(
                        {
                            "image_path": str(img_rel).replace("\\", "/"),
                            "mask_path": str(msk_rel).replace("\\", "/"),
                            "region": region,
                            "has_label": int(is_pos),
                            "class_pixels": json.dumps(
                                {c: int(mask[class_to_idx[c]].sum()) for c in classes}, ensure_ascii=False
                            ),
                            "lo": lo,
                            "hi": hi,
                        }
                    )
                    if is_pos:
                        saved_pos += 1
                    else:
                        saved_neg += 1
        print(f"[{split_name}] saved pos={saved_pos} neg={saved_neg}")
        return pd.DataFrame(rows)

    df_train = _export_split(tr_items, "train")
    df_val = _export_split(va_items, "val")
    df_train.to_csv(out_dir / "train.csv", index=False)
    df_val.to_csv(out_dir / "val.csv", index=False)
    print(f"Saved manifests to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Kaggle-ready dataset of tiles and masks with manifest")
    parser.add_argument("--train_csv", type=str, default=str(ROOT / "runs/default/train_regions.csv"))
    parser.add_argument("--val_csv", type=str, default=str(ROOT / "runs/default/val_regions.csv"))
    parser.add_argument("--out", type=str, default=str(ROOT / "runs/kaggle_dataset"))
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument(
        "--classes", type=str, default=None, help="Comma-separated class names; if omitted, auto-detect"
    )
    parser.add_argument("--pos_min_pixels", type=int, default=1, help="Min labeled pixels to treat tile as positive")
    parser.add_argument("--neg_per_pos", type=float, default=1.0, help="How many negatives per positive (approx)")
    parser.add_argument(
        "--max_images", type=int, default=None, help="Hard cap on total images (both splits separately)"
    )
    parser.add_argument(
        "--prefer_bg",
        type=str,
        default="_ch,_g",
        help="Comma-separated substrings to prefer in raster filename (e.g., _ch,_g)",
    )
    args = parser.parse_args()

    classes = [s.strip() for s in args.classes.split(",")] if args.classes else None
    prefer_bg = [s.strip() for s in args.prefer_bg.split(",") if s.strip()] if args.prefer_bg else None

    run_export(
        train_csv=Path(args.train_csv),
        val_csv=Path(args.val_csv),
        out_dir=Path(args.out),
        tile=args.tile,
        stride=args.stride,
        classes=classes,
        pos_min_pixels=args.pos_min_pixels,
        neg_per_pos=args.neg_per_pos,
        max_images=args.max_images,
        prefer_bg=prefer_bg,
    )
