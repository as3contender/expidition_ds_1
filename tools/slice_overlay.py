#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import box
import cv2


def robust_contrast01(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, (2, 98))
    if hi > lo:
        return np.clip((img - lo) / float(hi - lo), 0.0, 1.0)
    return np.zeros_like(img, dtype=np.float32)


def sliding_windows(H: int, W: int, tile: int, stride: int):
    tile = int(tile)
    stride = int(stride)
    H = int(H)
    W = int(W)
    ys = list(range(0, max(1, H - tile + 1), stride))
    end_y = max(0, H - tile)
    xs = list(range(0, max(1, W - tile + 1), stride))
    end_x = max(0, W - tile)
    if len(ys) == 0:
        ys = [0]
    elif ys[-1] != end_y:
        ys.append(end_y)
    if len(xs) == 0:
        xs = [0]
    elif xs[-1] != end_x:
        xs.append(end_x)
    for y in ys:
        for x in xs:
            yield (int(y), int(x), tile, tile)


def load_all_labels_to_common_crs(label_files: List[Path]) -> tuple[gpd.GeoDataFrame, CRS]:
    """Читает GeoJSON метки и приводит их к единому CRS (предпочитая EPSG:3857)."""
    frames: list[tuple[Path, gpd.GeoDataFrame]] = []
    crs_seen: list[CRS] = []
    for lp in label_files:
        g = None
        try:
            g = gpd.read_file(lp)
        except Exception:
            try:
                g = gpd.read_file(lp, engine="fiona")
            except Exception:
                g = None
        if g is None or g.empty:
            continue
        frames.append((lp, g))
        if g.crs is not None:
            try:
                crs_seen.append(CRS.from_user_input(g.crs))
            except Exception:
                pass

    if not frames:
        raise RuntimeError("Нет валидных GeoJSON разметок")

    # Выбираем целевой CRS: в приоритете 3857, иначе самый частый.
    target_crs: CRS | None = None
    for _, g in frames:
        try:
            crs = CRS.from_user_input(g.crs) if g.crs is not None else None
            if crs and crs.to_epsg() == 3857:
                target_crs = CRS.from_epsg(3857)
                break
        except Exception:
            pass
    if target_crs is None:
        if not crs_seen:
            target_crs = CRS.from_epsg(3857)
        else:
            counts = pd.Series([str(c) for c in crs_seen]).value_counts()
            target_crs = CRS.from_user_input(counts.index[0])

    aligned: list[gpd.GeoDataFrame] = []
    for lp, g in frames:
        try:
            if g.crs is None:
                g = g.set_crs(target_crs)
            elif CRS.from_user_input(g.crs) != target_crs:
                g = g.to_crs(target_crs)
            aligned.append(g)
        except Exception:
            continue

    if not aligned:
        raise RuntimeError("Не удалось привести метки к общему CRS")

    gdf_all = gpd.GeoDataFrame(pd.concat(aligned, ignore_index=True), crs=target_crs)
    return gdf_all, target_crs


# === UTM helpers (как в quick_overlay) ===
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


def _extract_prefix(region_name: str) -> str:
    base = Path(region_name).name
    tok = base.split("_")[0]
    return tok if tok.isdigit() and len(tok) == 3 else tok


def _zone_to_epsg(zone_str: str) -> int:
    z = int(zone_str[:-1])
    hemi = zone_str[-1].upper()
    return (32600 if hemi == "N" else 32700) + z


def find_bg_raster(raster_dir: Path) -> Path:
    tifs = sorted([p for p in raster_dir.glob("*.tif") if p.is_file()])
    if not tifs:
        raise FileNotFoundError(f"No .tif in {raster_dir}")
    for p in tifs:
        s = p.stem.lower()
        if "_ch" in s or "lidar_ch" in s:
            return p
    return tifs[0]


def overlay_region(
    region_dir: Path,
    out_dir: Path,
    tile: int,
    stride: int,
    limit: int,
    only_with_labels: bool = True,
    min_valid_frac: float = 0.05,
    min_contrast_delta: float = 1e-3,
    raster_path: Path | None = None,
    label_paths: List[Path] | None = None,
):
    region_dir = region_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raster_dir = next((p for p in region_dir.glob("02_*Li*карты")), None)
    labels_dir = next((p for p in region_dir.glob("06_*разметка/Li")), None)
    if raster_path is None:
        if raster_dir is None:
            raise FileNotFoundError("Не найдена папка 02_*Li*карты и не указан --raster")
        bg_path = find_bg_raster(raster_dir)
    else:
        bg_path = Path(raster_path)
        if not bg_path.exists():
            raise FileNotFoundError(f"Raster path not found: {bg_path}")

    if label_paths is None:
        if labels_dir is None:
            raise FileNotFoundError("Не найдена папка 06_*разметка/Li и не указаны --label")
        label_files = sorted([p for p in labels_dir.glob("*.geojson") if p.is_file()])
        if not label_files:
            raise FileNotFoundError("Нет файлов .geojson в папке разметки")
    else:
        label_files = [Path(p) for p in label_paths]
        for lp in label_files:
            if not lp.exists():
                raise FileNotFoundError(f"Label path not found: {lp}")

    # 1) Считываем метки и приводим к общему CRS (предпочтительно 3857)
    try:
        labels_gdf, labels_crs = load_all_labels_to_common_crs(label_files)
    except Exception as e:
        labels_gdf, labels_crs = None, None

    # 2) Подберём UTM-зону для нат. CRS растра и проверим пересечение экстентов
    with rasterio.open(bg_path) as src:
        H, W = src.height, src.width
        src_bounds_native = src.bounds

    prefix = _extract_prefix(region_dir.name)
    base_zone = UTM_BY_PREFIX.get(prefix)
    if base_zone is None:
        base_zone = "36N"  # безопасный дефолт для части регионов
    candidate_epsg = [_zone_to_epsg(base_zone)]
    try:
        z = int(base_zone[:-1])
        hemi = base_zone[-1]
        for dz in (-1, +1):
            nz = z + dz
            if 1 <= nz <= 60:
                candidate_epsg.append(_zone_to_epsg(f"{nz}{hemi}"))
    except Exception:
        pass

    def iou_in_labels_crs(epsg: int) -> float:
        try:
            tb = transform_bounds(CRS.from_epsg(epsg), labels_crs, *src_bounds_native, densify_pts=16)
        except Exception:
            return 0.0
        # IoU bbox tb vs labels total bounds
        if labels_gdf is None or labels_gdf.empty:
            return 0.0
        lb = tuple(labels_gdf.total_bounds)
        ax1, ay1, ax2, ay2 = tb
        bx1, by1, bx2, by2 = lb
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter + 1e-9
        return float(inter / union)

    best_epsg = candidate_epsg[0]
    if labels_crs is not None:
        scores = [(epsg, iou_in_labels_crs(epsg)) for epsg in candidate_epsg]
        scores.sort(key=lambda t: t[1], reverse=True)
        best_epsg = scores[0][0]

    src_crs = CRS.from_epsg(best_epsg)

    # 3) Приведём метки к CRS растра
    if labels_gdf is not None and not labels_gdf.empty:
        try:
            if labels_gdf.crs is None:
                labels_gdf = labels_gdf.set_crs(src_crs)
            elif labels_gdf.crs != src_crs:
                labels_gdf = labels_gdf.to_crs(src_crs)
        except Exception:
            labels_gdf = None

    gdf = labels_gdf

    saved = 0
    with rasterio.open(bg_path) as src:
        for y, x, h, w in sliding_windows(H=src.height, W=src.width, tile=tile, stride=stride):
            if saved >= limit:
                break
            win = Window(x, y, w, h)

            # read data and mask to detect nodata/empty
            img = src.read(1, window=win, boundless=True).astype(np.float32)
            try:
                m = src.read_masks(1, window=win)
                valid = m > 0
            except Exception:
                nod = src.nodata
                valid = np.ones_like(img, dtype=bool) if nod is None else (img != nod)

            valid_frac = float(valid.mean()) if valid.size > 0 else 0.0
            if valid_frac < min_valid_frac:
                # skip almost empty tiles
                continue

            # contrast normalize using only valid pixels
            v = img[valid]
            if v.size == 0:
                continue
            lo, hi = np.percentile(v, (2, 98))
            if (hi - lo) < min_contrast_delta:
                # skip flat tiles
                continue
            img = np.clip((img - lo) / float(hi - lo), 0.0, 1.0)
            bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # label intersection check
            wb = rasterio.windows.bounds(win, src.transform)
            has_labels = False
            sel = None
            if gdf is not None and not gdf.empty:
                sel = gdf[gdf.geometry.intersects(box(*wb))].copy()
                has_labels = not sel.empty
            if only_with_labels and not has_labels:
                continue

            overlay = bgr.copy()
            if has_labels:
                sel["geometry"] = sel.geometry.intersection(box(*wb))
                rast = rasterize(
                    ((geom, 1) for geom in sel.geometry if geom is not None and not geom.is_empty),
                    out_shape=(h, w),
                    transform=rasterio.windows.transform(win, src.transform),
                    fill=0,
                    all_touched=True,
                ).astype(np.uint8)
                color = np.array([0, 128, 255], dtype=np.uint8)
                layer = np.zeros_like(overlay)
                layer[rast == 1] = color
                overlay = cv2.addWeighted(overlay, 1.0, layer, 0.45, 0)

            out_path = out_dir / f"{region_dir.name}_slice_y{y}_x{x}.png"
            cv2.imwrite(str(out_path), overlay)
            saved += 1

    print(f"Saved {saved} overlays to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скользящая нарезка тайлов с наложением разметки для одной территории")
    parser.add_argument("region_dir", type=str, help="Путь к каталогу региона *_FINAL")
    parser.add_argument("--out", type=str, default="runs/debug_overlays", help="Куда сохранить PNG")
    parser.add_argument("--tile", type=int, default=512, help="Размер тайла, пикс")
    parser.add_argument("--stride", type=int, default=512, help="Шаг окна, пикс")
    parser.add_argument("--limit", type=int, default=50, help="Лимит числа сохранённых тайлов")
    parser.add_argument("--only_with_labels", action="store_true", help="Сохранять только окна, где есть разметка")
    parser.add_argument("--min_valid_frac", type=float, default=0.05, help="Мин. доля валидных пикселей (по маске)")
    parser.add_argument("--min_contrast_delta", type=float, default=1e-3, help="Мин. динамический диапазон 98-2 перц.")
    parser.add_argument("--raster", type=str, default=None, help="Явный путь к одному TIF для фона")
    parser.add_argument(
        "--label",
        type=str,
        action="append",
        default=None,
        help="Путь к GeoJSON метке; можно указывать несколько параметром --label",
    )
    args = parser.parse_args()

    overlay_region(
        Path(args.region_dir),
        Path(args.out),
        tile=args.tile,
        stride=args.stride,
        limit=args.limit,
        only_with_labels=args.only_with_labels,
        min_valid_frac=args.min_valid_frac,
        min_contrast_delta=args.min_contrast_delta,
        raster_path=Path(args.raster) if args.raster else None,
        label_paths=[Path(p) for p in args.label] if args.label else None,
    )
