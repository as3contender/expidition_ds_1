#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Собирает мини-индекс (train/val) с заданным количеством тайлов для быстрого обучения.
Берёт данные в формате:
  data/train/<REGION>/{02_*Li*карты/*.tif, 06_*_разметка/Li/*.geojson}

Выход:
  runs/mini/mini_train.csv
  runs/mini/mini_val.csv
которые совместимы с твоим train-пайплайном (колонки: region_name, raster_paths, labels_paths).

Запуск (пример):
  python -m tools.make_mini_index --train 80 --val 20 --out runs/mini
  python -m tools.make_mini_index --train 100 --val 30 --tile 512 --stride 384 --out runs/mini_overlap
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.geometry import box
import shapely

# ---- UTM по префиксу папки (как мы уже использовали) ----
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


def _bbox_iou(a, b):
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


def _sliding_windows(H: int, W: int, tile: int, stride: int):
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
            yield y, x, tile, tile


def _pick_bg(rasters: List[Path]) -> Path:
    for p in rasters:
        s = p.stem.lower()
        if "_ch" in s or "lidar_ch" in s:
            return p
    return rasters[0]


def _load_labels_to_common_crs(label_files: List[Path]) -> tuple[gpd.GeoDataFrame, CRS]:
    # приводим все GeoJSON к одному CRS (предпочтительно 3857)
    frames, crs_seen = [], []
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
        frames.append(g)
        if g.crs is not None:
            try:
                crs_seen.append(CRS.from_user_input(g.crs))
            except:
                pass
    if not frames:
        raise RuntimeError("Нет валидных GeoJSON.")
    target = None
    for g in frames:
        try:
            crs = CRS.from_user_input(g.crs) if g.crs is not None else None
            if crs and crs.to_epsg() == 3857:
                target = CRS.from_epsg(3857)
                break
        except:
            pass
    if target is None:
        if not crs_seen:
            target = CRS.from_epsg(3857)
        else:
            from collections import Counter

            most_str, _ = Counter(str(c) for c in crs_seen).most_common(1)[0]
            target = CRS.from_user_input(most_str)
    frames2 = []
    for g in frames:
        if g.crs is None:
            g = g.set_crs(target)
        elif CRS.from_user_input(g.crs) != target:
            g = g.to_crs(target)
        frames2.append(g)
    return gpd.GeoDataFrame(pd.concat(frames2, ignore_index=True), crs=target), target


def _choose_src_crs_by_iou(ref_tif: Path, labels_crs: CRS, labels_bounds, base_zone: str) -> CRS:
    with rasterio.open(ref_tif) as src:
        src_bounds = src.bounds
    base_epsg = _zone_to_epsg(base_zone)
    best_epsg, best_iou = base_epsg, 0.0
    for dz in (0, -1, +1):
        z = int(base_zone[:-1]) + dz
        if not (1 <= z <= 60):
            continue
        epsg = (32600 if base_zone[-1].upper() == "N" else 32700) + z
        try:
            tb = transform_bounds(CRS.from_epsg(epsg), labels_crs, *src_bounds, densify_pts=16)
            iou = _bbox_iou(tb, labels_bounds)
            if iou > best_iou:
                best_iou, best_epsg = iou, epsg
        except Exception:
            pass
    return CRS.from_epsg(best_epsg)


def _count_tiles_with_labels(region_dir: Path, tile: int, stride: int) -> int:
    """Подсчитывает количество тайлов с пересечением разметки в регионе."""
    raster_dir = next((p for p in region_dir.glob("02_*Li*карты")), None)
    labels_dir = next((p for p in region_dir.glob("06_*разметка/Li")), None)
    if raster_dir is None or labels_dir is None:
        return 0

    rasters = sorted([p for p in raster_dir.glob("*.tif") if p.is_file()])
    labels = sorted([p for p in labels_dir.glob("*.geojson") if p.is_file()])
    if not rasters or not labels:
        return 0

    try:
        gdf_all, _ = _load_labels_to_common_crs(labels)
        if gdf_all.empty:
            return 0

        # Открываем один растр для получения размеров
        ref_tif = _pick_bg(rasters)
        with rasterio.open(ref_tif) as src:
            H, W = src.height, src.width

        # Подсчитываем тайлы
        count = 0
        for y, x, th, tw in _sliding_windows(H, W, tile, stride):
            count += 1

        return count
    except Exception:
        return 0


def build_mini_index(
    regions_root: Path,
    n_train: int,
    n_val: int,
    tile: int,
    stride: int,
    out_dir: Path,
    prefer_regions: List[str] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Собирает мини-индекс с заданным количеством тайлов для train и val.

    Args:
        regions_root: корневая папка с регионами
        n_train: примерное количество тайлов для train
        n_val: примерное количество тайлов для val
        tile: размер тайла
        stride: шаг окна
        out_dir: папка для сохранения CSV
        prefer_regions: список предпочитаемых регионов (по префиксу)
        seed: random seed
    """
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Найдём все регионы с данными
    regions = sorted([p for p in regions_root.iterdir() if p.is_dir()])
    if prefer_regions:
        prset = set(prefer_regions)
        preferred = [p for p in regions if any(p.name.startswith(x) for x in prset)]
        others = [p for p in regions if not any(p.name.startswith(x) for x in prset)]
        regions = preferred + others

    # 2) Собираем регионы с подсчётом тайлов
    region_data = []
    print(f"[scan] Сканирую регионы...")
    for region_dir in regions:
        raster_dir = next((p for p in region_dir.glob("02_*Li*карты")), None)
        labels_dir = next((p for p in region_dir.glob("06_*разметка/Li")), None)
        if raster_dir is None or labels_dir is None:
            continue

        rasters = sorted([p for p in raster_dir.glob("*.tif") if p.is_file()])
        labels = sorted([p for p in labels_dir.glob("*.geojson") if p.is_file()])
        if not rasters or not labels:
            continue

        # Проверяем наличие разметки
        try:
            gdf_all, _ = _load_labels_to_common_crs(labels)
            if gdf_all.empty:
                continue
        except Exception:
            continue

        # Подсчитываем примерное количество тайлов
        n_tiles = _count_tiles_with_labels(region_dir, tile, stride)

        region_data.append(
            {
                "region_dir": region_dir,
                "region_name": region_dir.name,
                "rasters": rasters,
                "labels": labels,
                "n_tiles": n_tiles,
            }
        )
        print(f"  {region_dir.name}: ~{n_tiles} тайлов")

    if not region_data:
        raise RuntimeError("Не удалось найти ни одного региона с данными.")

    # 3) Распределяем регионы между train и val
    # Для train берём первые регионы до набора n_train тайлов
    # Для val берём следующие до n_val тайлов
    train_regions = []
    val_regions = []
    train_tiles = 0
    val_tiles = 0

    for rd in region_data:
        if train_tiles < n_train:
            train_regions.append(rd)
            train_tiles += rd["n_tiles"]
        elif val_tiles < n_val:
            val_regions.append(rd)
            val_tiles += rd["n_tiles"]
        else:
            break

    if not train_regions:
        raise RuntimeError("Не удалось набрать регионы для train.")
    if not val_regions:
        # Если нет отдельных регионов для val, используем последний train-регион
        print("[warning] Недостаточно регионов для val, используем последний train-регион")
        val_regions = [train_regions[-1]]

    # 4) Формируем DataFrame в формате для SlidingGeoDataset
    def make_df(regions_list):
        rows = []
        for rd in regions_list:
            rows.append(
                {
                    "region_name": rd["region_name"],
                    "raster_paths": json.dumps([str(p) for p in rd["rasters"]], ensure_ascii=False),
                    "labels_paths": json.dumps([str(p) for p in rd["labels"]], ensure_ascii=False),
                }
            )
        return pd.DataFrame(rows)

    df_tr = make_df(train_regions)
    df_val = make_df(val_regions)

    # 5) Сохраняем
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mini_train.csv").write_text(df_tr.to_csv(index=False))
    (out_dir / "mini_val.csv").write_text(df_val.to_csv(index=False))

    print(f"\n[done] Создано:")
    print(f"  train: {len(train_regions)} регионов (~{train_tiles} тайлов) → {out_dir / 'mini_train.csv'}")
    print(f"  val:   {len(val_regions)} регионов (~{val_tiles} тайлов) → {out_dir / 'mini_val.csv'}")

    return df_tr, df_val


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Создаёт мини-индекс train/val с заданным количеством тайлов (примерно)")
    ap.add_argument("--regions_root", type=str, default="data/train", help="Путь к папке с регионами")
    ap.add_argument("--train", type=int, required=True, help="Примерное количество тайлов для train")
    ap.add_argument("--val", type=int, required=True, help="Примерное количество тайлов для val")
    ap.add_argument("--tile", type=int, default=512, help="Размер тайла (по умолчанию 512)")
    ap.add_argument("--stride", type=int, default=512, help="Шаг скользящего окна (по умолчанию 512)")
    ap.add_argument("--out", type=str, default="runs/mini", help="Папка для сохранения CSV (по умолчанию runs/mini)")
    ap.add_argument(
        "--prefer_regions",
        type=str,
        nargs="*",
        default=None,
        help="Предпочитаемые регионы (по префиксу, например: 014 002)",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    build_mini_index(
        regions_root=Path(args.regions_root),
        n_train=args.train,
        n_val=args.val,
        tile=args.tile,
        stride=args.stride,
        out_dir=Path(args.out),
        prefer_regions=args.prefer_regions,
        seed=args.seed,
    )
