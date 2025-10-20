#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.features import rasterize
from shapely.geometry import box
import shapely
import cv2

# ==========================
#   Справочник UTM по папкам
# ==========================
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

PALETTE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (128, 128, 128),
]


# ==========================
#          Утилиты
# ==========================
def _extract_prefix(region_name: str) -> str:
    base = Path(region_name).name
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


def _sliding_windows(H: int, W: int, tile: int, stride: int):
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


def _pick_bg(rasters: List[Path]) -> Path:
    for p in rasters:
        s = p.stem.lower()
        if "_ch" in s or "lidar_ch" in s:
            return p
    return rasters[0]


# ==========================
#   Загрузка меток в 1 CRS
# ==========================
def load_all_labels_to_common_crs(label_files: List[Path]) -> tuple[gpd.GeoDataFrame, CRS]:
    """Приводит каждый GeoJSON к единому CRS (предпочитает EPSG:3857)."""
    raw_gdfs: list[tuple[Path, gpd.GeoDataFrame]] = []
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
            print(f"[labels][EMPTY] {lp.name}")
            continue
        raw_gdfs.append((lp, g))
        if g.crs is not None:
            try:
                crs_seen.append(CRS.from_user_input(g.crs))
            except Exception:
                pass
    if not raw_gdfs:
        raise RuntimeError("Нет валидных GeoJSON.")

    # целевой CRS: сперва ищем 3857, иначе — самый частый
    target_crs: CRS | None = None
    for _, g in raw_gdfs:
        try:
            crs = CRS.from_user_input(g.crs) if g.crs is not None else None
            if crs and crs.to_epsg() == 3857:
                target_crs = CRS.from_epsg(3857)
                break
        except Exception:
            pass
    if target_crs is None:
        if not crs_seen:
            # если у всех нет CRS — примем 3857 (распространённый для твоих меток)
            target_crs = CRS.from_epsg(3857)
        else:
            cnt = Counter(str(c) for c in crs_seen)
            common_str, _ = cnt.most_common(1)[0]
            target_crs = CRS.from_user_input(common_str)

    print(f"[LABELS][target_crs] {target_crs}")

    frames = []
    for lp, g in raw_gdfs:
        try:
            if g.crs is None:
                g = g.set_crs(target_crs)
            elif CRS.from_user_input(g.crs) != target_crs:
                g = g.to_crs(target_crs)
            frames.append(g)
        except Exception as e:
            print(f"[labels][CRS-WARN] {lp.name}: {e} — пропуск")

    if not frames:
        raise RuntimeError("Все GeoJSON не удалось привести к единому CRS.")

    gdf_all = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=target_crs)
    return gdf_all, target_crs


# ==========================
#          Main
# ==========================
def main(region_dir: Path, out_dir: Path, tile: int, stride: int, limit: int):
    region_dir = region_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Найдём папки/файлы
    raster_dir = next((p for p in region_dir.glob("02_*Li*карты")), None)
    labels_dir = next((p for p in region_dir.glob("06_*разметка/Li")), None)
    if raster_dir is None or labels_dir is None:
        raise FileNotFoundError("Нет подпапок 02_*Li*карты и/или 06_*разметка/Li")

    rasters = sorted([p for p in raster_dir.glob("*.tif") if p.is_file()])
    labels = sorted([p for p in labels_dir.glob("*.geojson") if p.is_file()])
    if not rasters or not labels:
        raise FileNotFoundError("Нет .tif или .geojson файлов")

    # 2) Загрузим все метки в один CRS (предпочтительно 3857)
    gdf_all, labels_crs = load_all_labels_to_common_crs(labels)
    labels_bounds = tuple(gdf_all.total_bounds)
    print(f"[LABELS] feats={len(gdf_all)} crs={labels_crs} bounds={labels_bounds}")
    print(f"[LABELS] geom_types={dict(pd.Series(gdf_all.geometry.geom_type).value_counts())}")

    # 3) Выберем исходный CRS растра из таблицы и при необходимости докрутим ±1 зону
    ref_path = rasters[0]
    prefix = _extract_prefix(region_dir.name)
    if prefix not in UTM_BY_PREFIX:
        raise RuntimeError(f"Нет UTM записи для '{prefix}'")

    base_zone = UTM_BY_PREFIX[prefix]
    src_crs_guess = CRS.from_epsg(_zone_to_epsg(base_zone))

    # Оценим IoU экстентов: src.bounds → labels_crs, предполагая, что цифры src.bounds — в src_crs_guess
    with rasterio.open(ref_path) as src:
        H, W = src.height, src.width
        src_bounds = src.bounds

    def iou_for_zone(zone_epsg: int) -> float:
        try:
            tb = transform_bounds(CRS.from_epsg(zone_epsg), labels_crs, *src_bounds, densify_pts=16)
            return _bbox_iou(tb, labels_bounds)
        except Exception:
            return 0.0

    iou_base = iou_for_zone(src_crs_guess.to_epsg())
    best_iou, best_epsg = iou_base, src_crs_guess.to_epsg()
    for dz in (-1, +1):
        z = int(base_zone[:-1]) + dz
        if 1 <= z <= 60:
            epsg = _zone_to_epsg(f"{z}{base_zone[-1]}")
            i = iou_for_zone(epsg)
            if i > best_iou:
                best_iou, best_epsg = i, epsg

    src_crs = CRS.from_epsg(best_epsg)
    note = "OK" if best_iou >= 0.05 else f"LOW IoU={best_iou:.3f}"
    print(f"[CRS-AUTO] base {base_zone} → {src_crs.to_string()} ({note})")

    # 4) Переведём все метки в CRS растра (нативный); растеризуем строго в пиксельной сетке TIF
    gdf_src = gdf_all.to_crs(src_crs)

    # 5) Выберем фоновый канал (предпочтительно *_ch.tif)
    bg_path = _pick_bg(rasters)

    saved = 0
    with rasterio.open(bg_path) as src:
        tr = src.transform

        # 5a) Сначала — окна вокруг центроидов самых крупных объектов (гарантированное попадание)
        gdf_src["area_like"] = gdf_src.geometry.area.fillna(0) + gdf_src.geometry.length.fillna(0)
        for geom in gdf_src.sort_values("area_like", ascending=False).geometry.head(max(limit, 5)):
            if geom is None or geom.is_empty:
                continue
            try:
                c = geom.centroid
            except Exception:
                minx, miny, maxx, maxy = geom.bounds
                c = shapely.geometry.Point((minx + maxx) / 2, (miny + maxy) / 2)

            # world(x,y) → pixel(col,row) через обратный аффин
            col_f, row_f = (~tr) * (c.x, c.y)
            row0 = int(np.clip(int(round(row_f)) - tile // 2, 0, max(0, src.height - tile)))
            col0 = int(np.clip(int(round(col_f)) - tile // 2, 0, max(0, src.width - tile)))
            win = Window(col0, row0, tile, tile)

            # чтение окна с паддингом (если вылезаем за край)
            full = Window(0, 0, src.width, src.height)
            inter = win.intersection(full)
            if inter.width <= 0 or inter.height <= 0:
                continue
            img_part = src.read(1, window=inter).astype(np.float32)
            img = np.zeros((tile, tile), dtype=np.float32)
            dy = int(round(inter.row_off - win.row_off))
            dx = int(round(inter.col_off - win.col_off))
            img[dy : dy + img_part.shape[0], dx : dx + img_part.shape[1]] = img_part

            # контраст фоновой подложки
            lo, hi = np.percentile(img, (2, 98))
            img = np.clip((img - lo) / (max(1e-6, hi - lo)), 0, 1) if hi > lo else np.zeros_like(img)
            bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # растеризация всех геометрий, попавших в окно (в CRS растра!)
            bounds = rasterio.windows.bounds(win, tr)
            sel = gdf_src[gdf_src.geometry.intersects(box(*bounds))].copy()
            overlay = bgr.copy()
            if not sel.empty:
                sel["geometry"] = sel.geometry.intersection(box(*bounds))
                rast = rasterize(
                    ((g, 1) for g in sel.geometry if g and not g.is_empty),
                    out_shape=(tile, tile),
                    transform=rasterio.windows.transform(win, tr),
                    fill=0,
                    all_touched=True,
                ).astype(np.uint8)
                layer = np.zeros_like(overlay)
                layer[rast == 1] = np.array(PALETTE[0], dtype=np.uint8)
                overlay = cv2.addWeighted(overlay, 1.0, layer, 0.45, 0)

            out_path = out_dir / f"{region_dir.name}_centroid_{saved + 1}.png"
            cv2.imwrite(str(out_path), overlay)
            print(f"[save-centroid] {out_path}")
            saved += 1
            if saved >= limit:
                break

        # 5b) Если центроиды вдруг не дали попаданий — перебор по сетке (в CRS растра)
        if saved == 0:
            print("[FALLBACK] перебор по сетке (native CRS)…")
            for y, x, h, w in _sliding_windows(src.height, src.width, tile, stride):
                win = Window(x, y, w, h)
                bounds = rasterio.windows.bounds(win, tr)
                if not gdf_src.geometry.intersects(box(*bounds)).any():
                    continue
                img = src.read(1, window=win).astype(np.float32)
                lo, hi = np.percentile(img, (2, 98))
                img = np.clip((img - lo) / (max(1e-6, hi - lo)), 0, 1) if hi > lo else np.zeros_like(img)
                bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

                sel = gdf_src[gdf_src.geometry.intersects(box(*bounds))].copy()
                overlay = bgr.copy()
                if not sel.empty:
                    sel["geometry"] = sel.geometry.intersection(box(*bounds))
                    rast = rasterize(
                        ((g, 1) for g in sel.geometry if g and not g.is_empty),
                        out_shape=(tile, tile),
                        transform=rasterio.windows.transform(win, tr),
                        fill=0,
                        all_touched=True,
                    ).astype(np.uint8)
                    layer = np.zeros_like(overlay)
                    layer[rast == 1] = np.array(PALETTE[0], dtype=np.uint8)
                    overlay = cv2.addWeighted(overlay, 1.0, layer, 0.45, 0)
                out_path = out_dir / f"{region_dir.name}_grid_{saved + 1}.png"
                cv2.imwrite(str(out_path), overlay)
                print(f"[save-grid] {out_path}")
                saved += 1
                if saved >= limit:
                    break


# ==========================
#        CLI
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Отладочный оверлей меток на hillshade для одной папки региона.")
    parser.add_argument("region_dir", type=str, help="Путь к папке региона (например, data/train/014_КАБЛУКОВО_FINAL)")
    parser.add_argument("--out", type=str, default="runs/debug_overlays", help="Куда сохранить PNG")
    parser.add_argument("--tile", type=int, default=512, help="Размер тайла")
    parser.add_argument("--stride", type=int, default=512, help="Шаг по сетке (для fallback)")
    parser.add_argument("--limit", type=int, default=5, help="Сколько оверлеев сохранить")
    args = parser.parse_args()

    main(Path(args.region_dir), Path(args.out), tile=args.tile, stride=args.stride, limit=args.limit)
