# src/dataset_sliding.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import math
import json
import numpy as np
import pandas as pd

import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.features import rasterize

import geopandas as gpd
from shapely.geometry import box
from shapely.strtree import STRtree

import cv2
import albumentations as A
import torch
from torch.utils.data import Dataset


def _to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _safe_crs_equal(crs_a, crs_b) -> bool:
    try:
        return (
            (crs_a is not None)
            and (crs_b is not None)
            and (rasterio.crs.CRS.from_user_input(crs_a) == rasterio.crs.CRS.from_user_input(crs_b))
        )
    except Exception:
        return False


def _compute_pixel_size_m(transform: Affine) -> float:
    try:
        px_x = abs(transform.a)
        px_y = abs(transform.e)
        px = float((px_x + px_y) / 2.0)
        if px <= 0 or not np.isfinite(px):
            return 1.0
        return px
    except Exception:
        return 1.0


def _build_simple_aug(augment: bool, tile: int):
    if not augment:
        return A.Compose([A.Normalize()])
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, rotate=(-5, 5), shear={"x": (-3, 3), "y": (-3, 3)}, p=0.5
            ),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.25),
            A.Normalize(),
        ]
    )


class SlidingGeoDataset(Dataset):
    """
    index_df: DataFrame со строками по регионам.
        Ожидаемые колонки (гибкие, но желательно такие):
            - region_dir: str (путь к папке региона)
            - raster_paths: json-список путей (4 hillshade tif)
            - labels_paths: json-список geojson по классам (можно пусто)
    classes: список имён классов (порядок каналов маски).
    tile: размер окна (px)
    stride: шаг (px)
    class_buffers_m: {class_name: buffer_m} — необязательно
    boundary_mode: {class_name: {enabled: bool, ring_width_m: float}}
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        classes: List[str],
        tile: int,
        stride: int,
        augment: bool = False,
        class_buffers_m: Optional[Dict[str, float]] = None,
        boundary_mode: Optional[Dict[str, Dict]] = None,
    ):

        self.classes = list(classes)
        self.cls_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.tile = int(tile)
        self.stride = int(stride)
        self.class_buffers_m = class_buffers_m or {}
        self.boundary_mode = boundary_mode or {}

        self.aug = _build_simple_aug(augment, tile=self.tile)

        # --- нормализуем пути и поля ---
        df = index_df.copy()
        if "raster_paths" in df.columns:
            # может быть json-строкой
            df["raster_paths"] = df["raster_paths"].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith("[") else _to_list(x)
            )
        if "labels_paths" in df.columns:
            df["labels_paths"] = df["labels_paths"].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith("[") else _to_list(x)
            )
        if "region_dir" in df.columns:
            df["region_dir"] = df["region_dir"].apply(lambda p: str(p))

        # --- строим индексы тайлов и кэш по регионам ---
        self.index: List[Dict] = []
        self.region_meta: Dict[str, Dict] = {}  # region -> {crs, transform, bounds, pixel_size_m}
        self.region_labels: Dict[str, gpd.GeoDataFrame] = {}  # region -> gdf
        self.region_sindex: Dict[str, object] = {}  # region -> spatial index

        # --- UTM per-region hints (fallback for missing raster CRS) ---
        UTM_BY_PREFIX: Dict[str, str] = {
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

        for _, row in df.iterrows():
            region_name = row.get("region_name") or row.get("region") or Path(row.get("region_dir", ".")).name
            region_dir = Path(row.get("region_dir", "."))
            raster_paths = [str(Path(p)) for p in row.get("raster_paths", [])]
            labels_paths = [str(Path(p)) for p in row.get("labels_paths", [])]

            if not raster_paths:
                continue

            # Открываем первый растер (референс для сетки)
            with rasterio.open(raster_paths[0]) as ref:
                width, height = ref.width, ref.height
                transform = ref.transform
                crs = ref.crs
                bounds = ref.bounds
                px_size_m = _compute_pixel_size_m(transform)

            self.region_meta[region_name] = {
                "crs": crs,
                "transform": transform,
                "bounds": bounds,
                "pixel_size_m": px_size_m,
                "raster_paths": raster_paths,
                "region_dir": str(region_dir),
            }

            # Подгружаем и объединяем все geojson для региона (если есть).
            gdf_all = None
            labels_crs_seen: List[CRS] = []
            for gp in labels_paths:
                try:
                    gdf = gpd.read_file(gp)
                    if gdf is None or gdf.empty:
                        continue
                    gdf = gdf[gdf.geometry.notna()].copy()
                    # Запомним исходные CRS меток (для автоподбора UTM при отсутствии CRS у растра)
                    try:
                        if gdf.crs is not None:
                            labels_crs_seen.append(CRS.from_user_input(gdf.crs))
                    except Exception:
                        pass
                    # Приведение CRS пока откладываем (если у растра нет CRS)
                    if crs is not None and not _safe_crs_equal(gdf.crs, crs):
                        try:
                            gdf = gdf.to_crs(crs)
                        except Exception:
                            pass
                    # небольшой positive buffer для линейных объектов (если задан class_buffers_m)
                    if "name" in gdf.columns:
                        # если строка «..._класс» — достанем имя класса
                        # но надежнее — не трогать здесь; буфер применим позже в растеризации per-class
                        pass
                    gdf_all = gdf if gdf_all is None else pd.concat([gdf_all, gdf], ignore_index=True)
                except Exception:
                    continue

            # Если у растра нет CRS — попробуем подобрать UTM по префиксу, сверяясь с экстентом меток
            if (crs is None) and (gdf_all is not None) and (not gdf_all.empty):
                prefix = _extract_prefix(region_name)
                base_zone = UTM_BY_PREFIX.get(prefix)
                candidate_epsg: List[int] = []
                if base_zone:
                    candidate_epsg.append(_zone_to_epsg(base_zone))
                    try:
                        z = int(base_zone[:-1])
                        hemi = base_zone[-1]
                        for dz in (-1, +1):
                            nz = z + dz
                            if 1 <= nz <= 60:
                                candidate_epsg.append(_zone_to_epsg(f"{nz}{hemi}"))
                    except Exception:
                        pass
                # labels target CRS (предпочтительно наиболее частый)
                labels_target: Optional[CRS] = None
                if labels_crs_seen:
                    vals = pd.Series([str(c) for c in labels_crs_seen]).value_counts()
                    labels_target = CRS.from_user_input(vals.index[0])

                # Выберем epsg с наибольшим IoU bbox'ов (в CRS меток)
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

                best_epsg = None
                best_iou = -1.0
                if candidate_epsg and labels_target is not None:
                    for epsg in candidate_epsg:
                        try:
                            tb = transform_bounds(CRS.from_epsg(epsg), labels_target, *bounds, densify_pts=16)
                            iou = float(_bbox_iou(tb, tuple(gdf_all.to_crs(labels_target).total_bounds)))
                            if iou > best_iou:
                                best_iou, best_epsg = iou, epsg
                        except Exception:
                            continue
                if best_epsg is not None:
                    crs = CRS.from_epsg(best_epsg)
                    # обновим мета
                    self.region_meta[region_name]["crs"] = crs
                else:
                    # как минимум выставим базовую зону, если есть
                    if base_zone:
                        crs = CRS.from_epsg(_zone_to_epsg(base_zone))
                        self.region_meta[region_name]["crs"] = crs

            if gdf_all is not None and not gdf_all.empty:
                gdf_all = gdf_all[gdf_all.geometry.notna()].copy()
                # Теперь окончательно приводим к CRS региона (учитывая возможный автоподбор)
                reg_crs = self.region_meta[region_name]["crs"]
                if reg_crs is not None and not _safe_crs_equal(gdf_all.crs, reg_crs):
                    try:
                        gdf_all = gdf_all.to_crs(reg_crs)
                    except Exception:
                        pass
                self.region_labels[region_name] = gdf_all
                try:
                    self.region_sindex[region_name] = gdf_all.sindex
                except Exception:
                    self.region_sindex[region_name] = STRtree(gdf_all.geometry.values)
            else:
                self.region_labels[region_name] = None
                self.region_sindex[region_name] = None

            # Индексация окон (скользящее окно с перекрытием)
            for row_off in range(0, height - self.tile + 1, self.stride):
                for col_off in range(0, width - self.tile + 1, self.stride):
                    win = Window(col_off=col_off, row_off=row_off, width=self.tile, height=self.tile)
                    # bounds окна в координатах проекта
                    x0, y0 = transform * (col_off, row_off)
                    x1, y1 = transform * (col_off + self.tile, row_off + self.tile)
                    xmin, xmax = min(x0, x1), max(x0, x1)
                    ymin, ymax = min(y0, y1), max(y0, y1)

                    meta = {
                        "region": region_name,
                        "raster_paths": raster_paths,
                        "labels_paths": labels_paths,
                        "window": (col_off, row_off, self.tile, self.tile),
                        "bounds": (xmin, ymin, xmax, ymax),
                        "transform": transform,
                    }
                    self.index.append(meta)

        # Быстрая пометка окон с разметкой (bbox-пересечение через sindex)
        non_empty = []
        for i, meta in enumerate(self.index):
            reg = meta["region"]
            sidx = self.region_sindex.get(reg)
            gdf = self.region_labels.get(reg)
            if sidx is None or gdf is None or gdf is None or len(gdf) == 0:
                meta["has_label"] = False
                continue
            xmin, ymin, xmax, ymax = meta["bounds"]
            win_poly = box(xmin, ymin, xmax, ymax)
            has = False
            try:
                # geopandas sindex
                cand = list(sidx.intersection(win_poly.bounds))
                if cand:
                    if gdf.iloc[cand].intersects(win_poly).any():
                        has = True
            except Exception:
                # STRtree
                try:
                    tree = sidx if isinstance(sidx, STRtree) else None
                    if tree is not None:
                        cand = tree.query(win_poly)
                        has = len(cand) > 0
                except Exception:
                    has = False
            meta["has_label"] = bool(has)
            if has:
                non_empty.append(i)

        self.non_empty_indices = non_empty
        print(f"Prepared {len(self.index)} tiles from {len(self.region_meta)} regions (skipped 0).")

        # pixel size для boundary-mode (берём из первого региона как дефолт, но при растеризации будем брать per-region)
        if len(self.region_meta) > 0:
            first_reg = next(iter(self.region_meta.keys()))
            self.pixel_size_m = float(self.region_meta[first_reg]["pixel_size_m"])
        else:
            self.pixel_size_m = 1.0

    # ---------- utils ----------
    def __len__(self) -> int:
        return len(self.index)

    def _apply_boundary_mode(self, class_name: str, mask_c: np.ndarray, pixel_size_m: float) -> np.ndarray:
        bm = (self.boundary_mode or {}).get(class_name, {})
        if not bm or not bm.get("enabled", False):
            return mask_c
        ring_m = float(bm.get("ring_width_m", 3.0))
        px = max(1, int(round(ring_m / max(pixel_size_m, 1e-6))))
        k = np.ones((3, 3), np.uint8)
        edge = cv2.morphologyEx(mask_c.astype(np.uint8), cv2.MORPH_GRADIENT, k, iterations=px)
        return (edge > 0).astype(np.uint8)

    def _read_multi_channel_window(self, raster_paths: List[str], window: Tuple[int, int, int, int]) -> np.ndarray:
        col_off, row_off, w, h = window
        target_shape = (h, w)
        chans = []

        for rp in raster_paths:
            with rasterio.open(rp) as ds:
                # читаем единственный канал (boundless=True заполняет области за границами нулями)
                arr = ds.read(1, window=Window(col_off, row_off, w, h), boundless=True).astype(np.float32)

                # Проверяем, что массив не пустой
                if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
                    # Если пустой - создаём нулевой массив нужного размера
                    arr = np.zeros(target_shape, dtype=np.float32)
                elif arr.shape != target_shape:
                    # Приводим к целевому размеру, если отличается
                    arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)

                # нормализация в [0,1] по per-window (простая, но устойчивая для hillshade)
                if arr.size > 0:
                    mn = float(arr.min())
                    mx = float(arr.max())
                    if mx > mn:
                        arr = (arr - mn) / (mx - mn)
                    else:
                        arr = np.zeros_like(arr, dtype=np.float32)

                chans.append(arr)

        img = np.stack(chans, axis=0)  # [C,H,W]
        return img

    def _rasterize_labels(
        self, region: str, bounds: Tuple[float, float, float, float], out_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Растеризуем маску для всех классов по bbox окна.
        """
        H, W = out_shape
        mask = np.zeros((len(self.classes), H, W), dtype=np.uint8)

        gdf = self.region_labels.get(region)
        if gdf is None or len(gdf) == 0:
            return mask

        # отберём только геометрии, пересекающие окно
        xmin, ymin, xmax, ymax = bounds
        win_poly = box(xmin, ymin, xmax, ymax)
        try:
            sidx = self.region_sindex.get(region)
            cand_idx = list(sidx.intersection(win_poly.bounds)) if sidx is not None else None
            sub = gdf.iloc[cand_idx] if cand_idx else gdf
            sub = sub[sub.intersects(win_poly)]
        except Exception:
            sub = gdf[gdf.intersects(win_poly)]

        if sub.empty:
            return mask

        # affine окна: локальный Affine от левого верхнего угла
        # найдём transform региона
        transform = self.region_meta[region]["transform"]
        # вычислим кол/стр смещения (приблизительно)
        col_off = int(round((xmin - transform.c) / transform.a))
        row_off = int(round((ymin - transform.f) / transform.e))
        local_transform = Affine(
            transform.a, 0, transform.c + col_off * transform.a, 0, transform.e, transform.f + row_off * transform.e
        )

        pixel_size_m = float(self.region_meta[region]["pixel_size_m"])

        # группируем по классам
        # ожидаем, что имя класса можно получить из имени файла, или столбца 'name'/'label'
        # Если в gdf есть столбец 'label' или 'class' — используем его,
        # иначе применим эвристику: из filename метки вытащить часть после последнего "_"
        if "label" in sub.columns:
            cls_series = sub["label"].astype(str)
        elif "class" in sub.columns:
            cls_series = sub["class"].astype(str)
        elif "name" in sub.columns:
            cls_series = sub["name"].astype(str)
        else:
            # fallback — один класс "unknown" → скорее всего пусто
            cls_series = pd.Series(["unknown"] * len(sub), index=sub.index)

        for cls_name in self.classes:
            # отбираем полигоны текущего класса (по подстроке — учитывая формат «регион_Li_городища»)
            # ищем те строки, где в названии встречается cls_name
            sub_cls = sub[cls_series.str.contains(cls_name, case=False, na=False)]
            if sub_cls.empty:
                continue

            geoms = (
                sub_cls.geometry.buffer(self.class_buffers_m.get(cls_name, 0.0))
                if cls_name in self.class_buffers_m
                else sub_cls.geometry
            )

            try:
                arr = rasterize(
                    [(geom, 1) for geom in geoms if geom.is_valid],
                    out_shape=(H, W),
                    transform=local_transform,
                    fill=0,
                    all_touched=True,
                    dtype=np.uint8,
                )
            except Exception:
                # на всякий случай — без буфера
                arr = rasterize(
                    [(geom, 1) for geom in sub_cls.geometry if geom.is_valid],
                    out_shape=(H, W),
                    transform=local_transform,
                    fill=0,
                    all_touched=True,
                    dtype=np.uint8,
                )

            arr = self._apply_boundary_mode(cls_name, arr, pixel_size_m)
            mask[self.cls_to_idx[cls_name]] = arr.astype(np.uint8)

        return mask

    def __getitem__(self, idx: int):
        meta = self.index[idx]
        region = meta["region"]
        raster_paths = meta["raster_paths"]
        window = meta["window"]  # (col_off, row_off, w, h)

        img = self._read_multi_channel_window(raster_paths, window)  # [C,H,W]
        mask = self._rasterize_labels(region, meta["bounds"], img.shape[1:])  # [C,H,W]

        # albumentations ждёт HWC, а у нас CHW
        img_hwc = np.transpose(img, (1, 2, 0))
        # маска — многоканальная, но albumentations работает с одной; обойдёмся без геометрии
        # Поскольку мы уже делали геометрические аугментации на img_hwc, применим их синхронно с mask через additional_targets
        aug = A.Compose(
            self.aug.transforms if hasattr(self.aug, "transforms") else [],
            additional_targets={f"mask{i}": "mask" for i in range(mask.shape[0])},
        )
        data = {"image": img_hwc}
        for i in range(mask.shape[0]):
            data[f"mask{i}"] = mask[i]
        out = aug(**data)
        img_hwc = out["image"]
        for i in range(mask.shape[0]):
            mask[i] = out[f"mask{i}"]

        # в тензоры
        img_t = torch.from_numpy(np.transpose(img_hwc, (2, 0, 1))).float()  # [C,H,W]
        mask_t = torch.from_numpy(mask.astype(np.float32))  # [C,H,W]

        return img_t, mask_t, meta
