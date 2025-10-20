# src/dataset_sliding.py
from __future__ import annotations

from pathlib import Path
import json
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
import cv2
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import Window
from shapely.geometry import box

from .raster_utils import sliding_windows
from .geojson_utils import load_labels_gdf, rasterize_window


# ---------------------------
# UTM зоны по номеру региона
# ---------------------------
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
    """
    Извлекаем номер региона (первые 3 цифры до '_').
    Если формат отличается — берём первый токен до '_'.
    """
    base = Path(region_name).name
    tok = base.split("_")[0]
    return tok if tok.isdigit() and len(tok) == 3 else tok


def _crs_from_zone(zone_str: str) -> CRS:
    """'36N' -> EPSG:32636; '36S' -> EPSG:32736"""
    zone = int(zone_str[:-1])
    hemi = zone_str[-1].upper()
    return CRS.from_epsg((32600 if hemi == "N" else 32700) + zone)


def _read_vrt_window(vrt, window: Window, fill_value: float = 0.0) -> np.ndarray:
    """
    Читаем окно из VRT корректно у краёв: обрезаем по границам и паддим до нужного размера.
    Без boundless (WarpedVRT его не поддерживает).
    """
    Ht, Wt = int(window.height), int(window.width)
    out = np.full((Ht, Wt), fill_value, dtype=np.float32)

    full = Window(0, 0, vrt.width, vrt.height)
    try:
        inter = window.intersection(full)
    except Exception:
        return out
    if inter.width <= 0 or inter.height <= 0:
        return out

    arr = vrt.read(1, window=inter, resampling=Resampling.bilinear).astype(np.float32)

    dy = int(round(inter.row_off - window.row_off))
    dx = int(round(inter.col_off - window.col_off))
    out[dy : dy + arr.shape[0], dx : dx + arr.shape[1]] = arr
    return out


class SlidingGeoDataset(Dataset):
    """
    Датасет:
      • собирает окна по сетке VRT (растр перепроецирован в CRS меток);
      • читает все доступные hillshade-каналы через WarpedVRT(src_crs=UTM(zone), dst_crs=labels_crs);
      • растеризует GeoJSON в границах окна;
      • применяет аугментации.
    """

    def __init__(
        self,
        index_df,
        classes: List[str],
        tile_size: int,
        tile_stride: int,
        augment: bool,
        class_buffers_m: Dict[str, float] | None = None,
        tta: bool = False,
    ):
        self.df = index_df.reset_index(drop=True).copy()
        self.classes = list(classes)
        self.tile = int(tile_size)
        self.stride = int(tile_stride)
        self.augment = bool(augment)
        self.class_buffers_m = class_buffers_m or {}
        self.tta = tta

        # decode json arrays if needed
        def _decode_col(s):
            if isinstance(s, str):
                try:
                    return json.loads(s)
                except Exception:
                    return []
            return s if s is not None else []

        self.df["raster_paths"] = self.df["raster_paths"].apply(_decode_col)
        self.df["labels_paths"] = self.df["labels_paths"].apply(_decode_col)

        # Albumentations (работает с N-каналами)
        aug = [A.Resize(self.tile, self.tile)]
        if self.augment:
            aug += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=25, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ]
        self.augs = A.Compose(aug)

        # предрасчёт по регионам
        self._labels_crs: Dict[int, CRS] = {}
        self._src_crs: Dict[int, CRS] = {}
        self._gdfs: Dict[int, gpd.GeoDataFrame] = {}
        self.items: List[Tuple[int, int, int, int, int]] = []  # (ridx, y, x, h, w)

        for ridx, row in self.df.iterrows():
            region_name = row["region_name"]
            raster_paths = row["raster_paths"]
            labels_paths = row["labels_paths"]

            # 1) CRS меток (берём из любого geojson)
            labels_crs = None
            for lp in labels_paths:
                try:
                    g0 = gpd.read_file(lp, engine="fiona")
                    if g0 is not None and g0.crs is not None:
                        labels_crs = g0.crs
                        break
                except Exception:
                    pass
            if labels_crs is None:
                raise RuntimeError(f"[{region_name}] Не удалось определить CRS меток (labels_crs).")

            # 2) Загружаем разметку и приводим к labels_crs
            gdf = load_labels_gdf(labels_paths, target_crs=labels_crs)
            self._labels_crs[ridx] = labels_crs
            self._gdfs[ridx] = gdf

            # 3) Определяем src_crs по таблице зон (номер региона → зона UTM)
            ref_path = next((p for p in raster_paths if p and Path(p).exists()), None)
            if ref_path is None:
                raise FileNotFoundError(f"[{region_name}] нет ни одного hillshade-файла.")

            prefix = _extract_prefix(region_name)
            if prefix in UTM_BY_PREFIX:
                src_crs = _crs_from_zone(UTM_BY_PREFIX[prefix])
                print(f"[CRS-MAP] {region_name}: src_crs ← UTM {UTM_BY_PREFIX[prefix]} ({src_crs.to_string()})")
            else:
                # fallback — считаем исходник тоже в labels_crs (редко потребуется)
                src_crs = labels_crs
                print(f"[CRS-MAP][WARN] {region_name}: нет ключа '{prefix}' в UTM_BY_PREFIX → fallback to {labels_crs}")

            self._src_crs[ridx] = src_crs

            # 4) Индексация окон по VRT (в CRS меток)
            with rasterio.open(ref_path) as ref, WarpedVRT(
                ref, src_crs=src_crs, dst_crs=labels_crs, resampling=Resampling.bilinear
            ) as vrt:
                H, W = vrt.height, vrt.width
            for y, x, h, w in sliding_windows(H, W, self.tile, self.stride):
                self.items.append((ridx, y, x, h, w))

        print(f"Prepared {len(self.items)} tiles from {len(self.df)} regions (skipped 0).")

    def __len__(self):
        return len(self.items)

    def _read_multi_channel_window(
        self, raster_paths: List[str], window: Window, labels_crs: CRS, src_crs: CRS
    ) -> np.ndarray:
        """Читает окно из каждого канала через WarpedVRT(src_crs, dst_crs=labels_crs), нормализует 2–98 перцентили, склеивает в [H,W,C]."""
        h, w = int(window.height), int(window.width)
        imgs = []
        for p in raster_paths:
            if p and Path(p).exists():
                with rasterio.open(p) as src, WarpedVRT(
                    src, src_crs=src_crs, dst_crs=labels_crs, resampling=Resampling.bilinear
                ) as vrt:
                    arr = _read_vrt_window(vrt, window, fill_value=0.0)
            else:
                arr = np.zeros((h, w), dtype=np.float32)

            if arr.size > 0:
                lo, hi = np.percentile(arr, (2, 98))
                if hi <= lo:
                    arr = np.zeros_like(arr, dtype=np.float32)
                else:
                    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
            imgs.append(arr)

        return np.stack(imgs, axis=-1)  # HWC

    def __getitem__(self, i):
        ridx, y, x, h, w = self.items[i]
        row = self.df.iloc[ridx]
        region_name = row["region_name"]
        raster_paths: List[str] = row["raster_paths"]
        labels_paths: List[str] = row["labels_paths"]

        labels_crs = self._labels_crs[ridx]
        src_crs = self._src_crs[ridx]
        gdf = self._gdfs[ridx]

        # референсный канал (любой существующий)
        ref_path = next((p for p in raster_paths if p and Path(p).exists()), None)
        if ref_path is None:
            raise FileNotFoundError(f"[{region_name}] no raster paths available")

        # работаем в CRS меток: bounds/transform берём из VRT (src_crs → dst=labels_crs)
        with rasterio.open(ref_path) as ref, WarpedVRT(
            ref, src_crs=src_crs, dst_crs=labels_crs, resampling=Resampling.bilinear
        ) as vrt:
            window = Window(x, y, w, h)
            transform_w = rasterio.windows.transform(window, vrt.transform)
            bounds = rasterio.windows.bounds(window, vrt.transform)

        # читаем мультиканал
        img = self._read_multi_channel_window(raster_paths, window, labels_crs, src_crs)  # HWC, float32[0..1]

        # растеризация маски в том же CRS/окне
        mask = rasterize_window(
            gdf=gdf,
            classes=self.classes,
            window_bounds=bounds,
            out_shape=(h, w),
            transform=transform_w,
            class_buffers_m=self.class_buffers_m,
        )  # [C,H,W] uint8

        # аугментации
        mask_hwc = np.transpose(mask, (1, 2, 0)).astype(np.uint8)
        out = self.augs(image=img, mask=mask_hwc)
        img_aug = out["image"]
        mask_aug = out["mask"]

        # в тензоры
        img_t = torch.from_numpy(np.transpose(img_aug, (2, 0, 1))).float()  # [C,H,W]
        mask_t = torch.from_numpy(np.transpose(mask_aug, (2, 0, 1))).float()  # [C,H,W]

        # modality dropout (иногда гасим случайные каналы)
        if self.augment and np.random.rand() < 0.25:
            c = img_t.shape[0]
            k = np.random.randint(1, c + 1)
            drop_idx = np.random.choice(c, size=k, replace=False)
            img_t[drop_idx] = 0.0

        meta = {
            "region_idx": ridx,
            "region_name": region_name,
            "y": int(y),
            "x": int(x),
            "h": int(h),
            "w": int(w),
            "tile_id": f"{Path(ref_path).stem}_{y}_{x}",
        }
        return img_t, mask_t, meta
