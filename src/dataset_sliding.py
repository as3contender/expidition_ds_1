# src/dataset_sliding.py
from __future__ import annotations

from pathlib import Path
import json
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
import cv2
import geopandas as gpd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import Window
from shapely.geometry import box

from .raster_utils import sliding_windows
from .geojson_utils import load_labels_gdf, rasterize_window


class SlidingGeoDataset(Dataset):
    """
    Датасет, который:
      • читает все доступные hillshade-каналы;
      • ПЕРЕПРОЕЦИРУЕТ РАСТРЫ в CRS разметки (через WarpedVRT) — это ключевой фикс против LOCAL_CS;
      • онлайн-тайлит и растеризует GeoJSON в окно.
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
        """
        index_df: DataFrame со столбцами:
          - region_dir
          - region_name
          - raster_paths : JSON-список путей по каналам (или list)
          - labels_paths : JSON-список путей к geojson
        """
        self.df = index_df.reset_index(drop=True).copy()
        self.classes = list(classes)
        self.tile = int(tile_size)
        self.stride = int(tile_stride)
        self.augment = bool(augment)
        self.class_buffers_m = class_buffers_m or {}
        self.tta = tta

        # --- decode json lists if needed
        def _decode_col(s):
            if isinstance(s, str):
                try:
                    return json.loads(s)
                except Exception:
                    return []
            return s if s is not None else []

        self.df["raster_paths"] = self.df["raster_paths"].apply(_decode_col)
        self.df["labels_paths"] = self.df["labels_paths"].apply(_decode_col)

        # --- augs (работают для N-канального входа)
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

        # --- предрасчёт: для каждого региона:
        #     • labels_crs (CRS разметки)
        #     • gdf разметки, приведённый к labels_crs
        #     • список окон (y, x, h, w) по размеру РЕФЕРЕНС-растра (в пикселях)
        self._labels_crs: Dict[int, any] = {}
        self._gdfs: Dict[int, gpd.GeoDataFrame] = {}
        self.items: List[tuple[int, int, int, int, int]] = []

        for ridx, row in self.df.iterrows():
            region_name = row["region_name"]
            raster_paths = row["raster_paths"]
            labels_paths = row["labels_paths"]

            # --- извлекаем CRS меток (labels_crs) из любого geojson файла региона
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

            self._labels_crs[ridx] = labels_crs

            # --- подгружаем gdf и приводим К labels_crs
            gdf = load_labels_gdf(labels_paths, target_crs=labels_crs)
            self._gdfs[ridx] = gdf

            # --- найдём референс-канал (первый существующий файл)
            ref_path = None
            for p in raster_paths:
                if p and Path(p).exists():
                    ref_path = p
                    break
            if ref_path is None:
                raise FileNotFoundError(f"[{region_name}] нет ни одного доступного hillshade-файла.")

            # --- индексация тайлов по размеру референс-растра (в пикселях исходного файла)
            with rasterio.open(ref_path) as ref:
                H, W = ref.height, ref.width
            for y, x, h, w in sliding_windows(H, W, self.tile, self.stride):
                self.items.append((ridx, y, x, h, w))

        # небольшой лог:
        n_tr = sum(1 for it in self.items if it[0] in set(self.df.index[: len(self.df)]))
        print(f"Prepared {len(self.items)} tiles from {len(self.df)} regions (skipped 0).")

    def __len__(self):
        return len(self.items)

    def _read_multi_channel_window(self, raster_paths: List[str], window: Window, labels_crs) -> np.ndarray:
        """
        Читает окно из каждого канала через WarpedVRT(dst_crs=labels_crs), нормализует 2–98 перцентили,
        возвращает стек [H, W, C].
        """
        h, w = int(window.height), int(window.width)
        imgs = []
        for p in raster_paths:
            if p and Path(p).exists():
                with rasterio.open(p) as src, WarpedVRT(src, dst_crs=labels_crs, resampling=Resampling.bilinear) as vrt:
                    arr = vrt.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
            else:
                arr = np.zeros((h, w), dtype=np.float32)

            if arr.size > 0:
                lo, hi = np.percentile(arr, (2, 98))
                if hi <= lo:
                    # деградантный канал — просто нулевой
                    arr = np.zeros_like(arr, dtype=np.float32)
                else:
                    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
            imgs.append(arr)

        img = np.stack(imgs, axis=-1)  # HWC
        return img

    def __getitem__(self, i):
        ridx, y, x, h, w = self.items[i]
        row = self.df.iloc[ridx]
        region_name = row["region_name"]
        raster_paths: List[str] = row["raster_paths"]
        labels_paths: List[str] = row["labels_paths"]
        labels_crs = self._labels_crs[ridx]
        gdf = self._gdfs[ridx]

        # выберем референс-канал (любой существующий)
        ref_path = None
        for p in raster_paths:
            if p and Path(p).exists():
                ref_path = p
                break
        if ref_path is None:
            # не должно случаться — отловлено при инициализации
            raise FileNotFoundError(f"[{region_name}] no raster paths available")

        # --- ключевой момент: работаем в CRS МЕТОК.
        #     Берем bounds и transform окна из WarpedVRT (dst_crs = labels_crs).
        with rasterio.open(ref_path) as ref, WarpedVRT(ref, dst_crs=labels_crs, resampling=Resampling.bilinear) as vrt:
            window = Window(x, y, w, h)
            # трансформ окна и его географические границы в labels_crs
            transform_w = rasterio.windows.transform(window, vrt.transform)
            bounds = rasterio.windows.bounds(window, vrt.transform)

        # --- читаем многоканальное окно (каждый канал через свой WarpedVRT в labels_crs)
        img = self._read_multi_channel_window(raster_paths, window, labels_crs)

        # --- растеризация маски в том же CRS/окне
        mask = rasterize_window(
            gdf=gdf,
            classes=self.classes,
            window_bounds=bounds,
            out_shape=(h, w),
            transform=transform_w,
            class_buffers_m=self.class_buffers_m,
        )  # [C, H, W] uint8

        # --- аугментации (Albumentations ожидает HWC)
        mask_hwc = np.transpose(mask, (1, 2, 0)).astype(np.uint8)
        out = self.augs(image=img, mask=mask_hwc)
        img_aug = out["image"]
        mask_aug = out["mask"]

        # --- в тензоры
        img_t = torch.from_numpy(np.transpose(img_aug, (2, 0, 1))).float()  # [C,H,W]
        mask_t = torch.from_numpy(np.transpose(mask_aug, (2, 0, 1))).float()  # [C,H,W]

        # --- modality dropout (повышает устойчивость к отсутствующим каналам)
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
