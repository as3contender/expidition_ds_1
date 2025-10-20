from pathlib import Path
from typing import List, Dict
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.geometry import box
from rasterio.features import rasterize
import numpy as np

from .class_mapping import normalize_class


def _read_one_geojson(p: Path) -> gpd.GeoDataFrame | None:
    gdf = None
    # 1) pyogrio (по умолчанию у geopandas)
    try:
        gdf = gpd.read_file(p)
    except Exception:
        gdf = None
    # 2) fiona fallback
    if gdf is None or gdf.empty:
        try:
            gdf = gpd.read_file(p, engine="fiona")
        except Exception:
            gdf = None
    # 3) ручной JSON
    if gdf is None or gdf.empty:
        try:
            data = json.loads(p.read_text(encoding="utf-8-sig"))
            feats = data.get("features", [])
            if feats:
                recs, geoms = [], []
                for f in feats:
                    geom = f.get("geometry")
                    if geom:
                        geoms.append(shape(geom))
                        recs.append(f.get("properties", {}))
                if geoms:
                    gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs=None)
        except Exception:
            gdf = None
    return gdf


def load_labels_gdf(geojson_paths: List[str | Path], target_crs) -> gpd.GeoDataFrame:
    """
    Читает список GeoJSON и ПРИВОДИТ К ЕДИНОМУ target_crs ДО конкатенации.
    target_crs — CRS, в котором будем работать (у нас это CRS меток, далее совпадает с WarpedVRT).
    """
    frames: list[gpd.GeoDataFrame] = []

    for p in geojson_paths:
        p = Path(p)
        gdf = _read_one_geojson(p)
        if gdf is None or gdf.empty:
            print(f"[labels][EMPTY] {p.name}")
            continue

        # --- нормализуем class_name: сначала из явного столбца, затем из properties.name, затем из имени файла
        cname = None
        if "class_name" in gdf.columns and pd.notna(gdf["class_name"].iloc[0]):
            cname = normalize_class(gdf["class_name"].iloc[0])
        if not cname and "name" in gdf.columns and pd.notna(gdf["name"].iloc[0]):
            raw = str(gdf["name"].iloc[0]).split("_")[-1]
            cname = normalize_class(raw)
        if not cname:
            raw = p.stem.split("_")[-1]
            cname = normalize_class(raw)
        gdf["class_name"] = cname

        # --- приводим К target_crs ДО того, как положим в список
        if target_crs is not None:
            try:
                if gdf.crs is None:
                    # если у файла нет CRS — назначим target_crs (лучше так, чем смешивать)
                    gdf = gdf.set_crs(target_crs)
                elif gdf.crs != target_crs:
                    gdf = gdf.to_crs(target_crs)
            except Exception as e:
                print(f"[labels][CRS-WARN] {p.name}: to_crs failed: {e} — пропускаю файл")
                continue  # этот файл лучше пропустить, чтобы не ломать весь регион

        frames.append(gdf)

    if not frames:
        # вернуть валидный пустой GDF с target_crs, чтобы дальше не падать
        return gpd.GeoDataFrame(columns=["class_name", "geometry"], geometry="geometry", crs=target_crs)

    # все фреймы уже в ОДНОМ CRS → конкатенация безопасна
    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=target_crs)
    return out


def rasterize_window(
    gdf: gpd.GeoDataFrame,
    classes: List[str],
    window_bounds,
    out_shape,
    transform,
    class_buffers_m: Dict[str, float] | None = None,
) -> np.ndarray:
    """Растеризация по окну в CRS окна (совпадает с target_crs/labels_crs)"""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    mask = np.zeros((len(classes), out_shape[0], out_shape[1]), dtype=np.uint8)
    if gdf is None or gdf.empty:
        return mask

    # обрезаем по bbox окна (в том же CRS)
    bbox_poly = box(*window_bounds)
    g = gdf[gdf.geometry.intersects(bbox_poly)].copy()
    if g.empty:
        return mask
    g["geometry"] = g.geometry.intersection(bbox_poly)

    # буферизация тонких классов
    if class_buffers_m:

        def _buf(row):
            c = row["class_name"]
            if c in class_buffers_m:
                return row.geometry.buffer(class_buffers_m[c])
            return row.geometry

        g["geometry"] = g.apply(_buf, axis=1)

    for cname in g["class_name"].unique():
        if cname not in class_to_idx:
            continue
        layer = g[g["class_name"] == cname]
        rast = rasterize(
            ((geom, 1) for geom in layer.geometry if geom and not geom.is_empty),
            out_shape=out_shape,
            transform=transform,
            fill=0,
            all_touched=True,
        ).astype(np.uint8)
        mask[class_to_idx[cname]] |= rast
    return mask
