from pathlib import Path
from typing import List, Dict
import json
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, shapes
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, box
from .class_mapping import normalize_class
from pyproj.exceptions import ProjError


def load_labels_gdf(geojson_paths, target_crs):
    import json, pandas as pd
    from shapely.geometry import shape

    frames = []
    for p in geojson_paths:
        p = Path(p)
        gdf = None
        # pyogrio
        try:
            gdf = gpd.read_file(p)
        except Exception:
            gdf = None
        # fiona
        if gdf is None or gdf.empty:
            try:
                gdf = gpd.read_file(p, engine="fiona")
            except Exception:
                gdf = None
        # ручной json
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
        if gdf is None or gdf.empty:
            print(f"[labels][EMPTY] {p.name}")
            continue

        # class_name из столбца / properties.name / имени файла
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

        frames.append(gdf)

    if not frames:
        return gpd.GeoDataFrame(columns=["class_name", "geometry"], geometry="geometry", crs=target_crs)

    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    # Приводим к target_crs, если он задан и преобразование возможно
    if target_crs is not None:
        try:
            if out.crs is None:
                out = out.set_crs(target_crs)
            elif out.crs != target_crs:
                out = out.to_crs(target_crs)
        except Exception as e:
            print(f"[labels][CRS-WARN] to_crs failed: {e} — оставляем исходный CRS")
    return out


def rasterize_window(
    gdf: gpd.GeoDataFrame,
    classes: List[str],
    window_bounds,
    out_shape,
    transform,
    class_buffers_m: Dict[str, float] | None = None,
) -> np.ndarray:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    mask = np.zeros((len(classes), out_shape[0], out_shape[1]), dtype=np.uint8)
    if gdf.empty:
        return mask

    # режем по bbox окна
    bbox_poly = box(*window_bounds)
    g = gdf[gdf.geometry.intersects(bbox_poly)].copy()
    if g.empty:
        return mask
    g["geometry"] = g.geometry.intersection(bbox_poly)

    # буферизация для тонких линейных классов
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
            ((geom, 1) for geom in layer.geometry), out_shape=out_shape, transform=transform, fill=0, all_touched=True
        ).astype(np.uint8)
        mask[class_to_idx[cname]] |= rast
    return mask


def polygons_from_mask(mask_ch: np.ndarray, transform, simplify_tol: float = 0.0):
    """
    mask_ch: [H,W] uint8
    Возвращает список (geom: shapely, area_px)
    """
    if mask_ch.max() == 0:
        return []
    geoms = []
    for geom, val in shapes(mask_ch, transform=transform):
        if val == 0:
            continue
        shp = shape(geom)
        if simplify_tol > 0:
            shp = shp.simplify(simplify_tol, preserve_topology=True)
        geoms.append(shp)
    return geoms
