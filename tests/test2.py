import geopandas as gpd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.features import rasterize
import numpy as np
from pathlib import Path

tif = Path("data/train/002_ДЕМИДОВКА_FINAL/02_Демидовка_Li_карты/02_Демидовка_Lidar_сh.tif")
gj = Path("data/train/002_ДЕМИДОВКА_FINAL/06_Демидовка_разметка/Li/Демидовка_Li_городища.geojson")

gdf = gpd.read_file(gj)  # EPSG:3857
print("labels feats:", len(gdf), "crs:", gdf.crs)

with rasterio.open(tif) as src:
    # Перепроецируем РАСТР в CRS меток (EPSG:3857)
    with WarpedVRT(src, dst_crs=gdf.crs, resampling=Resampling.bilinear) as vrt:
        # ограничимся окном вокруг меток — так быстрее и меньше памяти
        xmin, ymin, xmax, ymax = gdf.total_bounds
        win = vrt.window(xmin, ymin, xmax, ymax).round_offsets().round_lengths()
        H, W = int(win.height), int(win.width)
        if H <= 0 or W <= 0:
            print("empty window")
            raise SystemExit(0)
        transform = vrt.window_transform(win)

        mask = rasterize(
            ((geom, 1) for geom in gdf.geometry),
            out_shape=(H, W),
            transform=transform,
            fill=0,
            all_touched=True,
        )
        print("mask sum:", int(mask.sum()))
