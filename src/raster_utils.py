from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window


def read_hillshade(path: str | Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        img = src.read(1).astype(np.float32)
        meta = src.meta.copy()
    # robust normalization 2..98 перцентили
    lo, hi = np.percentile(img, (2, 98))
    img = np.clip((img - lo) / max(1e-6, (hi - lo)), 0, 1)
    return img, meta


def sliding_windows(h: int, w: int, tile: int, stride: int):
    for y in range(0, max(1, h - tile + 1), stride):
        for x in range(0, max(1, w - tile + 1), stride):
            yield y, x, tile, tile


def window_transform(src_transform, y, x):
    # аффинное преобразование для окна с offset (x,y)
    return src_transform * rasterio.Affine.translation(x, y)
