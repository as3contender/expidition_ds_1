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


# src/raster_utils.py
def sliding_windows(H: int, W: int, tile: int, stride: int):
    """
    Генерирует окна (y, x, h, w) так, чтобы:
      - стартовые смещения не выходили за границы,
      - последний тайл "прижимался" к правому/нижнему краю,
      - если картинка меньше тайла, всё равно вернуть одно окно (0,0,tile,tile).
    """
    tile = int(tile)
    stride = int(stride)
    H = int(H)
    W = int(W)

    if H <= 0 or W <= 0:
        return  # пустой генератор

    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))

    end_y = max(0, H - tile)
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


def window_transform(src_transform, y, x):
    # аффинное преобразование для окна с offset (x,y)
    return src_transform * rasterio.Affine.translation(x, y)
