import yaml, json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import rasterio
from tqdm import tqdm
from shapely.geometry import mapping
from .dataset_sliding import SlidingGeoDataset
from .model_unet import build_unet
from .geojson_utils import polygons_from_mask


def main(cfg_path="configs/default.yaml", ckpt_path=None, out_geojson="runs/default/submission.geojson"):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    classes = list(cfg["data"]["classes"].keys())
    out_dir = Path(cfg["log"]["out_dir"])

    if ckpt_path is None:
        ckpt_path = out_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    num_ch = len(cfg["data"].get("hillshade_channels", [None]))
    model = build_unet(
        cfg["model"]["encoder"], cfg["model"]["encoder_weights"], in_channels=num_ch, classes=len(classes)
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    val_regions = pd.read_csv(out_dir / "val_regions.csv")  # или test_regions.csv, если сделаешь
    tile, stride = cfg["data"]["tile_size"], cfg["data"]["tile_stride"]
    class_buffers = cfg["data"].get("class_buffer_m", {})
    ds = SlidingGeoDataset(val_regions, classes, tile, stride, augment=False, class_buffers_m=class_buffers)

    features = []
    with torch.no_grad():
        i = 0
        for ridx, row in val_regions.iterrows():
            raster_path = row.raster_path
            region_name = row.region_name
            paths = row.raster_paths if isinstance(row.raster_paths, list) else eval(row.raster_paths)
            with rasterio.open(paths[0] or paths[1] or paths[2] or paths[3]) as src_ref:
                H, W = src_ref.height, src_ref.width
                full_probs = np.zeros((len(classes), H, W), dtype=np.float32)
                counts = np.zeros((H, W), dtype=np.float32)

                # прогоняем слайдинг для конкретной территории
                for ii, (ridx2, y, x, h, w) in enumerate([it for it in ds.items if it[0] == ridx]):
                    patch = []
                    for p in paths:
                        if p and Path(p).exists():
                            with rasterio.open(p) as src:
                                im = src.read(1, window=rasterio.windows.Window(x, y, w, h)).astype(np.float32)
                                lo, hi = np.percentile(im, (2, 98))
                                im = np.clip((im - lo) / max(1e-6, hi - lo), 0, 1)
                        else:
                            im = np.zeros((h, w), dtype=np.float32)
                        patch.append(im)
                    im = np.stack(patch, axis=0)  # C,h,w
                    t = torch.from_numpy(im[None, ...]).float().to(device)
                    p = torch.sigmoid(model(t))[0].cpu().numpy()
                    full_probs[:, y : y + h, x : x + w] += p
                    counts[y : y + h, x : x + w] += 1.0

                full_probs /= np.maximum(counts, 1e-6)

                # порогование и полигонализация по классам
                for k, cname in enumerate(classes):
                    thr = cfg["thresholds"].get(cname, 0.35)
                    binmask = (full_probs[k] >= thr).astype(np.uint8)
                    geoms = polygons_from_mask(binmask, src.transform, simplify_tol=0.0)
                    for g in geoms:
                        if g.is_empty:
                            continue
                        features.append(
                            {
                                "type": "Feature",
                                "properties": {
                                    "region_name": region_name,
                                    "class_name": cname,
                                },
                                "geometry": mapping(g),
                            }
                        )

    submission = {"type": "FeatureCollection", "features": features}
    Path(out_geojson).parent.mkdir(parents=True, exist_ok=True)
    json.dump(submission, open(out_geojson, "w", encoding="utf-8"), ensure_ascii=False)
    print("Saved submission:", out_geojson)


if __name__ == "__main__":
    main()
