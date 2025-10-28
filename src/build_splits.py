from pathlib import Path
import glob
import pandas as pd
import yaml
import json
from collections import Counter
from typing import List


def build_splits(
    regions_root: str | Path,
    hillshade_globs: List[str],
    labels_glob: str,
    limit_regions: int | None = None,
    limit_val_regions: int | None = None,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Строит train/val сплиты из регионов.

    Args:
        regions_root: путь к папке с регионами (*_FINAL)
        hillshade_globs: список паттернов для поиска hillshade каналов
        labels_glob: паттерн для поиска GeoJSON разметки
        limit_regions: ограничить количество регионов для train (None = все)
        limit_val_regions: ограничить количество регионов для val (None = по train_ratio)
        train_ratio: доля регионов для train (если limit_val_regions не задан)
        seed: random seed

    Returns:
        (df_train, df_val): DataFrames с колонками region_name, raster_paths, labels_paths
    """
    root = Path(regions_root)
    rows = []

    for region_dir in sorted(root.glob("*_FINAL")):
        chan_paths = []
        for pat in hillshade_globs:
            matches = list(region_dir.glob(pat))
            chan_paths.append(str(matches[0]) if matches else None)

        # Skip regions without any hillshade channels
        if not any(chan_paths):
            continue

        labels = list(region_dir.glob(labels_glob))
        if len(labels) == 0:
            print(f"[build_splits] SKIP region={region_dir.name}: no labels")
            continue

        # Сохраняем только существующие пути (без None)
        raster_store = [p for p in chan_paths if p]
        rows.append(
            {
                "region_name": region_dir.name,
                "raster_paths": json.dumps(raster_store, ensure_ascii=False),
                "labels_paths": json.dumps([str(p) for p in labels], ensure_ascii=False),
            }
        )

    df = pd.DataFrame(rows)

    # Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Apply limits
    if limit_regions is not None:
        n_tr = min(limit_regions, len(df))
    else:
        n_tr = max(1, int(len(df) * train_ratio))

    df_tr = df.iloc[:n_tr].reset_index(drop=True)
    df_remaining = df.iloc[n_tr:].reset_index(drop=True)

    if limit_val_regions is not None:
        df_val = df_remaining.iloc[:limit_val_regions].reset_index(drop=True)
    else:
        df_val = df_remaining

    print(f"[build_splits] train={len(df_tr)} regions, val={len(df_val)} regions")
    return df_tr, df_val


def build_index(cfg_path="configs/default.yaml", out_csv="runs/default/index.csv", train_ratio=0.8, seed=42):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    root = Path(cfg["data"]["regions_root"])
    labels_glob = cfg["data"]["labels_subpath"]

    rows = []
    total_regions = 0
    used_regions = 0
    skipped_regions = 0
    coverage = Counter()
    for region_dir in sorted(root.glob("*_FINAL")):
        total_regions += 1
        hs_patterns = cfg["data"]["hillshade_channels"]
        chan_paths = []
        for pat in hs_patterns:
            matches = list(region_dir.glob(pat))
            chan_paths.append(str(matches[0]) if matches else None)

        # Skip regions without any hillshade channels present
        if not any(chan_paths):
            skipped_regions += 1
            continue

        for i, p in enumerate(chan_paths):
            if p:
                coverage[i] += 1

        labels = list(region_dir.glob(labels_glob))

        if len(labels) == 0:
            print(f"[build] SKIP region={region_dir.name}: no labels for pattern `{labels_glob}`")
            continue

        # Сохраняем только существующие пути (без None)
        raster_store = [p for p in chan_paths if p]
        rows.append(
            {
                "region_dir": str(region_dir),
                "region_name": region_dir.name,
                "raster_paths": json.dumps(raster_store, ensure_ascii=False),  # ← JSON!
                "labels_paths": json.dumps([str(p) for p in labels], ensure_ascii=False),
            }
        )
        used_regions += 1

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    # Сплит по территориям
    df = df.sample(frac=1.0, random_state=seed)
    n_tr = max(1, int(len(df) * train_ratio))
    df.iloc[:n_tr].to_csv("runs/default/train_regions.csv", index=False)
    df.iloc[n_tr:].to_csv("runs/default/val_regions.csv", index=False)
    print(f"Saved index to {out_csv}")
    print(f"Regions: total={total_regions}, used={used_regions}, skipped={skipped_regions}")
    if coverage:
        print("Channel coverage per index (0-based):", dict(coverage))


if __name__ == "__main__":
    build_index()
