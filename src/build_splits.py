from pathlib import Path
import glob
import pandas as pd
import yaml
import json
from collections import Counter


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

        rows.append(
            {
                "region_dir": str(region_dir),
                "region_name": region_dir.name,
                "raster_paths": json.dumps(chan_paths, ensure_ascii=False),  # ← JSON!
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
