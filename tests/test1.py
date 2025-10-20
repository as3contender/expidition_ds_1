import geopandas as gpd
from pathlib import Path

p = Path("data/train/014_КАБЛУКОВО_FINAL/06_Каблуково_разметка/Li/Каблуково_Li_дороги.geojson")

for eng in (None, "fiona"):
    try:
        gdf = gpd.read_file(p) if eng is None else gpd.read_file(p, engine=eng)
        print(f"{p.name} | engine={eng or 'pyogrio'} | feats={len(gdf)} | crs={gdf.crs}")
    except Exception as e:
        print(f"{p.name} | engine={eng or 'pyogrio'} | error={e}")
