"""Merge the 4 scattered folders into ONE dataset: merged_dataset/images/*.png + master.csv
Usage: python build_dataset.py "Final Database" merged_dataset"""
import os, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "Final Database")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "merged_dataset")
(OUT / "images").mkdir(parents=True, exist_ok=True)
norm = lambda s: re.sub(r"[^\w]+", "_", os.path.splitext(str(s))[0]).strip("_").lower()
def save(src, id_):
    Image.open(src).convert("RGB").save(OUT / "images" / f"{id_}.png"); return f"images/{id_}.png"
T = ["ph", "turbidity", "cod", "bod"]
rows = []

# --- Original 100 (columns: Image Name,pH,Turbidity (NTU),COD (mg/L),BOD (mg/L))
d = pd.read_csv(SRC / "Original 100.csv"); d.columns = ["name"] + T
for r in d.itertuples():
    i = "o100_" + norm(r.name)
    rows.append(dict(id=i, file=save(SRC / "Original 100" / r.name, i), source_set="original_100", photographer="set100",
                     is_synthetic=0, parent_id="", ph=r.ph, turbidity=r.turbidity, cod=r.cod, bod=r.bod))

# --- Original 80 (columns: Image,PH,Turbidity,COD,BOD); CSV has no extension, so match by stem
d = pd.read_csv(SRC / "Original 80.csv"); d.columns = ["name"] + T
files = {norm(f): f for f in os.listdir(SRC / "Original 80") if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))}
for r in d.itertuples():
    i = "o80_" + norm(r.name)
    rows.append(dict(id=i, file=save(SRC / "Original 80" / files[norm(r.name)], i), source_set="original_80",
                     photographer=re.match(r"[A-Za-z]+", r.name).group(0).lower(), is_synthetic=0, parent_id="",
                     ph=r.ph, turbidity=r.turbidity, cod=r.cod, bod=r.bod))
orig = pd.DataFrame(rows).set_index("id")

# --- Synthetic sets: labels are copied from the parent original (they are NOT new measurements)
def add_syn(csv, folder, tag, name_col, parent_fn, parent_prefix):
    for r in pd.read_csv(SRC / csv).itertuples():
        fname = os.path.basename(getattr(r, name_col)); i = f"{tag}_{norm(fname)}"
        p = parent_prefix + norm(parent_fn(r)); par = orig.loc[p]
        rows.append(dict(id=i, file=save(SRC / folder / fname, i), source_set=f"synthetic_{tag[1:]}", photographer=par.photographer,
                         is_synthetic=1, parent_id=p, ph=par.ph, turbidity=par.turbidity, cod=par.cod, bod=par.bod))
add_syn("Synthetic 100.csv", "Synthetic 100", "s100", "generated_image", lambda r: r.source_image, "o100_")
add_syn("Synthetic 160.csv", "Synthetic 160", "s160", "synthetic_filename", lambda r: r.Image, "o80_")

m = pd.DataFrame(rows)
# group_id: identical label tuples = same physical sample (e.g. Sakshi_16 and Vedika(6)); children share the parent's group.
key = m[T].round(3).apply(lambda r: "|".join(map(str, r)), axis=1)
orig_key = key[m.is_synthetic == 0].set_axis(m.id[m.is_synthetic == 0])
gid = {k: n for n, k in enumerate(orig_key.unique())}
m["group_id"] = [gid[orig_key[r.id]] if not r.is_synthetic else gid[orig_key[r.parent_id]] for r in m.itertuples()]
m.to_csv(OUT / "master.csv", index=False)
print(m.groupby("source_set").size().to_string()); print("total", len(m), "| groups", m.group_id.nunique())
print("missing labels:\n", m[m.is_synthetic == 0][T].isna().sum().to_string())
