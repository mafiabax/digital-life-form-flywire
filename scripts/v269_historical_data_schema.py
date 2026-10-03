#!/usr/bin/env python3
import json
import os
import pickle
import subprocess
from pathlib import Path
import pandas as pd

ROOT = "https://raw.githubusercontent.com/borstlab/T4_T5_Dendrite_Morphology_Paper/56901ad1853b44aeca15504cd908fa4c31009a3e"
OUT = Path("v269_results")
OUT.mkdir(exist_ok=True)

FILES = {
    "Point_data.pkl": "Point_data",
    "Vertex_data.pkl": "Vertex_data",
    "Edge_data.pkl": "Edge_data",
    "Neuron_ids.csv": "Neuron_ids",
}

def run(cmd):
    subprocess.run(cmd, check=True)

def describe(name, obj):
    d = {"name": name, "type": type(obj).__name__}
    if isinstance(obj, pd.DataFrame):
        d["shape"] = list(obj.shape)
        d["columns"] = [str(x) for x in obj.columns]
        d["index_name"] = str(obj.index.name)
        d["dtypes"] = {str(k): str(v) for k, v in obj.dtypes.items()}
        d["head_records"] = obj.head(3).astype(str).to_dict(orient="records")
        d["index_head"] = [str(x) for x in obj.index[:5]]
    elif isinstance(obj, (list, tuple)):
        d["length"] = len(obj)
        d["item_types"] = [type(x).__name__ for x in obj[:5]]
    elif isinstance(obj, dict):
        d["keys_head"] = [str(k) for k in list(obj)[:30]]
        d["value_types"] = {str(k): type(obj[k]).__name__ for k in list(obj)[:10]}
    else:
        d["repr_head"] = repr(obj)[:1000]
    return d

# Download exact historical Git objects via raw URLs.
for fn, short in FILES.items():
    dest = Path("/tmp") / fn
    if not dest.exists():
        run(["curl", "-L", "--fail", "--retry", "3", "-o", str(dest), f"{ROOT}/Data/{fn}"])
    if fn.endswith(".pkl"):
        with open(dest, "rb") as f:
            obj = pickle.load(f)
    else:
        obj = pd.read_csv(dest)
    (OUT / f"{short}_schema.json").write_text(json.dumps(describe(fn, obj), indent=2, default=str))

point = pickle.load(open("/tmp/Point_data.pkl", "rb"))
vertex = pickle.load(open("/tmp/Vertex_data.pkl", "rb"))
edge = pickle.load(open("/tmp/Edge_data.pkl", "rb"))
ids = pd.read_csv("/tmp/Neuron_ids.csv")

summary = {
    "historical_commit": "56901ad1853b44aeca15504cd908fa4c31009a3e",
    "point": describe("Point_data.pkl", point),
    "vertex": describe("Vertex_data.pkl", vertex),
    "edge": describe("Edge_data.pkl", edge),
    "neuron_ids": describe("Neuron_ids.csv", ids),
}

# Cross-link obvious neuron-ID columns.
def idcol(df):
    if not isinstance(df, pd.DataFrame):
        return None
    candidates = [c for c in df.columns if str(c).lower() in {"id","flywire_id","flywireid","root_id"} or "id" == str(c).lower()]
    return candidates[0] if candidates else None

pc = idcol(point); vc = idcol(vertex)
summary["id_columns"] = {"point": pc, "vertex": vc}

# Emit coordinate-like columns and per-ID grouping hints.
for nm, df in [("point", point), ("vertex", vertex), ("edge", edge)]:
    if isinstance(df, pd.DataFrame):
        cols = [str(c) for c in df.columns]
        summary[f"{nm}_coordinate_columns"] = [c for c in cols if any(tok in c.lower() for tok in ("root","coord","x","y","z"))]
        summary[f"{nm}_id_columns"] = [c for c in cols if "id" in c.lower()]

# Try to match Point_data root coordinates to any Vertex_data coordinates by ID,
# without assuming a particular vertex-row convention.
if isinstance(point, pd.DataFrame) and isinstance(vertex, pd.DataFrame):
    p = point.reset_index()
    v = vertex.reset_index()
    p_cols = [c for c in p.columns if str(c).lower() in ("id","flywire_id","root_id") or "flywire" in str(c).lower()]
    v_cols = [c for c in v.columns if str(c).lower() in ("id","flywire_id","root_id") or "flywire" in str(c).lower()]
    summary["join_candidate_columns"] = {"point": [str(c) for c in p_cols], "vertex": [str(c) for c in v_cols]}

(OUT / "V269_historical_data_schema.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
