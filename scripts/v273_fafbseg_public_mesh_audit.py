#!/usr/bin/env python3
import json, math, os, pathlib, pickle, subprocess
import numpy as np, pandas as pd
import navis
from fafbseg import flywire as fwy

ROOT="https://raw.githubusercontent.com/borstlab/T4_T5_Dendrite_Morphology_Paper/56901ad1853b44aeca15504cd908fa4c31009a3e"
OUT=pathlib.Path("v273_results"); OUT.mkdir(exist_ok=True)
subprocess.run(["curl","-L","--fail","--retry","3","-o","/tmp/Point_data.pkl",f"{ROOT}/Data/Point_data.pkl"],check=True)
with open("/tmp/Point_data.pkl","rb") as f: point=pickle.load(f)
targets=[720575940632008007,720575940616224414,720575940625571465,720575940617782941]
sel={targets[0]:292,targets[1]:358,targets[2]:343,targets[3]:323}
rows=[]
for rid in targets:
    p=point.loc[point.ID==rid].iloc[0]
    m=fwy.get_mesh_neuron(rid,omit_failures=False,threads=None,progress=False,dataset="flat_783")
    verts=np.asarray(m.vertices,dtype=float)
    # Skeletonize exactly as paper pipeline.
    s=m.skeletonize()
    swc=s.nodes[["node_id","parent_id","x","y","z"]].copy()
    coord=swc.loc[swc.node_id.astype(int)==sel[rid]]
    # Find nearest skeleton node to the published point-data root after no transform.
    q=np.array([p.Root_x,p.Root_y,p.Root_z])*1000.0
    d=np.linalg.norm(swc[["x","y","z"]].to_numpy()-q[None,:],axis=1)
    j=int(np.argmin(d)); near=swc.iloc[j]
    rows.append(dict(ID=rid,Subtype=p.Subtype,point_um=[float(p.Root_x),float(p.Root_y),float(p.Root_z)],
        mesh_vertices=len(verts),mesh_centroid_um=(verts.mean(0)/1000).tolist(),mesh_min_um=(verts.min(0)/1000).tolist(),mesh_max_um=(verts.max(0)/1000).tolist(),
        skeleton_nodes=len(swc),paper_selected_node=sel[rid],
        selected_coord_um=(coord[["x","y","z"]].iloc[0].to_numpy()/1000).tolist(),
        nearest_to_point_node=int(near.node_id),nearest_distance_um=float(d[j]/1000),
        nearest_coord_um=(near[["x","y","z"]].to_numpy()/1000).tolist()))
pd.DataFrame(rows).to_json(OUT/"V273_fafbseg_public_mesh_audit.jsonl",orient="records",lines=True)
(OUT/"V273_fafbseg_public_mesh_audit.json").write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
