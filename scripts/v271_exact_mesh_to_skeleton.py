#!/usr/bin/env python3
import json, pickle, pathlib, math
import numpy as np, pandas as pd
from fafbseg import flywire as fwy
import navis

IDS=[720575940632008007,720575940616224414,720575940625571465,720575940617782941]
OUT=pathlib.Path("v271_results"); OUT.mkdir(exist_ok=True)
with open("/tmp/Point_data.pkl","wb") as f:
    import urllib.request
    f.write(urllib.request.urlopen("https://raw.githubusercontent.com/borstlab/T4_T5_Dendrite_Morphology_Paper/56901ad1853b44aeca15504cd908fa4c31009a3e/Data/Point_data.pkl").read())
with open("/tmp/Point_data.pkl","rb") as f: point=pickle.load(f)

def select_pp3_root(nodes):
    """Reproduce the published NeuRosetta PP3 subtree score on a rooted SWC table."""
    cols = ["node_id", "parent_id", "x", "y", "z"]
    df = nodes[cols].copy()
    by = {int(r.node_id): r for r in df.itertuples(index=False)}
    children = {int(i): [] for i in df.node_id}
    roots = []
    for r in df.itertuples(index=False):
        n, p = int(r.node_id), int(r.parent_id)
        if p < 0:
            roots.append(n)
        else:
            children[p].append(n)
    if len(roots) != 1:
        raise RuntimeError(f"Expected one rooted tree, got roots={roots}")
    root = roots[0]
    edge_len = {}
    total_cable = 0.0
    for r in df.itertuples(index=False):
        n, p = int(r.node_id), int(r.parent_id)
        if p >= 0:
            d = math.dist((r.x, r.y, r.z), (by[p].x, by[p].y, by[p].z))
            edge_len[(p, n)] = d
            total_cable += d
    leaf_set = {n for n, ch in children.items() if not ch}
    order, stack = [], [root]
    while stack:
        n = stack.pop()
        order.append(n)
        stack.extend(reversed(children[n]))
    subtree_cable = {n: 0.0 for n in children}
    subtree_leaves = {n: (1.0 if n in leaf_set else 0.0) for n in children}
    for n in reversed(order):
        for ch in children[n]:
            subtree_cable[n] += edge_len[(n, ch)] + subtree_cable[ch]
            subtree_leaves[n] += subtree_leaves[ch]
    candidates = []
    for n in children:
        if len(children[n]) >= 2:
            score = (1.0 - subtree_cable[n] / total_cable) + (subtree_leaves[n] / len(leaf_set))
            candidates.append((score, n, subtree_cable[n], int(subtree_leaves[n])))
    if not candidates:
        raise RuntimeError("No branch nodes available for PP3 selection")
    score, node, cable, leaves = max(candidates)
    members = {node}
    stack = [node]
    while stack:
        n = stack.pop()
        for ch in children[n]:
            if ch not in members:
                members.add(ch)
                stack.append(ch)
    return {
        "node": int(node),
        "score": float(score),
        "subtree_cable_nm": float(cable),
        "subtree_leaves": int(leaves),
        "subtree_nodes": int(len(members)),
        "whole_root": int(root),
    }

rows=[]
for rid in IDS:
    p=point[point.ID==rid].iloc[0]
    m=fwy.get_mesh_neuron(rid, omit_failures=False, progress=True, dataset='flat_783')
    s=m.skeletonize()
    if getattr(s,'units',None) is None: s.units='1 nm'
    navis.resample_skeleton(s, resample_to=100, inplace=True)
    pp3=select_pp3_root(s.nodes)
    # Count branch points inside the selected subtree; the NeuRosetta reduction keeps
    # the selected root, all branch points below it, and all leaves.
    parent_map={int(r.node_id): int(r.parent_id) for r in s.nodes.itertuples(index=False)}
    child_count={int(r.node_id): 0 for r in s.nodes.itertuples(index=False)}
    for r in s.nodes.itertuples(index=False):
        if int(r.parent_id) >= 0:
            child_count[int(r.parent_id)] += 1
    # Recover the selected subtree membership by walking downward from the PP3 root.
    members=set([pp3["node"]]); stack=[pp3["node"]]
    while stack:
        cur=stack.pop()
        kids=[n for n,p in parent_map.items() if p==cur]
        for kid in kids:
            if kid not in members:
                members.add(kid); stack.append(kid)
    subtree_branches={n for n in members if child_count[n] >= 2}
    subtree_leaves={n for n in members if child_count[n] == 0}
    reduced_vertices=1+len(subtree_branches-{pp3["node"]})+len(subtree_leaves)
    reduced_edges=reduced_vertices-1
    root_row=s.nodes.set_index('node_id').loc[int(pp3["node"]), ['x','y','z']]
    root=np.asarray(root_row,dtype=float)
    rpoint=np.array([p.Root_x,p.Root_y,p.Root_z],float)
    rows.append(dict(ID=rid,Subtype=p.Subtype,Point_x=rpoint[0],Point_y=rpoint[1],Point_z=rpoint[2],
                     mesh_vertices=int(m.vertices.shape[0]),mesh_faces=int(m.faces.shape[0]),
                     skeleton_root_node_id=int(pp3["whole_root"]),pp3_selected_root_node_id=int(pp3["node"]),
                     pp3_score=float(pp3["score"]),pp3_subtree_nodes=int(pp3["subtree_nodes"]),
                     pp3_subtree_cable_nm=float(pp3["subtree_cable_nm"]),pp3_subtree_leaves=int(pp3["subtree_leaves"]),
                     pp3_subtree_branches_including_root=int(len(subtree_branches)),
                     reduced_vertices_numbers=int(reduced_vertices),reduced_segment_count=int(reduced_edges),
                     skel_nodes=int(len(s.nodes)),
                     pp3_root_x=root[0]/1000,pp3_root_y=root[1]/1000,pp3_root_z=root[2]/1000,
                     residual_x=root[0]/1000-rpoint[0],residual_y=root[1]/1000-rpoint[1],residual_z=root[2]/1000-rpoint[2],
                     residual_um=float(np.linalg.norm(root/1000-rpoint))))
df=pd.DataFrame(rows)
df.to_csv(OUT/"V271_mesh_skeleton_roots.csv",index=False)
summary={"versions":{"navis":navis.__version__,"fafbseg":fwy.__package__},"dataset_default":"fafbseg.get_mesh_neuron default","residuals":df.to_dict(orient="records"),
          "rms_um":float(np.sqrt(np.mean(df.residual_um**2))),"max_um":float(df.residual_um.max())}
(OUT/"V271_summary.json").write_text(json.dumps(summary,indent=2,default=str))
print(json.dumps(summary,indent=2,default=str))
