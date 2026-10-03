#!/usr/bin/env python3
import json, math, pathlib
import numpy as np
import pandas as pd
from fafbseg import flywire as fwy
import navis, skeletor

IDS=[
  (720575940632008007,"T4a"),
  (720575940616224414,"T4c"),
  (720575940625571465,"T5a"),
  (720575940617782941,"T5c"),
]
POINT={
  720575940632008007:(-46.85120703125,32.261150390625,-111.5519921875),
  720575940616224414:(-46.16929296875,22.233228515625,-116.4566640625),
  720575940625571465:(11.981576171875,50.1895,120.4021328125),
  720575940617782941:(12.7460712890625,54.907804687500004,117.5732734375),
}
out=pathlib.Path("v274_results"); out.mkdir(exist_ok=True)

def pp3(nodes):
    df=nodes[["node_id","parent_id","x","y","z"]].copy()
    by={int(r.node_id):r for r in df.itertuples(index=False)}
    children={int(r.node_id):[] for r in df.itertuples(index=False)}
    roots=[]
    for r in df.itertuples(index=False):
        n,p=int(r.node_id),int(r.parent_id)
        if p<0: roots.append(n)
        else: children[p].append(n)
    root=roots[0]
    edge={}; total=0.
    for r in df.itertuples(index=False):
        n,p=int(r.node_id),int(r.parent_id)
        if p>=0:
            d=math.dist((r.x,r.y,r.z),(by[p].x,by[p].y,by[p].z))
            edge[(p,n)]=d; total+=d
    leaves={n for n,ch in children.items() if not ch}
    order=[]; st=[root]
    while st:
        n=st.pop(); order.append(n); st.extend(reversed(children[n]))
    cable={n:0. for n in children}; lvc={n:1. if n in leaves else 0. for n in children}
    for n in reversed(order):
        for c in children[n]:
            cable[n]+=edge[(n,c)]+cable[c]; lvc[n]+=lvc[c]
    cand=[]
    for n,ch in children.items():
        if len(ch)>=2:
            score=(1-cable[n]/total)+(lvc[n]/len(leaves))
            cand.append((score,n,cable[n],int(lvc[n])))
    score,node,subc,subl=max(cand)
    members={node}; st=[node]
    while st:
        n=st.pop()
        for c in children[n]:
            if c not in members: members.add(c); st.append(c)
    b=sum(1 for n in members if len(children[n])>=2)
    l=sum(1 for n in members if len(children[n])==0)
    reduced=1+(b-1)+l
    coord=by[node]
    return dict(node=int(node),score=float(score),subtree_nodes=len(members),
                subtree_cable_nm=float(subc),subtree_leaves=l,subtree_branches=b,
                reduced_vertices=reduced,reduced_segments=reduced-1,
                root_xyz_nm=[float(coord.x),float(coord.y),float(coord.z)])

rows=[]
for rid,name in IDS:
    for lod in (0,1,2,3):
        m=fwy.get_mesh_neuron(rid,dataset="flat_783",lod=lod,omit_failures=False,progress=False)
        s=m.skeletonize()
        navis.resample_skeleton(s,resample_to="100 nanometer",inplace=True)
        d=pp3(s.nodes)
        p=np.array(POINT[rid]); r=np.array(d["root_xyz_nm"])/1000
        rows.append({"ID":rid,"Subtype":name,"requested_lod":lod,
                     "mesh_vertices":int(m.vertices.shape[0]),"mesh_faces":int(m.faces.shape[0]),
                     "skeleton_nodes":int(len(s.nodes)),
                     "Point_x":p[0],"Point_y":p[1],"Point_z":p[2],
                     "PP3_x":r[0],"PP3_y":r[1],"PP3_z":r[2],
                     "residual_um":float(np.linalg.norm(r-p)),
                     **d})
df=pd.DataFrame(rows)
df.to_csv(out/"V274_lod_sweep.csv",index=False)
summary={"navis":navis.__version__,"skeletor":skeletor.__version__,
         "fafbseg":getattr(fwy,"__version__","unknown"),"dataset":"flat_783",
         "rows":rows}
(out/"V274_summary.json").write_text(json.dumps(summary,indent=2,default=str))
print(json.dumps(summary,indent=2,default=str))
