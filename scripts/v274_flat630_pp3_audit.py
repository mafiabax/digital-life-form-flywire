#!/usr/bin/env python3
import json, pathlib, pickle, urllib.request, math
import numpy as np, pandas as pd
from fafbseg import flywire as fwy
import navis
IDS=[720575940632008007,720575940616224414,720575940625571465,720575940617782941]
SUB={IDS[0]:"T4a",IDS[1]:"T4c",IDS[2]:"T5a",IDS[3]:"T5c"}
URL="https://raw.githubusercontent.com/borstlab/T4_T5_Dendrite_Morphology_Paper/56901ad1853b44aeca15504cd908fa4c31009a3e/Data/Point_data.pkl"
p="/tmp/Point_data.pkl";open(p,"wb").write(urllib.request.urlopen(URL).read())
with open(p,"rb") as f: point=pickle.load(f)
def pp3(nodes):
    rows={int(r.node_id):r for r in nodes.itertuples(index=False)}
    ch={n:[] for n in rows}; roots=[]
    for r in nodes.itertuples(index=False):
        n,p=int(r.node_id),int(r.parent_id)
        if p<0: roots.append(n)
        else: ch[p].append(n)
    if len(roots)!=1: raise RuntimeError(roots)
    root=roots[0]; order=[root]; st=[root]
    while st:
        n=st.pop()
        for c in ch[n]: order.append(c);st.append(c)
    ec={}; total=0
    for r in nodes.itertuples(index=False):
        n,p=int(r.node_id),int(r.parent_id)
        if p>=0:
            d=math.dist((r.x,r.y,r.z),(rows[p].x,rows[p].y,rows[p].z));ec[(p,n)]=d;total+=d
    leaves={n:(1 if not ch[n] else 0) for n in rows}; cab={n:0.0 for n in rows}
    for n in reversed(order):
        for c in ch[n]: cab[n]+=ec[(n,c)]+cab[c];leaves[n]+=leaves[c]
    tl=sum(leaves[n] for n in rows if not ch[n]); best=None
    for n in rows:
        if len(ch[n])>=2:
            score=(1-cab[n]/total)+leaves[n]/tl
            if best is None or score>best[0]:best=(score,n)
    score,n=best; members={n};st=[n]
    while st:
        x=st.pop()
        for c in ch[x]:members.add(c);st.append(c)
    l=sum(1 for x in members if not any(c in members for c in ch[x]))
    b=sum(1 for x in members if sum(c in members for c in ch[x])>1)
    return n,score,len(members),l,b,cab[n]
out=pathlib.Path("v274_results");out.mkdir(exist_ok=True); rows=[]
for rid in IDS:
    p=point[point.ID==rid].iloc[0]
    m=fwy.get_mesh_neuron(rid,dataset="flat_630",omit_failures=False,progress=True)
    s=m.skeletonize(); s.units="1 nm"; navis.resample_skeleton(s,resample_to=100,inplace=True)
    q=pp3(s.nodes)
    rr=s.nodes.set_index("node_id").loc[q[0],["x","y","z"]].to_numpy(float)/1000
    pt=np.array([p.Root_x,p.Root_y,p.Root_z],float)
    rows.append({"ID":rid,"Subtype":SUB[rid],"point_root":pt.tolist(),"pp3_root":rr.tolist(),"residual_um":float(np.linalg.norm(rr-pt)),
                 "selected_node":q[0],"score":q[1],"subtree_nodes":q[2],"leaves":q[3],"branches":q[4],"cable_nm":q[5],
                 "point_segments":int(p.Segment_Count),"point_cable_nm":float(p.Total_Cable),"point_leaves":int(p.Leaf_number),"point_branches":int(p.Branch_number)})
df=pd.DataFrame(rows);df.to_csv(out/"V274_flat630_vs_Point_data.csv",index=False)
(out/"V274_summary.json").write_text(json.dumps({"rows":rows,"rms_um":float(np.sqrt(np.mean(df.residual_um**2)))},indent=2))
print(df.to_string(index=False));print("RMS",float(np.sqrt(np.mean(df.residual_um**2))))
