#!/usr/bin/env python3
import json, pathlib, pickle, urllib.request, math
import numpy as np, pandas as pd
from fafbseg import flywire as fwy
import navis
IDS=[720575940632008007,720575940616224414,720575940625571465,720575940617782941]
SUB={IDS[0]:"T4a",IDS[1]:"T4c",IDS[2]:"T5a",IDS[3]:"T5c"}
URL="https://raw.githubusercontent.com/borstlab/T4_T5_Dendrite_Morphology_Paper/56901ad1853b44aeca15504cd908fa4c31009a3e/Data/Point_data.pkl"
open("/tmp/Point_data.pkl","wb").write(urllib.request.urlopen(URL).read())
with open("/tmp/Point_data.pkl","rb") as f: point=pickle.load(f)
def pp3(nodes):
    M={int(r.node_id):r for r in nodes.itertuples(index=False)};C={n:[] for n in M};root=[]
    for r in nodes.itertuples(index=False):
        n,p=int(r.node_id),int(r.parent_id)
        if p<0:root.append(n)
        else:C[p].append(n)
    if len(root)!=1:raise RuntimeError(root)
    order=[root[0]];st=[root[0]]
    while st:
        n=st.pop()
        for c in C[n]:order.append(c);st.append(c)
    E={};total=0
    for r in nodes.itertuples(index=False):
        n,p=int(r.node_id),int(r.parent_id)
        if p>=0:
            d=math.dist((r.x,r.y,r.z),(M[p].x,M[p].y,M[p].z));E[(p,n)]=d;total+=d
    L={n:(0 if C[n] else 1) for n in M};cab={n:0.0 for n in M}
    for n in reversed(order):
        for c in C[n]:cab[n]+=E[(n,c)]+cab[c];L[n]+=L[c]
    totalL=sum(1 for n in M if not C[n]);score={n:(1-cab[n]/total)+L[n]/totalL if len(C[n])>=2 else 0 for n in M}
    r=max(score,key=score.get);S={r};st=[r]
    while st:
        n=st.pop()
        for c in C[n]:S.add(c);st.append(c)
    leaves=sum(1 for n in S if not any(c in S for c in C[n]));branches=sum(1 for n in S if sum(c in S for c in C[n])>1)
    return r,score[r],len(S),leaves,branches,cab[r]
out=pathlib.Path("v276_results");out.mkdir(exist_ok=True);rows=[]
for rid in IDS:
    n=fwy.get_skeletons(rid,dataset=630,progress=False)
    q=pp3(n.nodes); rr=n.nodes.set_index("node_id").loc[q[0],["x","y","z"]].to_numpy(float)/1000
    p=point[point.ID==rid].iloc[0];pt=np.array([p.Root_x,p.Root_y,p.Root_z])
    rows.append({"ID":rid,"Subtype":SUB[rid],"source_dataset":630,"source_nodes":int(n.n_nodes),"selected_node":q[0],"score":q[1],"subtree_nodes":q[2],"leaves":q[3],"branches":q[4],"pp3_cable_nm":q[5],"pp3_root_um":rr.tolist(),"point_root_um":pt.tolist(),"residual_um":float(np.linalg.norm(rr-pt)),"point_segments":int(p.Segment_Count),"point_cable_nm":float(p.Total_Cable),"point_leaves":int(p.Leaf_number),"point_branches":int(p.Branch_number)})
df=pd.DataFrame(rows);df.to_csv(out/"V276_precomputed630_vs_Point_data.csv",index=False)
(out/"V276_summary.json").write_text(json.dumps({"rows":rows,"rms_um":float(np.sqrt(np.mean(df.residual_um**2)))},indent=2))
print(df.to_string(index=False))
