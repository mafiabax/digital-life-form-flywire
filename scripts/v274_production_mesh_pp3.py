#!/usr/bin/env python3
import json, pathlib, pickle, urllib.request, numpy as np, pandas as pd
from fafbseg import flywire as fwy
import navis
IDS=[720575940632008007,720575940616224414,720575940625571465,720575940617782941]
URL="https://raw.githubusercontent.com/borstlab/T4_T5_Dendrite_Morphology_Paper/56901ad1853b44aeca15504cd908fa4c31009a3e/Data/Point_data.pkl"
p="/tmp/Point_data.pkl"; open(p,"wb").write(urllib.request.urlopen(URL).read())
with open(p,"rb") as f: point=pickle.load(f)
rows=[]
for rid in IDS:
    m=fwy.get_mesh_neuron(rid,dataset="production",omit_failures=False,progress=True)
    s=m.skeletonize()
    s.units="1 nm"
    navis.resample_skeleton(s,resample_to=100,inplace=True)
    # published PP3 selector
    df=s.nodes.copy(); by={int(r.node_id):r for r in df.itertuples(index=False)}
    ch={int(i):[] for i in df.node_id}; roots=[]
    for r in df.itertuples(index=False):
        n,pid=int(r.node_id),int(r.parent_id)
        if pid<0: roots.append(n)
        else: ch[pid].append(n)
    root=roots[0]; order=[]; stack=[root]
    while stack:
        n=stack.pop(); order.append(n); stack.extend(reversed(ch[n]))
    el={}; total=0
    for r in df.itertuples(index=False):
        n,pid=int(r.node_id),int(r.parent_id)
        if pid>=0:
            d=((r.x-by[pid].x)**2+(r.y-by[pid].y)**2+(r.z-by[pid].z)**2)**.5
            el[(pid,n)]=d; total+=d
    leaves={n for n,v in ch.items() if not v}; sc={n:0.0 for n in ch}; sl={n:float(n in leaves) for n in ch}
    for n in reversed(order):
        for cc in ch[n]: sc[n]+=el[(n,cc)]+sc[cc]; sl[n]+=sl[cc]
    cand=[((1-sc[n]/total)+(sl[n]/len(leaves)),n) for n in ch if len(ch[n])>=2]
    sel=max(cand)[1]
    rr=df.set_index("node_id").loc[sel,["x","y","z"]].to_numpy(float)/1000
    pnt=point[point.ID==rid].iloc[0][["Root_x","Root_y","Root_z"]].to_numpy(float)
    rows.append(dict(ID=rid,Subtype=point[point.ID==rid].iloc[0].Subtype,mesh_vertices=len(m.vertices),skel_nodes=len(s.nodes),selected=sel,x=rr[0],y=rr[1],z=rr[2],px=pnt[0],py=pnt[1],pz=pnt[2],residual_um=float(np.linalg.norm(rr-pnt))))
out=pathlib.Path("v274_results"); out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out/"V274_production_pp3.csv",index=False)
(out/"V274_summary.json").write_text(json.dumps({"rows":rows,"rms_um":float(np.sqrt(np.mean(np.array([x["residual_um"] for x in rows])**2)))},indent=2))
print(json.dumps(rows,indent=2))
