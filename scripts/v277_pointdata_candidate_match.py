#!/usr/bin/env python3
import json,pathlib,pickle,urllib.request,math
import numpy as np,pandas as pd
from fafbseg import flywire as fwy
IDS=[720575940632008007,720575940616224414,720575940625571465,720575940617782941]
PURL="https://raw.githubusercontent.com/borstlab/T4_T5_Dendrite_Morphology_Paper/56901ad1853b44aeca15504cd908fa4c31009a3e/Data/Point_data.pkl"
open("/tmp/p.pkl","wb").write(urllib.request.urlopen(PURL).read())
with open("/tmp/p.pkl","rb") as f:pdata=pickle.load(f)
out=pathlib.Path("v277_results");out.mkdir(exist_ok=True);allrows=[]
for rid in IDS:
    n=fwy.get_skeletons(rid,dataset=630,progress=False);df=n.nodes
    M={int(r.node_id):r for r in df.itertuples(index=False)};C={x:[] for x in M};root=None
    for r in df.itertuples(index=False):
        a,b=int(r.node_id),int(r.parent_id)
        if b<0:root=a
        else:C[b].append(a)
    order=[root];st=[root]
    while st:
        x=st.pop()
        for c in C[x]:order.append(c);st.append(c)
    E={};total=0
    for r in df.itertuples(index=False):
        a,b=int(r.node_id),int(r.parent_id)
        if b>=0:
            d=math.dist((r.x,r.y,r.z),(M[b].x,M[b].y,M[b].z));E[(b,a)]=d;total+=d
    L={x:(0 if C[x] else 1) for x in M};cab={x:0.0 for x in M}
    for x in reversed(order):
        for c in C[x]:cab[x]+=E[(x,c)]+cab[c];L[x]+=L[c]
    totalL=sum(1 for x in M if not C[x]); targets=pdata[pdata.ID==rid].iloc[0]
    cand=[]
    for x in M:
        if len(C[x])<2:continue
        S={x};st=[x]
        while st:
            y=st.pop()
            for c in C[y]:S.add(c);st.append(c)
        leaves=sum(1 for y in S if not any(c in S for c in C[y]))
        branches=sum(1 for y in S if sum(c in S for c in C[y])>1)
        score=(1-cab[x]/total)+L[x]/totalL
        err=abs(cab[x]-float(targets.Total_Cable))
        exact=(leaves==int(targets.Leaf_number) and branches==int(targets.Branch_number))
        cand.append({"node":x,"score":score,"leaves":leaves,"branches":branches,"segments":leaves+branches-1,"cable_nm":cab[x],"cable_error_nm":err,"root_xyz_um":(np.array([M[x].x,M[x].y,M[x].z])/1000).tolist(),"metric_exact":exact})
    cand=sorted(cand,key=lambda r:(not r["metric_exact"],r["cable_error_nm"]))
    pd.DataFrame(cand).to_csv(out/f"{rid}.csv",index=False)
    allrows.append({"ID":rid,"Subtype":str(targets.Subtype),"target":{"leaves":int(targets.Leaf_number),"branches":int(targets.Branch_number),"segments":int(targets.Segment_Count),"cable_nm":float(targets.Total_Cable)},"best":cand[:10]})
(out/"V277_summary.json").write_text(json.dumps(allrows,indent=2))
print(json.dumps(allrows,indent=2))
