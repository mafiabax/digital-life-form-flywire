#!/usr/bin/env python3
import json, pathlib, pickle, urllib.request, math, gzip, io, zlib
import numpy as np, pandas as pd
IDS=[720575940632008007,720575940616224414,720575940625571465,720575940617782941]
SUB={IDS[0]:"T4a",IDS[1]:"T4c",IDS[2]:"T5a",IDS[3]:"T5c"}
PURL="https://raw.githubusercontent.com/borstlab/T4_T5_Dendrite_Morphology_Paper/56901ad1853b44aeca15504cd908fa4c31009a3e/Data/Point_data.pkl"
open("/tmp/Point_data.pkl","wb").write(urllib.request.urlopen(PURL).read())
with open("/tmp/Point_data.pkl","rb") as f: point=pickle.load(f)
def parse(txt):
    ns=[]
    for l in txt.splitlines():
        if not l.strip() or l.lstrip().startswith("#"): continue
        a=l.split()
        if len(a)>=7: ns.append(dict(id=int(a[0]),x=float(a[2]),y=float(a[3]),z=float(a[4]),p=int(a[6])))
    return ns
def pp3(ns):
    M={n["id"]:n for n in ns}; C={n["id"]:[] for n in ns}; root=[]
    for n in ns:
        if n["p"]<0: root.append(n["id"])
        else:
            if n["p"] not in C: C[n["p"]]=[]
            C[n["p"]].append(n["id"])
    if len(root)!=1: raise RuntimeError(f"roots={root}")
    order=[root[0]];st=[root[0]]
    while st:
        n=st.pop()
        for c in C[n]:order.append(c);st.append(c)
    edge={};total=0
    for n in ns:
        if n["p"]>=0:
            p=M[n["p"]];d=math.dist((n["x"],n["y"],n["z"]),(p["x"],p["y"],p["z"]));edge[(n["p"],n["id"])]=d;total+=d
    leaf={n["id"]:(0 if C[n["id"]] else 1) for n in ns};cab={n["id"]:0.0 for n in ns}
    for n in reversed(order):
        for c in C[n]:cab[n]+=edge[(n,c)]+cab[c];leaf[n]+=leaf[c]
    L=sum(1 for n in ns if not C[n]);best=max(((1-cab[n]/total)+leaf[n]/L,n) for n in M if len(C[n])>=2);score,r=best
    S={r};st=[r]
    while st:
        n=st.pop()
        for c in C[n]:S.add(c);st.append(c)
    leaves=sum(1 for n in S if not any(c in S for c in C[n]))
    branches=sum(1 for n in S if sum(c in S for c in C[n])>1)
    return r,score,len(S),leaves,branches,cab[r]
out=pathlib.Path("v275_results");out.mkdir(exist_ok=True);rows=[]
for rid in IDS:
    url=f"https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_630/{rid}"
    req=urllib.request.Request(url,headers={"User-Agent":"Digital-Life-Form-V275/1.0"})
    raw=urllib.request.urlopen(req,timeout=120).read()
    print("payload_prefix", raw[:16].hex())
    decoded=None
    for fn in (
        lambda b: gzip.decompress(b),
        lambda b: zlib.decompress(b),
        lambda b: zlib.decompress(b, -zlib.MAX_WBITS),
    ):
        try:
            decoded=fn(raw)
            break
        except Exception:
            pass
    if decoded is None:
        raise RuntimeError(f"Unable to decompress historical payload; prefix={raw[:32].hex()}")
    data=decoded.decode()
    ns=parse(data); q=pp3(ns); M={n["id"]:n for n in ns}
    root=np.array([M[q[0]]["x"],M[q[0]]["y"],M[q[0]]["z"]])/1000
    p=point[point.ID==rid].iloc[0];pt=np.array([p.Root_x,p.Root_y,p.Root_z])
    rows.append({"ID":rid,"Subtype":SUB[rid],"source":url,"source_nodes":len(ns),"pp3_selected_node":q[0],"pp3_score":q[1],"pp3_subtree_nodes":q[2],"pp3_leaves":q[3],"pp3_branches":q[4],"pp3_cable_nm":q[5],"pp3_root_um":root.tolist(),"point_root_um":pt.tolist(),"residual_um":float(np.linalg.norm(root-pt)),"point_segments":int(p.Segment_Count),"point_cable_nm":float(p.Total_Cable),"point_leaves":int(p.Leaf_number),"point_branches":int(p.Branch_number)})
df=pd.DataFrame(rows);df.to_csv(out/"V275_historical_630_endpoint_vs_Point_data.csv",index=False)
(out/"V275_summary.json").write_text(json.dumps({"rows":rows,"rms_um":float(np.sqrt(np.mean(df.residual_um**2)))},indent=2))
print(df.to_string(index=False))
