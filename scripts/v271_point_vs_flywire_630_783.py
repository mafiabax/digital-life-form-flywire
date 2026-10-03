#!/usr/bin/env python3
import json, math, concurrent.futures, os, pathlib, pickle, subprocess
import navis, pandas as pd, numpy as np

ROOT="https://raw.githubusercontent.com/borstlab/T4_T5_Dendrite_Morphology_Paper/56901ad1853b44aeca15504cd908fa4c31009a3e"
OUT=pathlib.Path("v271_results"); OUT.mkdir(exist_ok=True)
MAX_N=int(os.environ.get("V271_MAX_N","64")); THREADS=int(os.environ.get("V271_THREADS","12"))
subprocess.run(["curl","-L","--fail","--retry","3","-o","/tmp/Point_data.pkl",f"{ROOT}/Data/Point_data.pkl"],check=True)
with open("/tmp/Point_data.pkl","rb") as f: point=pickle.load(f)
work=point[["ID","Type","Subtype","Root_x","Root_y","Root_z"]].drop_duplicates("ID").head(MAX_N).reset_index(drop=True)

def score_tree(df):
    ids=df.node_id.astype(int).tolist(); par=df.parent_id.astype(int).tolist(); by={int(r.node_id):r for r in df.itertuples(index=False)}
    ch={n:[] for n in ids}; roots=[]
    for n,p in zip(ids,par):
        if p<0: roots.append(n)
        elif p in ch: ch[p].append(n)
    if len(roots)!=1:return None
    root=roots[0]; edge={}; total=0.0
    for r in df.itertuples(index=False):
        if int(r.parent_id)>=0 and int(r.parent_id) in by:
            d=math.dist((r.x,r.y,r.z),(by[int(r.parent_id)].x,by[int(r.parent_id)].y,by[int(r.parent_id)].z)); edge[(int(r.parent_id),int(r.node_id))]=d; total+=d
    leaves=[n for n in ids if not ch[n]]
    order=[]; st=[root]
    while st:
        n=st.pop(); order.append(n); st.extend(reversed(ch[n]))
    cable={n:0.0 for n in ids}; lvs={n:(1 if n in set(leaves) else 0) for n in ids}
    for n in reversed(order):
        for c in ch[n]: cable[n]+=edge[(n,c)]+cable[c]; lvs[n]+=lvs[c]
    best=None
    for n in ids:
        if len(ch[n])<2:continue
        s=(1-cable[n]/total)+(lvs[n]/len(leaves))
        if best is None or s>best["score"]:best={"node":n,"score":s,"cable":cable[n],"leaves":lvs[n]}
    if best is None:return None
    r=by[best["node"]]
    return dict(node=int(best["node"]),score=float(best["score"]),x_um=float(r.x)/1000,y_um=float(r.y)/1000,z_um=float(r.z)/1000,
                nodes=len(ids),subtree_cable=float(best["cable"]),subtree_leaves=int(best["leaves"]))

def one(row,ds):
    rid=int(row.ID)
    try:
        tn=navis.read_precomputed(f"https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_{ds}/{rid}",
                                  datatype="skeleton",
                                  info={"@type":"neuroglancer_skeletons","transform":[1,0,0,0,0,1,0,0,0,0,1,0],
                                        "vertex_attributes":[{"id":"radius","data_type":"float32","num_components":1}]})
        df=tn.nodes[["node_id","parent_id","x","y","z"]]
        b=score_tree(df)
        if b is None: raise RuntimeError("no PP3 branch")
        return dict(ID=rid,Subtype=row.Subtype,Point_x=row.Root_x,Point_y=row.Root_y,Point_z=row.Root_z,dataset=ds,status="ok",**{f"VFB_{k}":v for k,v in b.items()})
    except Exception as e:
        return dict(ID=rid,Subtype=row.Subtype,dataset=ds,status="error",error=repr(e))

allr=[]
for ds in (630,783):
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        allr.extend(list(ex.map(lambda r:one(r,ds),[r for _,r in work.iterrows()])))
df=pd.DataFrame(allr); df.to_csv(OUT/"V271_roots_630_vs_783.csv",index=False)

for ds in (630,783):
    ok=df[(df.dataset==ds)&(df.status=="ok")].copy()
    if len(ok)<3:
        continue
    X=ok[["VFB_x_um","VFB_y_um","VFB_z_um"]].to_numpy(); Y=ok[["Point_x","Point_y","Point_z"]].to_numpy()
    def simfit(X,Y):
        mx=X.mean(0); my=Y.mean(0); A=X-mx; B=Y-my
        U,S,Vt=np.linalg.svd(A.T@B); R=U@Vt
        if np.linalg.det(R)<0: Vt[-1]*=-1; R=U@Vt
        scale=S.sum()/(A*A).sum(); t=my-scale*mx@R; pred=scale*X@R+t; e=np.linalg.norm(pred-Y,axis=1)
        return float(scale),t.tolist(),float(np.sqrt(np.mean(e*e))),float(np.median(e)),float(np.max(e))
    sc,t,rms,med,mx=simfit(X,Y)
    summary={"dataset":ds,"success":int(len(ok)),"errors":int((df[df.dataset==ds].status!="ok").sum()),
             "similarity_scale":sc,"similarity_translation_um":t,"similarity_rms_um":rms,"similarity_median_um":med,"similarity_max_um":mx}
    pathlib.Path(OUT/f"V271_summary_{ds}.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
