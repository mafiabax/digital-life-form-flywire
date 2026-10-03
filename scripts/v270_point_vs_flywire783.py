#!/usr/bin/env python3
import json, pickle, concurrent.futures, math, os, pathlib, traceback
import numpy as np
import pandas as pd
import navis

POINT_URL="https://raw.githubusercontent.com/borstlab/T4_T5_Dendrite_Morphology_Paper/56901ad1853b44aeca15504cd908fa4c31009a3e/Data/Point_data.pkl"
BASE="https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783"
OUT=pathlib.Path("v270_results"); OUT.mkdir(exist_ok=True)
MAX_N=int(os.environ.get("V270_MAX_N","512"))
THREADS=int(os.environ.get("V270_THREADS","12"))

os.system(f"curl -L --fail --retry 3 -o /tmp/Point_data.pkl '{POINT_URL}'")
with open("/tmp/Point_data.pkl","rb") as f: point=pickle.load(f)

cols=["ID","Type","Subtype","Root_x","Root_y","Root_z","Segment_Count","Vertices_numbers"]
work=point[cols].copy().drop_duplicates("ID").reset_index(drop=True)
work=work.head(MAX_N)
info={"historical_commit":"56901ad1853b44aeca15504cd908fa4c31009a3e","point_rows_total":len(point),
      "unique_ids_total":int(point.ID.nunique()),"sample_rows":len(work),"threads":THREADS}

SKELETON_INFO={"@type":"neuroglancer_skeletons","transform":[1,0,0,0,0,1,0,0,0,0,1,0],
               "vertex_attributes":[{"id":"radius","data_type":"float32","num_components":1}]}

def score_tree(df):
    # Published PP3 score: (1-subtree_cable/total_cable)+(subtree_leaves/total_leaves).
    node_ids=df["node_id"].astype(int).tolist()
    par=df["parent_id"].astype(int).tolist()
    by={int(r.node_id):r for r in df.itertuples(index=False)}
    children={n:[] for n in node_ids}
    roots=[]
    for n,p in zip(node_ids,par):
        if p < 0: roots.append(n)
        elif p in children: children[p].append(n)
    if len(roots)!=1: return None
    root=roots[0]
    edge={}
    total=0.0
    for r in df.itertuples(index=False):
        if int(r.parent_id)>=0 and int(r.parent_id) in by:
            d=math.dist((r.x,r.y,r.z),(by[int(r.parent_id)].x,by[int(r.parent_id)].y,by[int(r.parent_id)].z))
            edge[(int(r.parent_id),int(r.node_id))]=d
            total += d
    leaves=[n for n in node_ids if len(children[n])==0]
    if not leaves or total<=0: return None
    order=[]; st=[root]
    while st:
        n=st.pop(); order.append(n); st.extend(reversed(children[n]))
    cable={n:0.0 for n in node_ids}; lvs={n:(1 if n in set(leaves) else 0) for n in node_ids}
    for n in reversed(order):
        for c in children[n]:
            cable[n]+=edge[(n,c)]+cable[c]; lvs[n]+=lvs[c]
    best=None
    for n in node_ids:
        if len(children[n])<2: continue
        s=(1-cable[n]/total)+(lvs[n]/len(leaves))
        if best is None or s>best["score"]: best={"node":n,"score":s,"subtree_cable":cable[n],"subtree_leaves":lvs[n]}
    if best is None: return None
    return best

def one(row):
    rid=int(row.ID)
    try:
        tn=navis.read_precomputed(f"{BASE}/{rid}",datatype="skeleton",info=SKELETON_INFO)
        df=tn.nodes[["node_id","parent_id","x","y","z"]].copy()
        best=score_tree(df)
        if best is None: raise RuntimeError("PP3 score unavailable")
        n=df.loc[df.node_id.astype(int)==best["node"],].iloc[0]
        return {"ID":rid,"Subtype":row.Subtype,"Point_x":float(row.Root_x),"Point_y":float(row.Root_y),"Point_z":float(row.Root_z),
                "VFB_selected_node":int(best["node"]),"VFB_x_um":float(n.x)/1000,"VFB_y_um":float(n.y)/1000,"VFB_z_um":float(n.z)/1000,
                "VFB_nodes":int(len(df)),"status":"ok","score":best["score"],"subtree_cable":best["subtree_cable"],"subtree_leaves":best["subtree_leaves"]}
    except Exception as e:
        return {"ID":rid,"Subtype":row.Subtype,"status":"error","error":repr(e)}

with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
    res=list(ex.map(one,[r for _,r in work.iterrows()]))

df=pd.DataFrame(res)
df.to_csv(OUT/"V270_root_coordinate_sample.csv",index=False)
ok=df[df.status=="ok"].copy()
info["success"]=int(len(ok)); info["errors"]=int((df.status!="ok").sum())
if len(ok)>=8:
    X=ok[["VFB_x_um","VFB_y_um","VFB_z_um"]].to_numpy()
    Y=ok[["Point_x","Point_y","Point_z"]].to_numpy()
    # Similarity fit, plus full affine fit.
    def simfit(X,Y):
        mx=X.mean(0); my=Y.mean(0); A=X-mx; B=Y-my
        U,S,Vt=np.linalg.svd(A.T@B); R=U@Vt
        if np.linalg.det(R)<0: Vt[-1]*=-1; R=U@Vt
        scale=S.sum()/(A*A).sum()
        t=my-scale*mx@R
        pred=scale*X@R+t
        e=np.linalg.norm(pred-Y,axis=1)
        return scale,R,t,e
    scale,R,t,e=simfit(X,Y)
    A=np.c_[X,np.ones(len(X))]
    coef=np.linalg.lstsq(A,Y,rcond=None)[0]; pred=A@coef; ea=np.linalg.norm(pred-Y,axis=1)
    info.update({"similarity_scale":float(scale),"similarity_translation":t.tolist(),"similarity_rms_um":float(np.sqrt(np.mean(e*e))),
                 "similarity_median_um":float(np.median(e)),"similarity_max_um":float(np.max(e)),
                 "affine_rms_um":float(np.sqrt(np.mean(ea*ea))),"affine_median_um":float(np.median(ea)),"affine_max_um":float(np.max(ea)),
                 "selected_node_match_counts":ok.VFB_selected_node.value_counts().head(20).to_dict()})
(OUT/"V270_summary.json").write_text(json.dumps(info,indent=2,default=str))
print(json.dumps(info,indent=2,default=str))
