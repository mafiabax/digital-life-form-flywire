#!/usr/bin/env python3
import json, math, pathlib, urllib.request, pickle
import numpy as np, pandas as pd
from scipy.spatial import ConvexHull
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.transform import Rotation
from fafbseg import flywire as fwy
from fafbseg.flywire import NeuronCriteria as NC
import navis, skeletor

OUT=pathlib.Path("v275_results"); OUT.mkdir(exist_ok=True)
URL="https://raw.githubusercontent.com/borstlab/T4_T5_Dendrite_Morphology_Paper/56901ad1853b44aeca15504cd908fa4c31009a3e/Data/Point_data.pkl"
data=urllib.request.urlopen(URL).read(); (OUT/"Point_data.sha256.txt").write_text(__import__("hashlib").sha256(data).hexdigest())
point=pickle.loads(data)

# Use the historical Point_data hemisphere labels directly. This keeps the audit
# independent of a live CAVE token and prevents current annotation drift.
p=point.copy()
p["group"]=p["Subtype"].astype(str)
p=p.drop_duplicates("ID").sort_values(["group","ID"]).reset_index(drop=True)

# Deterministic farthest-point sample in Point_data's already-transformed space.
def fps(g,n):
    g=g.copy()
    if len(g)<=n: return g
    X=g[["Root_x","Root_y","Root_z"]].to_numpy(float)
    chosen=[int(np.argmin(np.linalg.norm(X-X.mean(0),axis=1)))]
    d=np.linalg.norm(X-X[chosen[0]],axis=1)
    while len(chosen)<n:
        j=int(np.argmax(d)); chosen.append(j)
        d=np.minimum(d,np.linalg.norm(X-X[j],axis=1))
    return g.iloc[chosen]

SAMP=pd.concat([fps(g,8) for _,g in p.groupby("group")],ignore_index=True)
SAMP.to_csv(OUT/"V275_sample_ids.csv",index=False)

def pp3(nodes):
    df=nodes[["node_id","parent_id","x","y","z"]].copy()
    by={int(r.node_id):r for r in df.itertuples(index=False)}
    children={int(r.node_id):[] for r in df.itertuples(index=False)}
    roots=[]
    for r in df.itertuples(index=False):
        n,p=int(r.node_id),int(r.parent_id)
        if p<0: roots.append(n)
        elif p in children: children[p].append(n)
    if len(roots)!=1: raise RuntimeError(f"roots={roots}")
    root=roots[0]
    edge={}; total=0.
    for r in df.itertuples(index=False):
        n,p=int(r.node_id),int(r.parent_id)
        if p>=0:
            d=math.dist((r.x,r.y,r.z),(by[p].x,by[p].y,by[p].z)); edge[(p,n)]=d; total+=d
    leaves={n for n,ch in children.items() if not ch}
    order=[]; st=[root]
    while st:
        n=st.pop(); order.append(n); st.extend(reversed(children[n]))
    cable={n:0. for n in children}; lc={n:(1 if n in leaves else 0) for n in children}
    for n in reversed(order):
        for c in children[n]:
            cable[n]+=edge[(n,c)]+cable[c]; lc[n]+=lc[c]
    cand=[]
    for n,ch in children.items():
        if len(ch)>=2:
            s=(1-cable[n]/total)+(lc[n]/len(leaves)); cand.append((s,n,cable[n],lc[n]))
    score,node,subc,subl=max(cand)
    xyz=np.array([by[node].x,by[node].y,by[node].z],float)
    return {"node":int(node),"score":float(score),"xyz":xyz,"subtree_cable_nm":float(subc)}

# Small Kabsch helper: maps X -> Y with one rotation + translation.
def rigid_fit(X,Y):
    mx=X.mean(0); my=Y.mean(0); A=X-mx; B=Y-my
    U,S,Vt=np.linalg.svd(A.T@B); R=U@Vt
    if np.linalg.det(R)<0: Vt[-1]*=-1; R=U@Vt
    t=my-mx@R
    pred=X@R+t; e=np.linalg.norm(pred-Y,axis=1)
    return R,t,e

raw_rows=[]; trees={}; failures=[]
for group,g in SAMP.groupby("group"):
    ids=g.ID.astype(int).tolist()
    try:
        meshes=fwy.get_mesh_neuron(ids,dataset="flat_783",lod=1,omit_failures=True,threads=12,progress=True)
        if not isinstance(meshes,(list,tuple)): meshes=[meshes]
        by_id={}
        for m in meshes:
            mid=getattr(m,"id",None)
            if mid is None: continue
            arr=np.asarray(mid).reshape(-1)
            if arr.size != 1: continue
            by_id[int(arr[0])]=m
        for row in g.itertuples(index=False):
            rid=int(row.ID)
            m=by_id.get(rid)
            if m is None: failures.append({"ID":rid,"group":group,"stage":"mesh"}); continue
            m.units="1 nanometer"
            s=m.skeletonize()
            navis.resample_skeleton(s,resample_to="100 nanometer",inplace=True)
            d=pp3(s.nodes)
            raw=np.array(d["xyz"])/1000.0
            raw_rows.append({"ID":rid,"Subtype":row.Subtype,"group":group,
                             "Point_x":row.Root_x,"Point_y":row.Root_y,"Point_z":row.Root_z,
                             "raw_x":raw[0],"raw_y":raw[1],"raw_z":raw[2],
                             "skeleton_nodes":len(s.nodes),"pp3_score":d["score"]})
            trees.setdefault(group,[]).append((rid,s,d))
    except Exception as e:
        failures.append({"group":group,"stage":"group","error":repr(e)})

rawdf=pd.DataFrame(raw_rows)
rawdf.to_csv(OUT/"V275_raw_pp3_vs_point.csv",index=False)

# For each group, fit raw-root -> Point_data rigid transform and also reproduce
# the documented PP4 concept on the pooled sampled skeleton node coordinates:
# centroid/PCA alignment -> sphere fit -> recenter -> 180° Z rotation.
summary={"versions":{"navis":navis.__version__,"skeletor":skeletor.__version__},
         "dataset":"flat_783","sample_per_group":8,
         "groups":{}, "failures":failures}

def pca_align(X):
    c=X.mean(0); A=X-c
    # robust=False here; the paper notebook did not override align_forest defaults,
    # while historical implementation may differ. This is an independent frame test.
    C=np.cov(A,rowvar=False); vals,vecs=np.linalg.eigh(C); idx=np.argsort(vals)[::-1]; V=vecs[:,idx]
    # orient signs deterministically so PCA axes have a stable handed basis.
    if np.linalg.det(V)<0: V[:,-1]*=-1
    # map columns PC1,PC2,PC3 to canonical y,x,z using orthogonal Procrustes.
    B=np.eye(3)[:,[1,0,2]]
    R,_=orthogonal_procrustes(V,B)
    if np.linalg.det(R)<0: R[:,-1]*=-1
    return c,R

def fit_sphere(X):
    A=np.c_[2*X,np.ones(len(X))]; b=(X**2).sum(1)
    sol=np.linalg.lstsq(A,b,rcond=None)[0]; return sol[:3], math.sqrt(max(sol[3]+(sol[:3]**2).sum(),0))

# map each tree's raw coordinates with one group transform
for group, items in trees.items():
    if group not in set(rawdf.group): continue
    rd=rawdf[rawdf.group==group].copy()
    merged=rd
    X=merged[["raw_x","raw_y","raw_z"]].to_numpy(float)
    Y=merged[["Point_x","Point_y","Point_z"]].to_numpy(float)
    if len(X)>=3:
        R,t,e=rigid_fit(X,Y)
        group_out={"n":len(X),
                   "rigid_rms_um":float(np.sqrt(np.mean(e*e))),
                   "rigid_median_um":float(np.median(e)),
                   "rigid_max_um":float(np.max(e)),
                   "rigid_R":R.tolist(),"rigid_t":t.tolist()}
    else: group_out={"n":len(X)}
    # pooled nodes for approximate PP4 transform.
    pooled=[]
    root_records=[]
    for rid,s,d in items:
        xyz=s.nodes[["x","y","z"]].to_numpy(float)/1000.0
        pooled.append(xyz)
        root_records.append((rid,np.array(d["xyz"])/1000.0))
    P=np.vstack(pooled)
    c,Rp=pca_align(P); PA=(P-c)@Rp
    center,rad=fit_sphere(PA); PA2=PA-center
    rz=np.array([[-1.,0,0],[0,-1,0],[0,0,1.]])
    PA3=PA2@rz
    # derive each root's transformed coordinate using same PP4-like transform.
    approx=[]
    for rid,rawroot in root_records:
        q=(rawroot-c)@Rp-center
        q=q@rz
        target=merged.loc[merged.ID==rid,["Point_x","Point_y","Point_z"]].to_numpy(float)[0]
        approx.append(np.linalg.norm(q-target))
    group_out.update({"pp4_sample_sphere_center_um":center.tolist(),
                      "pp4_sample_sphere_radius_um":float(rad),
                      "pp4_like_rms_um":float(np.sqrt(np.mean(np.asarray(approx)**2))) if approx else None,
                      "pp4_like_median_um":float(np.median(approx)) if approx else None,
                      "pp4_like_max_um":float(np.max(approx)) if approx else None})
    summary["groups"][group]=group_out

(OUT/"V275_summary.json").write_text(json.dumps(summary,indent=2,default=str))
print(json.dumps(summary,indent=2,default=str))
