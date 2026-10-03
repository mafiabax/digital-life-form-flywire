import json,pathlib,numpy as np,pandas as pd,navis,flybrains
X=np.array([[789.1496875,263.18265625,210.35521875],[788.7438125,272.94928125,205.0135625],[716.6768125,222.2784375,211.779453125],[713.4343125,216.06415625,213.151640625]])
P=np.array([[-46.85120703125,32.261150390625,-111.5519921875],[-46.16929296875,22.233228515625,-116.4566640625],[11.981576171875,50.1895,120.4021328125],[12.7460712890625,54.907804687500004,117.5732734375]])
ids=[720575940632008007,720575940616224414,720575940625571465,720575940617782941]
rows=[]
for i,(rid,x,p) in enumerate(zip(ids,X,P)):
    try:
        y=navis.xform_brain(x.reshape(1,3),source="FLYWIREum",target="FAFB14um")[0]
        rows.append(dict(ID=rid,raw=x.tolist(),xformed=y.tolist(),point=p.tolist(),residual_um=float(np.linalg.norm(y-p))))
    except Exception as e: rows.append(dict(ID=rid,error=repr(e)))
out=pathlib.Path("v278_results");out.mkdir(exist_ok=True);(out/"V278_summary.json").write_text(json.dumps({"rows":rows,"navis":navis.__version__},indent=2));print(json.dumps(rows,indent=2))
