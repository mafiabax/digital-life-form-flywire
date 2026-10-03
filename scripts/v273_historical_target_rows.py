#!/usr/bin/env python3
import json, pathlib, pickle, urllib.request
import pandas as pd
URL="https://raw.githubusercontent.com/borstlab/T4_T5_Dendrite_Morphology_Paper/56901ad1853b44aeca15504cd908fa4c31009a3e/Data/Point_data.pkl"
ids=[720575940632008007,720575940616224414,720575940625571465,720575940617782941]
p="/tmp/Point_data.pkl"
open(p,"wb").write(urllib.request.urlopen(URL).read())
with open(p,"rb") as f: df=pickle.load(f)
sel=df[df.ID.isin(ids)].copy()
sel=sel.set_index("ID").loc[ids].reset_index()
out=pathlib.Path("v273_results"); out.mkdir(exist_ok=True)
sel.to_csv(out/"V273_historical_target_rows.csv",index=False)
summary={"rows":sel.to_dict(orient="records"),"columns":[str(c) for c in sel.columns],"shape":list(sel.shape)}
(out/"V273_summary.json").write_text(json.dumps(summary,indent=2,default=str))
print(json.dumps(summary,indent=2,default=str))
