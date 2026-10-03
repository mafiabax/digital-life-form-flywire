#!/usr/bin/env python3
import json
import os
import pathlib
import subprocess
import numpy as np
import pandas as pd
import navis
import flybrains

RAW_PP3_UM = {
    720575940632008007: [786.4973715277778, 262.65043402777775, 209.97643706597222],
    720575940616224414: [788.7437874999999, 272.9492846590909, 205.01356051136364],
    720575940625571465: [723.8008709239131, 224.58238145380435, 215.02112364130434],
    720575940617782941: [713.2169583333334, 217.26955989583334, 213.54248697916665],
}
POINT_UM = {
    720575940632008007: [-46.85120703125, 32.261150390625, -111.5519921875],
    720575940616224414: [-46.16929296875, 22.233228515625, -116.4566640625],
    720575940625571465: [11.981576171875, 50.1895, 120.4021328125],
    720575940617782941: [12.7460712890625, 54.907804687500004, 117.5732734375],
}
OUT = pathlib.Path("v272_results")
OUT.mkdir(exist_ok=True)
DATA_HOME = pathlib.Path(os.environ.get("FLYBRAINS_DATA", "flybrain-data")).resolve()
DATA_HOME.mkdir(parents=True, exist_ok=True)

# The exact JRC2018F <-> FLYWIRE registration is a published Jefferis lab
# BridgingRegistrations entry. Clone at a fixed commit for auditability.
REG_REPO = "https://github.com/jefferislab/BridgingRegistrations.git"
REG_COMMIT = "b2aa89ace174fb8efb449ce07014310552ff8c3d"
reg_dir = DATA_HOME / "BridgingRegistrations"
if not reg_dir.exists():
    subprocess.run(["git", "clone", "--filter=blob:none", REG_REPO, str(reg_dir)], check=True)
subprocess.run(["git", "-C", str(reg_dir), "checkout", "--detach", REG_COMMIT], check=True)

# Re-register using the exact cloned registration tree.
flybrains.register_transforms()

ids = list(RAW_PP3_UM)
X = np.array([RAW_PP3_UM[i] for i in ids], dtype=float)
Y = np.array([POINT_UM[i] for i in ids], dtype=float)

rows=[]
for rid, raw, point in zip(ids, X, Y):
    try:
        out = navis.xform_brain(
            raw.reshape(1,3),
            source="FLYWIREum",
            target="JRC2018F",
        )[0]
        err = float(np.linalg.norm(out - point))
        rows.append({
            "ID": rid,
            "Subtype": {720575940632008007:"T4a",720575940616224414:"T4c",720575940625571465:"T5a",720575940617782941:"T5c"}[rid],
            "raw_pp3_x_um": raw[0], "raw_pp3_y_um": raw[1], "raw_pp3_z_um": raw[2],
            "point_x_um": point[0], "point_y_um": point[1], "point_z_um": point[2],
            "xformed_x_um": float(out[0]), "xformed_y_um": float(out[1]), "xformed_z_um": float(out[2]),
            "residual_x_um": float(out[0]-point[0]), "residual_y_um": float(out[1]-point[1]),
            "residual_z_um": float(out[2]-point[2]), "residual_um": err,
            "status":"ok"
        })
    except Exception as e:
        rows.append({"ID":rid,"status":"error","error":repr(e)})

df=pd.DataFrame(rows)
df.to_csv(OUT/"V272_FLYWIRE_to_JRC2018F.csv",index=False)
ok=df[df.status=="ok"]
summary={
  "navis_version":navis.__version__,
  "flybrains_version":getattr(flybrains,"__version__","unknown"),
  "registration_repo":REG_REPO,
  "registration_commit":REG_COMMIT,
  "registration_path":"JRC2018F_FLYWIRE.list",
  "source_frame":"FLYWIREum",
  "target_frame":"JRC2018Fum",
  "rows":rows,
  "status":"success" if len(ok)==len(df) else "partial",
}
if len(ok):
    summary.update({
      "rms_um":float(np.sqrt(np.mean(ok.residual_um.to_numpy()**2))),
      "median_um":float(np.median(ok.residual_um.to_numpy())),
      "max_um":float(np.max(ok.residual_um.to_numpy()))
    })
(OUT/"V272_summary.json").write_text(json.dumps(summary,indent=2,default=str))
print(json.dumps(summary,indent=2,default=str))
