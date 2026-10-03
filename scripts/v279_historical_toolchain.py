import json,pathlib,pickle,urllib.request,math,numpy as np,pandas as pd
from fafbseg import flywire as fwy
import navis
IDS=[720575940632008007,720575940616224414,720575940625571465,720575940617782941]
open("/tmp/p.pkl","wb").write(urllib.request.urlopen("https://raw.githubusercontent.com/borstlab/T4_T5_Dendrite_Morphology_Paper/56901ad1853b44aeca15504cd908fa4c31009a3e/Data/Point_data.pkl").read())
with open("/tmp/p.pkl","rb") as f:pdata=pickle.load(f)
out=pathlib.Path("v279_results");out.mkdir(exist_ok=True);rows=[]
for rid in IDS:
 m=fwy.get_mesh_neuron(rid,omit_failures=False,progress=False);s=m.skeletonize();s.units="1 nm";navis.resample_skeleton(s,resample_to="100 nanometer",inplace=True)
 rr=s.nodes.set_index("node_id").loc[s.root,["x","y","z"]].to_numpy(float)/1000
 p=pdata[pdata.ID==rid].iloc[0];pt=np.array([p.Root_x,p.Root_y,p.Root_z])
 rows.append({"ID":rid,"nodes":int(len(s.nodes)),"root_um":rr.tolist(),"point_um":pt.tolist(),"residual_um":float(np.linalg.norm(rr-pt)),"point_segments":int(p.Segment_Count),"point_cable":float(p.Total_Cable)})
pd.DataFrame(rows).to_csv(out/"V279_historical_toolchain_roots.csv",index=False)
(out/"V279_summary.json").write_text(json.dumps({"fafbseg":fwy.__package__,"navis":navis.__version__,"rows":rows},indent=2));print(json.dumps({"fafbseg":fwy.__package__,"navis":navis.__version__,"rows":rows},indent=2))
