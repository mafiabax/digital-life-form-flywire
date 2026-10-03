# V274 — Historical Point_data metric/frame reconciliation

## Evidence chain

The historical study repository is pinned to commit `3a1aa1a2e368ff8767f40791588eaf552e6d436d`.

1. `Notebooks/PP3_Dendrite_extraction.ipynb`:
   - `forest.convert_forest_to_subtrees(...)`
   - saves dendrites
   - `forest.reduce_forest(..., inplace=True)`
   - saves reduced dendrites.
2. `Notebooks/Metrics1_Point_data.ipynb` loads `Reduced_dendrites/`, converts units from nm to um, and obtains:
   `tree.get_root_coordinate()`.
3. Metrics1 contains no coordinate alignment, centering, JRC transform, or affine transform before `Root_x/y/z`.
4. PP4 performs a separate global alignment of a loaded forest using `align_forest`, sphere fitting, recentering and rotation. That notebook therefore cannot be the source of Metrics1 `Root_x/y/z`.
5. The historical `Point_data.pkl` is pinned independently at commit `56901ad1853b44aeca15504cd908fa4c31009a3e`, blob `b85caf49f45677f2075f7b5f2c8830141cd96d02`, SHA256 `76b7d6a1c44ad6b2ca730feff88174c71327e095cce45d8a47a0d998f77df58f`.

## Four target rows

| subtype | root ID | Point_data Root (um) | segments | cable (nm) |
|---|---:|---|---:|---:|
| T4a | 720575940632008007 | (-46.851207, 32.261150, -111.551992) | 220 | 315401.420 |
| T4c | 720575940616224414 | (-46.169293, 22.233229, -116.456664) | 207 | 299437.265 |
| T5a | 720575940625571465 | (11.981576, 50.189500, 120.402133) | 138 | 198019.153 |
| T5c | 720575940617782941 | (12.746071, 54.907805, 117.573273) | 148 | 194809.525 |

## Independent V229 PP3 reconstruction

Using the exact V229 SWCs and the pinned NeuRosetta PP3 algorithm, the selected roots are:

- T4a node 292; reduced: 213 nodes / 212 edges; cable 299110.429 nm
- T4c node 358; reduced: 212 nodes / 211 edges; cable 288451.472 nm
- T5a node 343; reduced: 181 nodes / 180 edges; cable 265200.242 nm
- T5c node 323; reduced: 153 nodes / 152 edges; cable 186566.680 nm

The selected node IDs independently reproduce V258.

## What this proves

- The historical Point_data coordinates are not produced by a later Metrics1 coordinate transform.
- The Point_data morphology metrics are computed from a NeuRosetta reduced-dendrite forest.
- V229 and the historical study share the same four FlyWire neuron IDs and the same PP3 algorithmic selection logic, but the V229 SWCs are **not yet proven bitwise-identical** to the study's hidden `.nr` trees.
- The metric differences (segment count and cable) independently demonstrate that V229 cannot currently be treated as an exact byte-for-byte substitute for the historical `.nr` morphology.
- Official FLYWIRE→JRC2018F registration was tested separately and does **not** reconcile the Point_data roots. Therefore JRC2018F is not an established explanation for the frame discrepancy.

## Current scientific boundary

The remaining missing artifact is the historical per-neuron `.nr`/SWC morphology snapshot (or an authoritative export of the reduced trees) used by the study before `Metrics1_Point_data.ipynb`. The public study Git repository contains the code and notebooks but not those `.nr` files.

The 2026 Zenodo data record contains the metric pickles, while the associated Zenodo software record contains the v2.0.0 source archive; neither public record has yet yielded the historical reduced `.nr` forest.

No arbitrary coordinate transform is accepted as a provenance solution.
