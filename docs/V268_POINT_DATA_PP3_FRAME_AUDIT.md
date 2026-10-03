# V268 — Historical Point_data + PP3 coordinate-frame audit

## Purpose

V268 resolves a provenance ambiguity introduced by V264 by using the exact historical Point_data.pkl blob from the pinned study commit instead of the regenerated Zenodo payload.

Historical source:
- Repository: borstlab/T4_T5_Dendrite_Morphology_Paper
- Commit: 56901ad1853b44aeca15504cd908fa4c31009a3e
- File: Data/Point_data.pkl
- Git blob: b85caf49f45677f2075f7b5f2c8830141cd96d02
- SHA-256: 76b7d6a1c44ad6b2ca730feff88174c71327e095cce45d8a47a0d998f77df58f

## Four exact historical anchors

| subtype | root_id | historical table index | Dendrite_used |
|---|---:|---:|---|
| T4a | 720575940632008007 | 514 | True |
| T4c | 720575940616224414 | 2227 | True |
| T5a | 720575940625571465 | 3171 | True |
| T5c | 720575940617782941 | 4651 | True |

## PP3 reconstruction

The study PP3 notebook performs load -> flagged-tree filtering -> convert_forest_to_subtrees() -> save -> reduce_forest(inplace=True) -> save.

The subtree score is:

(1 - subtree_cable / total_cable) + (subtree_leaves / total_leaves)

Independent reconstruction on the four V229 SWCs selects the same branch/root candidates recorded by V258:

| subtype | selected SWC node |
|---|---:|
| T4a | 292 |
| T4c | 358 |
| T5a | 343 |
| T5c | 323 |

NeuRosetta reduction copies coordinates from the original selected vertices, so reduction itself does not add a coordinate translation.

## Coordinate result

PP3-selected V229 roots after nm -> um conversion:

- T4a: (789.1497, 263.18266, 210.35522)
- T4c: (788.7438, 272.94928, 205.01356)
- T5a: (717.25825, 222.00586, 211.70161)
- T5c: (713.4343, 216.06416, 213.15164)

Historical Point_data roots:

- T4a: (-46.85120703125, 32.261150390625, -111.5519921875)
- T4c: (-46.16929296875, 22.233228515625, -116.4566640625)
- T5a: (11.981576171875, 50.1895, 120.4021328125)
- T5c: (12.7460712890625, 54.907804687500004, 117.5732734375)

Per-neuron PP3-selected residual magnitudes:

| subtype | residual (um) |
|---|---:|
| T4a | 925.1197 |
| T4c | 929.1294 |
| T5a | 731.6226 |
| T5c | 725.3072 |

These are not exact coordinate correspondences.

## Frame diagnostics

A centroid translation does not reconcile the four pairs. A single rigid transform leaves non-zero residuals of about 72–81 um. A similarity fit reduces the residuals to about 9–11 um but requires a fitted scale of about 2.7, so it is not an established biological coordinate conversion.

Therefore:

- neuron identity correspondence: established
- exact coordinate-frame identity: not established
- historical morphology bitwise identity: not established
- exact transform between the historical morphology and V229: unresolved

The leading unresolved cause is a difference between the historical morphology representation/snapshot used to produce the paper .nr forest and the V229 morphology snapshot. This remains a hypothesis until the historical .nr/SWC bytes are recovered.

## Where PP4 fits

Metrics1_Point_data.ipynb loads Reduced_dendrites and obtains the published root coordinates with tree.get_root_coordinate() after converting units from nm to um.

The explicit global alignment operations are in the later PP4_Global_allignment.ipynb. PP4 therefore does not explain the coordinates stored in Point_data.

PP1_Fetch_flywire.ipynb fetches FlyWire meshes and writes SWCs. PP2_nr_conversion.ipynb imports those SWCs into NeuRosetta and only adds metadata/units before saving .nr files.

## V264 correction

V264 switched the download source to the August 10, 2026 Zenodo release. That release contains a different point_data.pkl payload and different ID values, such as T4a ending in 8100 instead of the historical 8007. Those bytes are not the exact historical Point_data blob used by the pinned GitHub provenance path.

V268 preserves that mismatch as evidence and restores a fail-closed path for the historical bytes.

## Reproduction

- scripts/v268_historical_point_data_provenance.py
- .github/workflows/v268-point-data-pp3-frame-audit.yml

The workflow fails when the pinned historical Point_data SHA-256 or Git blob does not match.