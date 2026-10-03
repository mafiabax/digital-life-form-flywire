#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ROOTS = {
    "T4a": "720575940632008007",
    "T4c": "720575940616224414",
    "T5a": "720575940625571465",
    "T5c": "720575940617782941",
}

EXPECTED_V258_NODE = {
    "T4a": 292,
    "T4c": 358,
    "T5a": 343,
    "T5c": 323,
}

NEROROSSETTA = {
    "repository": "NikDrummond/NeuRosetta",
    "commit": "38f20f02194c129c234360db5a8be78a90c61db1",
    "subtrees_file": "src/NeuRosetta/ops/tree_graphs/subtrees.py",
    "subgraphs_file": "src/NeuRosetta/utils/graph_utils/subgraphs.py",
    "forest_file": "src/NeuRosetta/api/forest_class.py",
    "tree_editing_file": "src/NeuRosetta/ops/tree_graphs/tree_editing.py",
    "vertex_inds_file": "src/NeuRosetta/ops/tree_graphs/vertex_inds.py",
    "traversals_file": "src/NeuRosetta/utils/graph_utils/traversals.py",
}

STUDY = {
    "repository": "borstlab/T4_T5_Dendrite_Morphology_Paper",
    "pp3_commit": "3a1aa1a2e368ff8767f40791588eaf552e6d436d",
    "pp3_notebook": "Notebooks/PP3_Dendrite_extraction.ipynb",
    "point_data_commit": "56901ad1853b44aeca15504cd908fa4c31009a3e",
    "point_data_blob": "b85caf49f45677f2075f7b5f2c8830141cd96d02",
    "point_data_file": "Data/Point_data.pkl",
    "point_data_units": "um",
}

EXPECTED_POINT_SHA256 = "76b7d6a1c44ad6b2ca730feff88174c71327e095cce45d8a47a0d998f77df58f"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_point_data(path: Path) -> pd.DataFrame:
    obj = pd.read_pickle(path)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Point_data.pkl must contain a pandas DataFrame, got {type(obj).__name__}")
    data = obj.copy()
    normalized = {
        "".join(ch for ch in str(col).lower() if ch.isalnum()): col
        for col in data.columns
    }
    def pick(*aliases: str):
        for alias in aliases:
            key = "".join(ch for ch in alias.lower() if ch.isalnum())
            if key in normalized:
                return normalized[key]
        return None
    id_col = pick("ID", "Flywire_id", "FlyWire_id", "root_id")
    x_col = pick("Root_x", "root_x")
    y_col = pick("Root_y", "root_y")
    z_col = pick("Root_z", "root_z")
    missing = [
        name for name, col in
        (("ID", id_col), ("Root_x", x_col), ("Root_y", y_col), ("Root_z", z_col))
        if col is None
    ]
    if missing:
        raise RuntimeError(f"Point_data.pkl missing required fields {missing}")
    data["ID"] = data[id_col].astype(str)
    data["Root_x"] = pd.to_numeric(data[x_col], errors="raise")
    data["Root_y"] = pd.to_numeric(data[y_col], errors="raise")
    data["Root_z"] = pd.to_numeric(data[z_col], errors="raise")
    return data

def anchor_point_rows(data: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for subtype, root_id in ROOTS.items():
        rows = data[data["ID"] == root_id]
        if len(rows) != 1:
            raise RuntimeError(
                f"Point_data anchor regression failed for {subtype}: expected 1 row, got {len(rows)}"
            )
        row = rows.iloc[0]
        out[subtype] = {
            "root_x_um": float(row["Root_x"]),
            "root_y_um": float(row["Root_y"]),
            "root_z_um": float(row["Root_z"]),
        }
    return out

def parse_swc(path: Path) -> dict[str, Any]:
    nodes: dict[int, dict[str, float | int]] = {}
    children: dict[int, list[int]] = {}
    order: list[int] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                raise RuntimeError(f"Malformed SWC row in {path}")
            node_id = int(parts[0])
            if node_id in nodes:
                raise RuntimeError(f"Duplicate SWC node {node_id} in {path}")
            parent = int(parts[6])
            nodes[node_id] = {
                "id": node_id,
                "x": float(parts[2]),
                "y": float(parts[3]),
                "z": float(parts[4]),
                "radius": float(parts[5]),
                "parent": parent,
                "label": int(parts[1]),
            }
            if parent != -1:
                children.setdefault(parent, []).append(node_id)
    roots = [int(n["id"]) for n in nodes.values() if int(n["parent"]) == -1]
    if len(roots) != 1:
        raise RuntimeError(f"{path.name}: expected one structural root, got {roots}")
    for n in nodes.values():
        parent = int(n["parent"])
        if parent != -1 and parent not in nodes:
            raise RuntimeError(f"{path.name}: missing parent {parent}")
    root = roots[0]
    stack = [root]
    while stack:
        cur = stack.pop()
        order.append(cur)
        stack.extend(reversed(children.get(cur, [])))
    if len(order) != len(nodes):
        raise RuntimeError(f"{path.name}: graph is disconnected from root")
    edge_length: dict[tuple[int, int], float] = {}
    for n in nodes.values():
        parent = int(n["parent"])
        if parent == -1:
            continue
        p = nodes[parent]
        edge_length[(parent, int(n["id"]))] = math.dist(
            (float(p["x"]), float(p["y"]), float(p["z"])),
            (float(n["x"]), float(n["y"]), float(n["z"])),
        )
    return {
        "nodes": nodes,
        "children": children,
        "root": root,
        "order": order,
        "edge_length": edge_length,
    }

def select_pp3_subtree(tree: dict[str, Any]) -> dict[str, Any]:
    nodes = tree["nodes"]
    children = tree["children"]
    order = tree["order"]
    edge_length = tree["edge_length"]
    leaves = {node_id for node_id in nodes if len(children.get(node_id, [])) == 0}
    total_leaves = len(leaves)
    subtree_cable = {node_id: 0.0 for node_id in nodes}
    subtree_leaves = {
        node_id: 1 if node_id in leaves else 0
        for node_id in nodes
    }
    total_cable = sum(float(length) for length in edge_length.values())
    for current in reversed(order):
        for child in children.get(current, []):
            subtree_cable[current] += edge_length[(current, child)] + subtree_cable[child]
            subtree_leaves[current] += subtree_leaves[child]
    candidates = []
    for node_id in order:
        if len(children.get(node_id, [])) < 2:
            continue
        score = 1.0 - subtree_cable[node_id] / total_cable + subtree_leaves[node_id] / total_leaves
        candidates.append({
            "id": int(node_id),
            "score": float(score),
            "subtree_cable_nm": float(subtree_cable[node_id]),
            "subtree_leaves": int(subtree_leaves[node_id]),
        })
    if not candidates:
        raise RuntimeError("PP3 subtree selection found no branch candidates")
    candidates.sort(key=lambda item: (-item["score"], item["id"]))
    best = candidates[0]
    members: set[int] = set()
    stack = [best["id"]]
    while stack:
        current = stack.pop()
        if current in members:
            continue
        members.add(current)
        stack.extend(children.get(current, []))
    return {
        "selected_root": int(best["id"]),
        "selected_score": float(best["score"]),
        "second_score": float(candidates[1]["score"]) if len(candidates) > 1 else None,
        "score_margin": float(candidates[0]["score"] - candidates[1]["score"]) if len(candidates) > 1 else None,
        "total_cable_nm": float(total_cable),
        "global_leaf_count": int(total_leaves),
        "subtree_cable_nm": float(best["subtree_cable_nm"]),
        "subtree_leaf_count": int(best["subtree_leaves"]),
        "subtree_node_count": int(len(members)),
        "members": members,
    }

def reduce_pp3_tree(tree: dict[str, Any], selected_root: int, members: set[int]) -> dict[str, Any]:
    children = tree["children"]
    edge_length = tree["edge_length"]
    restricted_children = {
        node_id: [child for child in children.get(node_id, []) if child in members]
        for node_id in members
    }
    starts = {node_id for node_id in members if len(restricted_children.get(node_id, [])) > 1}
    if selected_root not in starts:
        starts.add(selected_root)
    leaves = {node_id for node_id in members if len(restricted_children.get(node_id, [])) == 0}
    stops = leaves | (starts - {selected_root})
    reduced_edges = []
    for stop in sorted(stops):
        current = stop
        path_length = 0.0
        while True:
            parent = int(tree["nodes"][current]["parent"])
            if parent == -1:
                raise RuntimeError(f"Reduction path reached structural root unexpectedly for stop {stop}")
            path_length += edge_length[(parent, current)]
            current = parent
            if current in starts:
                break
        reduced_edges.append((int(current), int(stop), float(path_length)))
    root_coord = (
        float(tree["nodes"][selected_root]["x"]),
        float(tree["nodes"][selected_root]["y"]),
        float(tree["nodes"][selected_root]["z"]),
    )
    reduced_cable = sum(edge[2] for edge in reduced_edges)
    expected_edge_count = len(leaves) + len(starts) - 1
    expected_node_count = len(reduced_edges) + 1
    return {
        "start_node_count_including_root": int(len(starts)),
        "leaf_count": int(len(leaves)),
        "reduced_node_count": int(expected_node_count),
        "reduced_edge_count": int(len(reduced_edges)),
        "expected_reduced_edge_count": int(expected_edge_count),
        "reduced_cable_nm": float(reduced_cable),
        "root_node_id": int(selected_root),
        "root_coordinate_nm": root_coord,
    }

def reconstruct(point_data: Path, morphology_dir: Path, v258_json: Path) -> dict[str, Any]:
    point_sha = sha256_file(point_data)
    if point_sha != EXPECTED_POINT_SHA256:
        raise RuntimeError(f"Point_data SHA-256 mismatch: expected {EXPECTED_POINT_SHA256}, got {point_sha}")
    point_rows = anchor_point_rows(load_point_data(point_data))
    v258 = json.loads(v258_json.read_text(encoding="utf-8"))
    records = []
    for subtype, root_id in ROOTS.items():
        swc_path = morphology_dir / f"{subtype}_{root_id}.swc"
        if not swc_path.exists():
            raise RuntimeError(f"Missing exact V229 SWC: {swc_path}")
        tree = parse_swc(swc_path)
        selection = select_pp3_subtree(tree)
        reduction = reduce_pp3_tree(tree, selection["selected_root"], selection["members"])
        expected_node = EXPECTED_V258_NODE[subtype]
        checked_node = int(v258["per_root"][subtype]["selected_algorithm_node_id"])
        if selection["selected_root"] != expected_node:
            raise RuntimeError(f"{subtype}: independent PP3 selection mismatch: expected {expected_node}, got {selection['selected_root']}")
        if checked_node != expected_node:
            raise RuntimeError(f"{subtype}: checked-in V258 selected node mismatch: expected {expected_node}, got {checked_node}")
        if reduction["expected_reduced_edge_count"] != reduction["reduced_edge_count"]:
            raise RuntimeError(f"{subtype}: reduced edge-count invariant failed")
        if not math.isclose(reduction["reduced_cable_nm"], selection["subtree_cable_nm"], rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(f"{subtype}: reduced cable is not conserved")
        selected_um = tuple(v / 1000.0 for v in reduction["root_coordinate_nm"])
        point_um = (
            point_rows[subtype]["root_x_um"],
            point_rows[subtype]["root_y_um"],
            point_rows[subtype]["root_z_um"],
        )
        residual_um = tuple(selected_um[i] - point_um[i] for i in range(3))
        residual_norm_um = math.dist(selected_um, point_um)
        previous_v261_nodes = reduction["leaf_count"] + reduction["start_node_count_including_root"] + 1
        previous_v261_edges = reduction["leaf_count"] + reduction["start_node_count_including_root"]
        records.append({
            "subtype": subtype,
            "root_id": root_id,
            "pp3_selected_root_node_id": selection["selected_root"],
            "pp3_selected_score": selection["selected_score"],
            "pp3_second_score": selection["second_score"],
            "pp3_score_margin": selection["score_margin"],
            "v229_full_node_count": len(tree["nodes"]),
            "v229_global_leaf_count": selection["global_leaf_count"],
            "pp3_subtree_node_count": selection["subtree_node_count"],
            "pp3_subtree_leaf_count": selection["subtree_leaf_count"],
            "pp3_subtree_branch_count_including_root": reduction["start_node_count_including_root"],
            "pp3_reduced_node_count": reduction["reduced_node_count"],
            "pp3_reduced_edge_count": reduction["reduced_edge_count"],
            "pp3_reduced_cable_nm": reduction["reduced_cable_nm"],
            "pp3_reduced_root_x_nm": reduction["root_coordinate_nm"][0],
            "pp3_reduced_root_y_nm": reduction["root_coordinate_nm"][1],
            "pp3_reduced_root_z_nm": reduction["root_coordinate_nm"][2],
            "pp3_reduced_root_x_um": selected_um[0],
            "pp3_reduced_root_y_um": selected_um[1],
            "pp3_reduced_root_z_um": selected_um[2],
            "point_data_root_x_um": point_um[0],
            "point_data_root_y_um": point_um[1],
            "point_data_root_z_um": point_um[2],
            "residual_x_um": residual_um[0],
            "residual_y_um": residual_um[1],
            "residual_z_um": residual_um[2],
            "residual_norm_um": residual_norm_um,
            "coordinate_comparison_status": "NOT_COMPARABLE_WITHOUT_FRAME_RECONCILIATION",
            "v258_selected_node_matches_independent_reconstruction": checked_node == selection["selected_root"],
            "v261_previous_reduced_node_formula": previous_v261_nodes,
            "v261_previous_reduced_edge_formula": previous_v261_edges,
            "v261_off_by_one_correction_nodes": reduction["reduced_node_count"] - previous_v261_nodes,
            "v261_off_by_one_correction_edges": reduction["reduced_edge_count"] - previous_v261_edges,
        })
    return {
        "schema_version": 1,
        "status": "PASS_PP3_ALGORITHMIC_RECONSTRUCTION_WITH_FRAME_SEPARATION",
        "path": [
            "PP3_Dendrite_extraction.ipynb: forest.convert_forest_to_subtrees(max_workers=10)",
            "PP3_Dendrite_extraction.ipynb: forest.save_forest(...)",
            "PP3_Dendrite_extraction.ipynb: forest.reduce_forest(parallel=True, max_workers=10, inplace=True)",
            "PP3_Dendrite_extraction.ipynb: forest.save_forest(...)",
        ],
        "algorithm_sources": {"study": STUDY, "nerosetta": NEROROSSETTA,
            "subtree_score": "(1 - subtree_cable / total_cable) + (subtree_leaves / total_leaves)",
            "branch_definition": "out_degree > 1",
            "reduction_stops": "leaves + non-root branches",
            "reduction_starts": "all branches + selected root"},
        "point_data_source": {"sha256": point_sha, "verified_expected_sha256": point_sha == EXPECTED_POINT_SHA256, **STUDY},
        "scientific_boundary": {
            "coordinate_frames": ["STUDY_NATIVE_FLYWIRE_PIPELINE", "V229_PROJECT_SWC_NATIVE"],
            "raw_coordinate_residuals_are_diagnostic_only": True,
            "published_to_v229_coordinate_identity": "UNRESOLVED",
            "manual_dendrite_curation_identity": "UNRESOLVED",
            "individual_synapse_biological_compartment": "UNRESOLVED",
            "exact_historical_flag_state_of_v229_swc": "UNRESOLVED"},
        "correction": {
            "issue": "V261 reduced-node/edge arithmetic double-counted the selected root because branch_indices() includes the root while core_indices(..., include_root=False) excludes it from reduction stops.",
            "correct_formula_with_branch_count_including_root": {"reduced_nodes": "leaves + branches", "reduced_edges": "leaves + branches - 1"}},
        "records": records,
        "not_proven": [
            "Bitwise identity between project V229 SWCs and the study's internal .nr forests.",
            "Recovery of the study per-tree flag metadata for the exact V229 SWCs.",
            "Recovery of manual dendrite corrections.",
            "A deterministic transform establishing a common coordinate frame between Point_data and V229.",
            "Individual synaptic-cleft biological compartment identity.",
        ],
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--point-data", type=Path, required=True)
    ap.add_argument("--morphology-dir", type=Path, required=True)
    ap.add_argument("--v258-json", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    result = reconstruct(args.point_data, args.morphology_dir, args.v258_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "V262_PP3_path_reconstruction.csv").write_text(
        pd.DataFrame(result["records"]).to_csv(index=False), encoding="utf-8")
    (args.output_dir / "V262_PP3_path_reconstruction.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "records": len(result["records"]), "point_data_sha256": result["point_data_source"]["sha256"]}, ensure_ascii=False))

if __name__ == "__main__":
    main()
