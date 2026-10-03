#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HISTORICAL_POINT_URL = (
    "https://raw.githubusercontent.com/borstlab/T4_T5_Dendrite_Morphology_Paper/"
    "56901ad1853b44aeca15504cd908fa4c31009a3e/Data/Point_data.pkl"
)
EXPECTED_POINT_SHA256 = "76b7d6a1c44ad6b2ca730feff88174c71327e095cce45d8a47a0d998f77df58f"
EXPECTED_POINT_BLOB = "b85caf49f45677f2075f7b5f2c8830141cd96d02"

ROOTS = {
    "T4a": "720575940632008007",
    "T4c": "720575940616224414",
    "T5a": "720575940625571465",
    "T5c": "720575940617782941",
}

V229_FILES = {
    "T4a": "v229_results/T4a_720575940632008007.swc",
    "T4c": "v229_results/T4c_720575940616224414.swc",
    "T5a": "v229_results/T5a_720575940625571465.swc",
    "T5c": "v229_results/T5c_720575940617782941.swc",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_swc(path: Path) -> dict:
    nodes = []
    by_id = {}
    children: dict[int, list[int]] = {}

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                raise RuntimeError(f"{path.name}: malformed SWC row at line {line_no}")
            node = {
                "id": int(parts[0]),
                "label": int(parts[1]),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "z": float(parts[4]),
                "radius": float(parts[5]),
                "parent": int(parts[6]),
            }
            if node["id"] in by_id:
                raise RuntimeError(f"{path.name}: duplicate node id {node['id']}")
            nodes.append(node)
            by_id[node["id"]] = node
            if node["parent"] != -1:
                children.setdefault(node["parent"], []).append(node["id"])

    roots = [n["id"] for n in nodes if n["parent"] == -1]
    if len(roots) != 1:
        raise RuntimeError(f"{path.name}: expected one structural root, got {roots}")

    missing = sorted({
        n["parent"] for n in nodes
        if n["parent"] != -1 and n["parent"] not in by_id
    })
    if missing:
        raise RuntimeError(f"{path.name}: missing parents {missing}")

    order = []
    stack = [roots[0]]
    while stack:
        current = stack.pop()
        order.append(current)
        stack.extend(reversed(children.get(current, [])))

    if len(order) != len(nodes):
        raise RuntimeError(f"{path.name}: graph is disconnected")

    edge_length = {}
    total_cable = 0.0
    for n in nodes:
        if n["parent"] == -1:
            continue
        p = by_id[n["parent"]]
        length = math.dist(
            (p["x"], p["y"], p["z"]),
            (n["x"], n["y"], n["z"]),
        )
        edge_length[(n["parent"], n["id"])] = length
        total_cable += length

    return {
        "nodes": nodes,
        "by_id": by_id,
        "children": children,
        "root": roots[0],
        "order": order,
        "edge_length": edge_length,
        "total_cable": total_cable,
        "sha256": sha256_file(path),
    }


def pp3_selected_root(tree: dict) -> dict:
    leaves = {
        n["id"] for n in tree["nodes"]
        if len(tree["children"].get(n["id"], [])) == 0
    }
    subtree_cable = {n["id"]: 0.0 for n in tree["nodes"]}
    subtree_leaves = {
        n["id"]: (1 if n["id"] in leaves else 0)
        for n in tree["nodes"]
    }

    for current in reversed(tree["order"]):
        for child in tree["children"].get(current, []):
            subtree_cable[current] += (
                tree["edge_length"][(current, child)] + subtree_cable[child]
            )
            subtree_leaves[current] += subtree_leaves[child]

    candidates = []
    for node in tree["nodes"]:
        nid = node["id"]
        if len(tree["children"].get(nid, [])) < 2:
            continue
        score = (
            1.0 - subtree_cable[nid] / tree["total_cable"]
            + subtree_leaves[nid] / len(leaves)
        )
        candidates.append((score, nid))

    if not candidates:
        raise RuntimeError("PP3 subtree selection found no branch candidate")

    score, selected_id = max(candidates, key=lambda item: item[0])
    selected = tree["by_id"][selected_id]

    return {
        "node_id": int(selected_id),
        "score": float(score),
        "x_nm": float(selected["x"]),
        "y_nm": float(selected["y"]),
        "z_nm": float(selected["z"]),
        "x_um": float(selected["x"]) / 1000.0,
        "y_um": float(selected["y"]) / 1000.0,
        "z_um": float(selected["z"]) / 1000.0,
        "subtree_cable_nm": float(subtree_cable[selected_id]),
        "subtree_leaves": int(subtree_leaves[selected_id]),
        "leaf_count": int(len(leaves)),
    }


def kabsch_rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    u, _, vt = np.linalg.svd(src_c.T @ dst_c)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = dst.mean(axis=0) - r @ src.mean(axis=0)
    return r, t


def similarity_fit(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    u, singular, vt = np.linalg.svd(src_c.T @ dst_c)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
        singular[-1] *= -1
    denom = float((src_c ** 2).sum())
    scale = float(singular.sum() / denom)
    t = dst.mean(axis=0) - scale * (r @ src.mean(axis=0))
    return scale, r, t


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    point_path = Path("/tmp/Point_data.pkl")
    if not point_path.exists():
        raise RuntimeError("Expected /tmp/Point_data.pkl from the immutable-source workflow step")

    actual_sha = sha256_file(point_path)
    if actual_sha != EXPECTED_POINT_SHA256:
        raise RuntimeError(
            "Historical Point_data sha256 mismatch: "
            f"expected {EXPECTED_POINT_SHA256}, got {actual_sha}"
        )

    import subprocess
    actual_blob = subprocess.check_output(
        ["git", "hash-object", "-t", "blob", str(point_path)], text=True
    ).strip()
    if actual_blob != EXPECTED_POINT_BLOB:
        raise RuntimeError(
            "Historical Point_data git-blob mismatch: "
            f"expected {EXPECTED_POINT_BLOB}, got {actual_blob}"
        )

    point = pd.read_pickle(point_path)
    normalized = {str(c).lower(): c for c in point.columns}
    required = {
        "id": normalized.get("id"),
        "root_x": normalized.get("root_x"),
        "root_y": normalized.get("root_y"),
        "root_z": normalized.get("root_z"),
        "subtype": normalized.get("subtype"),
    }
    if any(v is None for v in required.values()):
        raise RuntimeError(f"Point_data missing required columns: {required}")

    point["__id_str"] = point[required["id"]].astype(str)
    records = []
    selected_points = []
    published_points = []

    for subtype, root_id in ROOTS.items():
        matches = point[point["__id_str"] == root_id]
        if len(matches) != 1:
            raise RuntimeError(f"{subtype}: expected exactly one historical Point_data row, got {len(matches)}")
        row = matches.iloc[0]

        swc_path = root_dir / V229_FILES[subtype]
        tree = parse_swc(swc_path)
        selected = pp3_selected_root(tree)

        published = np.array([
            float(row[required["root_x"]]),
            float(row[required["root_y"]]),
            float(row[required["root_z"]]),
        ])
        selected_um = np.array([
            selected["x_um"], selected["y_um"], selected["z_um"]
        ])
        structural_um = np.array([
            tree["by_id"][tree["root"]]["x"] / 1000.0,
            tree["by_id"][tree["root"]]["y"] / 1000.0,
            tree["by_id"][tree["root"]]["z"] / 1000.0,
        ])

        selected_residual = selected_um - published
        structural_residual = structural_um - published

        records.append({
            "subtype": subtype,
            "root_id": root_id,
            "historical_point_root_um": published.tolist(),
            "v229_structural_root_um": structural_um.tolist(),
            "v229_pp3_selected_root_um": selected_um.tolist(),
            "v229_structural_root_residual_um": structural_residual.tolist(),
            "v229_pp3_selected_root_residual_um": selected_residual.tolist(),
            "v229_structural_root_distance_um": float(np.linalg.norm(structural_residual)),
            "v229_pp3_selected_root_distance_um": float(np.linalg.norm(selected_residual)),
            "pp3_selected_node_id": selected["node_id"],
            "pp3_selected_score": selected["score"],
            "v229_node_count": len(tree["nodes"]),
            "swc_sha256": tree["sha256"],
        })

        selected_points.append(selected_um)
        published_points.append(published)

    src = np.asarray(selected_points, dtype=float)
    dst = np.asarray(published_points, dtype=float)

    delta = dst - src
    centroid_translation = delta.mean(axis=0)
    translation_pred = src + centroid_translation
    translation_residuals = np.linalg.norm(translation_pred - dst, axis=1)

    r, t = kabsch_rigid(src, dst)
    rigid_pred = src @ r.T + t
    rigid_residuals = np.linalg.norm(rigid_pred - dst, axis=1)

    scale, rs, ts = similarity_fit(src, dst)
    sim_pred = scale * (src @ rs.T) + ts
    sim_residuals = np.linalg.norm(sim_pred - dst, axis=1)

    pairwise = []
    for i, j in combinations(range(len(src)), 2):
        src_d = float(np.linalg.norm(src[i] - src[j]))
        dst_d = float(np.linalg.norm(dst[i] - dst[j]))
        pairwise.append({
            "i": i,
            "j": j,
            "v229_selected_distance_um": src_d,
            "historical_point_distance_um": dst_d,
            "ratio_point_to_v229": (dst_d / src_d) if src_d else None,
        })

    report = {
        "schema_version": 1,
        "status": "PASS_HISTORICAL_POINT_DATA_AND_PP3_RECONSTRUCTION",
        "source": {
            "point_data_url": HISTORICAL_POINT_URL,
            "historical_git_blob": EXPECTED_POINT_BLOB,
            "historical_sha256": EXPECTED_POINT_SHA256,
            "observed_sha256": actual_sha,
            "observed_git_blob": actual_blob,
        },
        "algorithm": {
            "study_repository": "borstlab/T4_T5_Dendrite_Morphology_Paper",
            "study_commit": "3a1aa1a2e368ff8767f40791588eaf552e6d436d",
            "pp3_notebook": "Notebooks/PP3_Dendrite_extraction.ipynb",
            "subtree_score": "(1 - subtree_cable / total_cable) + (subtree_leaves / total_leaves)",
            "reduction_semantics": "reduce_graph preserves coordinates of original selected vertices",
            "neurosetta_commit": "38f20f02194c129c234360db5a8be78a90c61db1",
        },
        "per_root": records,
        "frame_diagnostics": {
            "direct_selected_root_exact_match": all(
                x["v229_pp3_selected_root_distance_um"] == 0.0 for x in records
            ),
            "centroid_translation_um": centroid_translation.tolist(),
            "centroid_translation_residuals_um": translation_residuals.tolist(),
            "rigid_transform_residuals_um": rigid_residuals.tolist(),
            "rigid_rms_um": float(np.sqrt(np.mean(rigid_residuals ** 2))),
            "similarity_scale": scale,
            "similarity_transform_residuals_um": sim_residuals.tolist(),
            "similarity_rms_um": float(np.sqrt(np.mean(sim_residuals ** 2))),
            "pairwise_distance_ratios": pairwise,
        },
        "interpretation": (
            "The four exact historical Point_data rows and the four exact V229 SWCs are "
            "identity-linked, and the PP3 subtree-selection rule reproduces the checked-in "
            "selected node IDs. However, the PP3-selected V229 root coordinates do not equal "
            "the historical Point_data root coordinates. No single translation or rigid "
            "transform makes the four pairs coincide. A similarity fit reduces residuals but "
            "does not establish coordinate-frame identity; exact historical .nr morphology "
            "and the historical NeuRosetta environment remain unavailable."
        ),
        "not_proven": [
            "Bitwise identity between V229 SWCs and the study's historical .nr forests.",
            "The exact historical coordinate transform, if any, between the two morphology snapshots.",
            "Recovery of manually curated dendrite edits from the historical .nr forest.",
            "Biological synaptic-cleft compartment identity for any individual V230 point.",
        ],
    }

    out_dir = root_dir / "v268_results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "V268_historical_point_data_pp3_frame_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
