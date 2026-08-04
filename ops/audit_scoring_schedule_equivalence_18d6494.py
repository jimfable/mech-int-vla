#!/usr/bin/env python3
"""Fail-closed audit for serial versus two-worker Calibration sidecars.

This reads only already-published sidecars and the frozen feature source.  It
proves that scheduling changed no scientific primitive or feature dependency;
runtime/resource cost differences remain visible and separately classified.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

import mech_int_vla.feature_pipeline as feature_pipeline
from mech_int_vla.features import (
    M0_FEATURE_NAMES,
    M1_FEATURE_NAMES,
    M2_EXPERT_FEATURE_NAMES,
    M2_VLM_FEATURE_NAMES,
    M0Primitives,
    M1PoseState,
    M2Primitives,
)
from mech_int_vla.scoring import COST_FIELDS


SCHEMA_VERSION = 1
KIND = "calibration_scoring_schedule_equivalence_audit"
COST_ARRAYS = (
    "original_cost",
    "transformed_cost",
    "intervention_minus_cost",
    "intervention_plus_cost",
)
DYNAMIC_COST_FIELDS = frozenset(
    {
        "cuda_event_ms",
        "wall_time_ns",
        "peak_allocated_bytes",
        "incremental_peak_allocated_bytes",
    }
)
DETERMINISTIC_COST_FIELDS = frozenset(COST_FIELDS) - DYNAMIC_COST_FIELDS
FEATURE_FUNCTIONS = (
    "_prepare_states",
    "_state_hierarchy",
    "build_calibration_features",
    "build_locked_test_features",
)
EXPECTED_FEATURE_ARRAY_DEPENDENCIES = frozenset(
    {
        "control_step",
        "transform_available",
        "intervention_available",
        "original_actions",
        "transformed_actions",
        "intervention_minus_actions",
        "intervention_plus_actions",
        "original_activation",
        "transformed_activation",
    }
)
ASSIGNMENTS = {
    "libero_10-task5-calibration-init10-cell2": "worker0",
    "libero_10-task5-calibration-init10-cell4": "worker0",
    "libero_10-task5-calibration-init10-cell3": "worker1",
    "libero_10-task5-calibration-init10-cell7": "worker1",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain an object")
    return value


def _write_exclusive(path: Path, value: dict[str, Any]) -> str:
    payload = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite audit receipt {path}")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha256_bytes(payload)


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = f"{array.dtype.str}|{array.shape}|".encode("ascii")
    return _sha256_bytes(descriptor + array.tobytes(order="C"))


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise RuntimeError(f"duplicate arrays in {path}")
        return {name: archive[name].copy() for name in archive.files}


def _metadata_differences(
    first: Any, second: Any, path: tuple[str | int, ...] = ()
) -> list[dict[str, Any]]:
    if type(first) is not type(second):
        return [{"path": list(path), "serial": first, "two_worker": second}]
    if isinstance(first, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(set(first) | set(second)):
            if key not in first or key not in second:
                result.append(
                    {
                        "path": [*path, key],
                        "serial": first.get(key, "<MISSING>"),
                        "two_worker": second.get(key, "<MISSING>"),
                    }
                )
            else:
                result.extend(
                    _metadata_differences(first[key], second[key], (*path, key))
                )
        return result
    if isinstance(first, list):
        if len(first) != len(second):
            return [
                {
                    "path": [*path, "length"],
                    "serial": len(first),
                    "two_worker": len(second),
                }
            ]
        result = []
        for index, (left, right) in enumerate(zip(first, second, strict=True)):
            result.extend(_metadata_differences(left, right, (*path, index)))
        return result
    if first != second:
        return [{"path": list(path), "serial": first, "two_worker": second}]
    return []


def _feature_source_audit() -> dict[str, Any]:
    module_path = Path(inspect.getsourcefile(feature_pipeline) or "").resolve()
    if not module_path.is_file():
        raise RuntimeError("cannot resolve frozen feature_pipeline.py")
    module_tree = ast.parse(module_path.read_text(encoding="utf-8"))
    definitions = {
        node.name: node
        for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = sorted(set(FEATURE_FUNCTIONS) - set(definitions))
    if missing:
        raise RuntimeError(f"feature source is missing functions: {missing}")

    score_array_names = set(feature_pipeline.SCORE_ARRAY_NAMES)
    dependencies: set[str] = set()
    cost_literals: set[str] = set()
    function_sha256: dict[str, str] = {}
    source_text = module_path.read_text(encoding="utf-8")
    source_lines = source_text.splitlines(keepends=True)
    for name in FEATURE_FUNCTIONS:
        node = definitions[name]
        segment = "".join(source_lines[node.lineno - 1 : node.end_lineno])
        function_sha256[name] = _sha256_bytes(segment.encode("utf-8"))
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                if child.value in score_array_names:
                    dependencies.add(child.value)
                if "cost" in child.value.lower():
                    cost_literals.add(child.value)

    if dependencies != set(EXPECTED_FEATURE_ARRAY_DEPENDENCIES):
        raise RuntimeError(
            "frozen feature array dependencies changed: "
            f"observed={sorted(dependencies)}"
        )
    if cost_literals:
        raise RuntimeError(
            f"feature construction unexpectedly references cost strings: {sorted(cost_literals)}"
        )

    primitive_fields = {
        "M0Primitives": [item.name for item in fields(M0Primitives)],
        "M1PoseState": [item.name for item in fields(M1PoseState)],
        "M2Primitives": [item.name for item in fields(M2Primitives)],
    }
    if any(
        "cost" in field.lower()
        for names in primitive_fields.values()
        for field in names
    ):
        raise RuntimeError("a frozen feature primitive unexpectedly contains cost")
    feature_names = {
        "M0": list(M0_FEATURE_NAMES),
        "M1": list(M1_FEATURE_NAMES),
        "M2_vlm": list(M2_VLM_FEATURE_NAMES),
        "M2_expert": list(M2_EXPERT_FEATURE_NAMES),
    }
    forbidden = ("cost", "cuda", "wall_time", "memory", "bytes", "forward_count")
    offending_names = sorted(
        name
        for names in feature_names.values()
        for name in names
        if any(token in name.lower() for token in forbidden)
    )
    if offending_names:
        raise RuntimeError(f"predictor schema contains cost features: {offending_names}")
    return {
        "feature_pipeline_path": str(module_path),
        "feature_pipeline_sha256": _sha256_file(module_path),
        "audited_function_sha256": function_sha256,
        "score_array_dependencies": sorted(dependencies),
        "excluded_cost_arrays": list(COST_ARRAYS),
        "feature_primitive_fields": primitive_fields,
        "feature_names": feature_names,
        "cost_feature_names": offending_names,
        "cost_arrays_used_as_predictors": False,
    }


def _episode_audit(
    episode_id: str,
    worker: str,
    reference_root: Path,
    benchmark_root: Path,
) -> dict[str, Any]:
    serial_path = reference_root / episode_id
    two_worker_path = benchmark_root / worker / "scores" / "calibration" / episode_id
    for path in (serial_path, two_worker_path):
        if not path.is_dir() or path.is_symlink():
            raise RuntimeError(f"sidecar path is missing or unsafe: {path}")
        if {item.name for item in path.iterdir()} != {"metadata.json", "primitives.npz"}:
            raise RuntimeError(f"sidecar file inventory differs: {path}")

    serial_metadata = _strict_json(serial_path / "metadata.json")
    two_worker_metadata = _strict_json(two_worker_path / "metadata.json")
    metadata_differences = _metadata_differences(
        serial_metadata, two_worker_metadata
    )
    allowed_metadata_paths = [["files", "primitives_sha256"]]
    if [item["path"] for item in metadata_differences] != allowed_metadata_paths:
        raise RuntimeError(
            f"{episode_id}: non-cost metadata differs: {metadata_differences}"
        )
    normalized_serial = json.loads(json.dumps(serial_metadata))
    normalized_two_worker = json.loads(json.dumps(two_worker_metadata))
    normalized_serial["files"]["primitives_sha256"] = "<COST-NORMALIZED>"
    normalized_two_worker["files"]["primitives_sha256"] = "<COST-NORMALIZED>"
    if _canonical(normalized_serial) != _canonical(normalized_two_worker):
        raise RuntimeError(f"{episode_id}: normalized metadata is not identical")

    serial_arrays = _load_arrays(serial_path / "primitives.npz")
    two_worker_arrays = _load_arrays(two_worker_path / "primitives.npz")
    if set(serial_arrays) != set(two_worker_arrays):
        raise RuntimeError(f"{episode_id}: array inventories differ")
    array_results: dict[str, Any] = {}
    differing_arrays: list[str] = []
    for name in sorted(serial_arrays):
        left = serial_arrays[name]
        right = two_worker_arrays[name]
        same_contract = left.dtype == right.dtype and left.shape == right.shape
        byte_identical = same_contract and left.tobytes(order="C") == right.tobytes(
            order="C"
        )
        if not byte_identical:
            differing_arrays.append(name)
        array_results[name] = {
            "dtype": left.dtype.str,
            "shape": list(left.shape),
            "contract_identical": same_contract,
            "byte_identical": byte_identical,
            "serial_sha256": _array_digest(left),
            "two_worker_sha256": _array_digest(right),
        }
        if not same_contract:
            raise RuntimeError(f"{episode_id}: {name} dtype/shape differs")
        if name not in COST_ARRAYS and not byte_identical:
            raise RuntimeError(f"{episode_id}: scientific array {name} differs")
    if differing_arrays != sorted(COST_ARRAYS):
        raise RuntimeError(
            f"{episode_id}: differing arrays are not exactly the cost arrays: "
            f"{differing_arrays}"
        )

    cost_field_results: dict[str, Any] = {}
    for name in COST_ARRAYS:
        left = serial_arrays[name]
        right = two_worker_arrays[name]
        field_results: dict[str, Any] = {}
        for index, field in enumerate(COST_FIELDS):
            left_field = left[..., index]
            right_field = right[..., index]
            equal = bool(np.array_equal(left_field, right_field, equal_nan=True))
            same = (left_field == right_field) | (
                np.isnan(left_field) & np.isnan(right_field)
            )
            field_results[field] = {
                "identical": equal,
                "differing_elements": int(np.count_nonzero(~same)),
            }
            if field in DETERMINISTIC_COST_FIELDS and not equal:
                raise RuntimeError(
                    f"{episode_id}: deterministic cost field {name}.{field} differs"
                )
        if not any(
            not field_results[field]["identical"] for field in DYNAMIC_COST_FIELDS
        ):
            raise RuntimeError(f"{episode_id}: {name} has no dynamic cost difference")
        cost_field_results[name] = field_results

    return {
        "episode_id": episode_id,
        "serial_path": str(serial_path),
        "two_worker_path": str(two_worker_path),
        "serial_files": {
            name: _sha256_file(serial_path / name)
            for name in ("metadata.json", "primitives.npz")
        },
        "two_worker_files": {
            name: _sha256_file(two_worker_path / name)
            for name in ("metadata.json", "primitives.npz")
        },
        "metadata_differences": metadata_differences,
        "all_non_cost_metadata_identical": True,
        "differing_arrays": differing_arrays,
        "all_scientific_arrays_byte_identical": True,
        "arrays": array_results,
        "cost_fields": cost_field_results,
        "deterministic_cost_fields_identical": True,
        "cost_differences_confined_to_runtime_resource_fields": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_root = args.reference_root.resolve()
    benchmark_root = args.benchmark_root.resolve()
    output = args.output.resolve()
    if output == reference_root or reference_root in output.parents:
        raise RuntimeError("audit output overlaps authoritative serial sidecars")
    if output == benchmark_root or benchmark_root in output.parents:
        raise RuntimeError("audit output overlaps benchmark sidecars")

    episodes = [
        _episode_audit(episode_id, worker, reference_root, benchmark_root)
        for episode_id, worker in sorted(ASSIGNMENTS.items())
    ]
    feature_audit = _feature_source_audit()
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "pass",
        "episode_count": len(episodes),
        "assignments": ASSIGNMENTS,
        "cost_arrays": list(COST_ARRAYS),
        "dynamic_cost_fields": sorted(DYNAMIC_COST_FIELDS),
        "deterministic_cost_fields": sorted(DETERMINISTIC_COST_FIELDS),
        "episodes": episodes,
        "feature_source_audit": feature_audit,
        "all_non_cost_metadata_identical": True,
        "all_scientific_arrays_byte_identical": True,
        "differing_arrays_exactly_cost_arrays": True,
        "deterministic_cost_fields_identical": True,
        "cost_differences_confined_to_runtime_resource_fields": True,
        "cost_arrays_used_as_m0_m1_m2_predictors": False,
        "authoritative_sidecars_written": False,
        "benchmark_sidecars_written": False,
        "locked_test_accessed": False,
    }
    digest = _write_exclusive(output, receipt)
    print(json.dumps({"audit_sha256": digest, **receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
