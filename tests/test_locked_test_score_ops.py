from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mech_int_vla.config import SplitName, load_protocol_config

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_protocol_config(ROOT / "configs")
SPEC = importlib.util.spec_from_file_location(
    "locked_test_score_ops_under_test", ROOT / "ops" / "locked_test_score.py"
)
assert SPEC is not None and SPEC.loader is not None
locked_test_score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(locked_test_score)


def _source(split: SplitName, *, manifest_sha256: str, ids: tuple[str, ...]):
    return SimpleNamespace(
        split=split,
        task=PROTOCOL.task_order.tasks[0],
        policy_revision="policy",
        base_vlm_revision="base",
        manifest_sha256=manifest_sha256,
        valid_episode_ids=ids,
    )


def test_calibration_probe_is_accepted_across_manifest_and_episode_sets() -> None:
    calibration = _source(
        SplitName.CALIBRATION,
        manifest_sha256="a" * 64,
        ids=("calibration-a", "calibration-b"),
    )
    locked = _source(
        SplitName.LOCKED_TEST,
        manifest_sha256="b" * 64,
        ids=("locked-a", "locked-b"),
    )

    # These sets must differ by construction.  The old implementation rejected
    # the valid Calibration-bound probe by comparing both against Locked Test.
    assert calibration.manifest_sha256 != locked.manifest_sha256
    assert calibration.valid_episode_ids != locked.valid_episode_ids
    locked_test_score._validate_locked_probe_compatibility(
        SimpleNamespace(rollout=SimpleNamespace(source=calibration)),
        SimpleNamespace(source=locked),
    )


def test_cross_split_probe_still_fails_closed_on_model_source_drift() -> None:
    calibration = _source(
        SplitName.CALIBRATION, manifest_sha256="a" * 64, ids=("calibration",)
    )
    locked = _source(SplitName.LOCKED_TEST, manifest_sha256="b" * 64, ids=("locked",))
    locked.policy_revision = "stale-policy"

    with pytest.raises(RuntimeError, match="differ from bound Calibration"):
        locked_test_score._validate_locked_probe_compatibility(
            SimpleNamespace(rollout=SimpleNamespace(source=calibration)),
            SimpleNamespace(source=locked),
        )


def test_resumed_sidecar_fails_closed_on_stale_config_link(monkeypatch) -> None:
    sha = "a" * 64
    monkeypatch.setattr(
        locked_test_score,
        "artifact_identity_from_rollout",
        lambda artifact: SimpleNamespace(),
    )
    monkeypatch.setattr(
        locked_test_score,
        "content_links_for",
        lambda *args, **kwargs: SimpleNamespace(
            raw_metadata_sha256=sha,
            raw_trajectory_sha256=sha,
            probe_sha256=sha,
            config_sha256=sha,
            code_sha256=sha,
        ),
    )
    sidecar = SimpleNamespace(
        metadata={
            "split": "locked_test",
            "links": {
                "raw_metadata_sha256": sha,
                "raw_trajectory_sha256": sha,
                "probe_sha256": sha,
                "config_sha256": "b" * 64,
                "code_sha256": sha,
            },
        }
    )

    with pytest.raises(RuntimeError, match="stale source links"):
        locked_test_score._validate_resumed_sidecar(
            sidecar,
            episode_id="synthetic",
            artifact=SimpleNamespace(),
            bound=SimpleNamespace(),
            root=ROOT,
            protocol=PROTOCOL,
        )


def _invalid_allocation(invalid_ids: tuple[str, ...]):
    episodes = tuple(
        SimpleNamespace(
            episode_id=f"episode-{index:03d}",
            condition_name=f"cell-{index // 20}",
            condition_index=index // 20,
        )
        for index in range(160)
    )
    return SimpleNamespace(
        invalid_episode_ids=invalid_ids,
        invalid_fraction=len(invalid_ids) / len(episodes),
        manifest=SimpleNamespace(episodes=episodes),
    )


def test_invalid_resets_are_excluded_within_frozen_allocation_cap() -> None:
    allocation = _invalid_allocation(("episode-000", "episode-001"))

    locked_test_score._validate_invalid_allocation(
        allocation, max_invalid_fraction=0.10
    )


def test_invalid_reset_cap_is_checked_per_condition_cell() -> None:
    # Three among one condition's 20 episodes exceeds its 10% cap, even though
    # the global invalid fraction remains far below 10%.
    allocation = _invalid_allocation(("episode-000", "episode-001", "episode-002"))

    with pytest.raises(RuntimeError, match="in cell"):
        locked_test_score._validate_invalid_allocation(
            allocation, max_invalid_fraction=0.10
        )


class _LabelTrap:
    @property
    def terminal_failure_label(self):
        raise AssertionError("Locked Test label was read during prediction")


class _FrozenPredictors:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def model(self, name: str):
        return SimpleNamespace(feature_names=(f"{name}-feature",))

    def predict_proba(self, name: str, values: np.ndarray) -> np.ndarray:
        self.calls.append(name)
        return np.full(values.shape[0], {"M0": 0.2, "M1": 0.3, "M2": 0.4}[name])


def test_frozen_prediction_does_not_read_or_fit_locked_labels() -> None:
    cohort = SimpleNamespace(
        records=[_LabelTrap(), _LabelTrap()],
        m0_matrix=np.asarray([[1.0], [2.0]]),
        m1_matrix=np.asarray([[3.0], [4.0]]),
        m2_matrix=np.asarray([[5.0], [6.0]]),
        m0_names=("M0-feature",),
        m1_names=("M1-feature",),
        m2_names=("M2-feature",),
    )
    frozen = _FrozenPredictors()

    result = locked_test_score._apply_frozen_predictors(
        m0_matrix=cohort.m0_matrix,
        m1_matrix=cohort.m1_matrix,
        m2_matrix=cohort.m2_matrix,
        m0_names=cohort.m0_names,
        m1_names=cohort.m1_names,
        m2_names=cohort.m2_names,
        predictors=frozen,
    )

    assert frozen.calls == ["M0", "M1", "M2"]
    assert result["M2"].tolist() == [0.4, 0.4]
    source = Path(locked_test_score.__file__).read_text(encoding="utf-8")
    assert "build_calibration_features" not in source
    assert "fit_failure_predictors" not in source


def test_prediction_receipt_is_content_addressed_and_resumable(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "kind": "synthetic",
        "records": [{"episode_id": "not-a-real-locked-episode", "p": 0.5}],
    }

    first, first_sha = locked_test_score._publish_canonical_json(
        tmp_path, "predictions.json", payload
    )
    second, second_sha = locked_test_score._publish_canonical_json(
        tmp_path, "predictions.json", payload
    )

    assert first == second == tmp_path / first_sha / "predictions.json"
    assert first_sha == second_sha
    assert first.read_bytes() == locked_test_score._canonical(payload)


def test_score_root_rejects_foreign_episode_and_orphan_staging(
    tmp_path: Path,
) -> None:
    split = tmp_path / "locked_test"
    split.mkdir()
    (split / "foreign-episode").mkdir()
    with pytest.raises(RuntimeError, match="unexpected Locked Test"):
        locked_test_score._validate_score_root_layout(
            tmp_path, {"expected-episode"}, allow_active_publication=True
        )

    (split / "foreign-episode").rmdir()
    (split / ".expected-episode.tmp-abandoned").mkdir()
    with pytest.raises(RuntimeError, match="orphan"):
        locked_test_score._validate_score_root_layout(
            tmp_path, {"expected-episode"}, allow_active_publication=True
        )


def test_raw_split_requires_exact_real_episode_directories(tmp_path: Path) -> None:
    split = tmp_path / "locked_test"
    split.mkdir()
    (split / "expected-a").mkdir()
    (split / "expected-b").mkdir()
    locked_test_score._validate_raw_split_layout(tmp_path, {"expected-a", "expected-b"})

    (split / ".expected-a.tmp-abandoned").mkdir()
    with pytest.raises(RuntimeError, match="topology differs"):
        locked_test_score._validate_raw_split_layout(
            tmp_path, {"expected-a", "expected-b"}
        )
    (split / ".expected-a.tmp-abandoned").rmdir()
    (split / "expected-b").rmdir()
    (split / "expected-b").symlink_to(split / "expected-a", target_is_directory=True)
    with pytest.raises(RuntimeError, match="unsafe"):
        locked_test_score._validate_raw_split_layout(
            tmp_path, {"expected-a", "expected-b"}
        )


def test_feature_receipt_root_rejects_ambiguous_old_finalization(
    tmp_path: Path,
) -> None:
    old = tmp_path / ("a" * 64)
    old.mkdir()

    with pytest.raises(RuntimeError, match="ambiguous artifact"):
        locked_test_score._publish_canonical_json(
            tmp_path,
            "predictions.json",
            {"kind": "new-synthetic-finalization"},
        )
