from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

import model.policy as policy_module
from model.policy import PolicyError, load_policy

ROOT = Path(__file__).resolve().parents[1]
LEGACY_POLICY = ROOT / "artifacts" / "city_recovery_ppo.v3.selected.onnx"
LEGACY_POLICY_SHA256 = (
    "6a08ae284fb93cff1155ce37dcec4fac1121697add0fabd9d367486be344bf0b"
)
ZERO_ACTION_SHA256 = (
    "ffb0d191e76274edaff60d384695431984cd084871140f529e93484f3512542d"
)
RAMP_ACTION_SHA256 = (
    "909e357e9fedcec80815276788b0da22b8478556d2038b36559960089f5ef9f1"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _raw_action_sha256(action: np.ndarray) -> str:
    raw = action.astype(np.float32).reshape(1, 22)
    return _sha256(raw.tobytes())


def _write_identity_model(
    path: Path,
    *,
    input_name: str = "observation",
    output_name: str = "action",
    width: int = 73,
) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", [input_name], [output_name])],
        "malformed-policy-contract",
        [
            helper.make_tensor_value_info(
                input_name,
                TensorProto.FLOAT,
                ["batch", width],
            )
        ],
        [
            helper.make_tensor_value_info(
                output_name,
                TensorProto.FLOAT,
                ["batch", width],
            )
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 10
    onnx.save(model, path)


def _write_out_of_bounds_model(path: Path) -> None:
    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), "starts")
    ends = numpy_helper.from_array(np.array([22], dtype=np.int64), "ends")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")
    steps = numpy_helper.from_array(np.array([1], dtype=np.int64), "steps")
    zero = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero")
    two = numpy_helper.from_array(np.array(2.0, dtype=np.float32), "two")
    graph = helper.make_graph(
        [
            helper.make_node(
                "Slice",
                ["observation", "starts", "ends", "axes", "steps"],
                ["selected"],
            ),
            helper.make_node("Mul", ["selected", "zero"], ["zeros"]),
            helper.make_node("Add", ["zeros", "two"], ["action"]),
        ],
        "out-of-bounds-policy",
        [
            helper.make_tensor_value_info(
                "observation",
                TensorProto.FLOAT,
                ["batch", 73],
            )
        ],
        [
            helper.make_tensor_value_info(
                "action",
                TensorProto.FLOAT,
                ["batch", 22],
            )
        ],
        initializer=[starts, ends, axes, steps, zero, two],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 10
    onnx.save(model, path)


def test_legacy_artifact_identity_and_deterministic_predictions() -> None:
    assert _sha256(LEGACY_POLICY.read_bytes()) == LEGACY_POLICY_SHA256

    policy = load_policy(LEGACY_POLICY, expected_sha256=LEGACY_POLICY_SHA256)
    zero = policy.predict(np.zeros(73, dtype=np.float64))
    ramp_input = np.linspace(-1.0, 1.0, 73, dtype=np.float64)
    ramp = policy.predict(ramp_input)

    assert policy.path == LEGACY_POLICY.resolve()
    assert policy.sha256 == LEGACY_POLICY_SHA256
    assert policy.session.get_providers() == ["CPUExecutionProvider"]
    assert zero.shape == (22,)
    assert zero.dtype == np.float64
    assert ramp.shape == (22,)
    assert ramp.dtype == np.float64
    assert _raw_action_sha256(zero) == ZERO_ACTION_SHA256
    assert _raw_action_sha256(ramp) == RAMP_ACTION_SHA256
    assert np.array_equal(ramp, policy.predict(ramp_input))


def test_load_policy_cache_is_keyed_by_resolved_path_and_hash() -> None:
    first = load_policy(LEGACY_POLICY)
    second = load_policy(LEGACY_POLICY.parent / "." / LEGACY_POLICY.name)

    assert first is second


def test_load_policy_does_not_reuse_cache_after_bytes_change(tmp_path: Path) -> None:
    copied_policy = tmp_path / "policy.onnx"
    shutil.copyfile(LEGACY_POLICY, copied_policy)
    loaded = load_policy(copied_policy)
    copied_policy.write_bytes(b"changed after the first load")

    with pytest.raises(PolicyError, match="loadable ONNX"):
        load_policy(copied_policy)

    shutil.copyfile(LEGACY_POLICY, copied_policy)
    assert load_policy(copied_policy) is loaded


def test_session_is_cpu_sequential_and_single_threaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_policy = tmp_path / "policy.onnx"
    shutil.copyfile(LEGACY_POLICY, copied_policy)
    captured: dict[str, object] = {}
    original_session = policy_module.ort.InferenceSession

    def recording_session(*args: object, **kwargs: object) -> ort.InferenceSession:
        options = kwargs["sess_options"]
        captured["execution_mode"] = options.execution_mode
        captured["intra_op_num_threads"] = options.intra_op_num_threads
        captured["inter_op_num_threads"] = options.inter_op_num_threads
        captured["providers"] = kwargs["providers"]
        return original_session(*args, **kwargs)

    monkeypatch.setattr(policy_module.ort, "InferenceSession", recording_session)
    load_policy(copied_policy)

    assert captured == {
        "execution_mode": ort.ExecutionMode.ORT_SEQUENTIAL,
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "providers": ["CPUExecutionProvider"],
    }


@pytest.mark.parametrize(
    "observation, message",
    [
        (np.zeros(72), "shape"),
        (np.zeros((1, 73)), "shape"),
        (np.full(73, np.nan), "finite"),
        (np.full(73, np.inf), "finite"),
        (["not-a-number"] * 73, "numeric"),
    ],
)
def test_predict_rejects_invalid_observations(
    observation: object,
    message: str,
) -> None:
    policy = load_policy(LEGACY_POLICY)

    with pytest.raises(PolicyError, match=message):
        policy.predict(observation)


def test_load_policy_rejects_missing_artifact_and_bad_hash(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match="missing or unreadable"):
        load_policy(tmp_path / "missing.onnx")
    with pytest.raises(PolicyError, match="64-character"):
        load_policy(LEGACY_POLICY, expected_sha256="not-a-digest")
    with pytest.raises(PolicyError, match="does not match"):
        load_policy(LEGACY_POLICY, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    ("input_name", "output_name", "width", "message"),
    [
        ("state", "action", 73, "input"),
        ("observation", "decision", 73, "output"),
        ("observation", "action", 72, "input"),
    ],
)
def test_load_policy_rejects_malformed_onnx_contract(
    tmp_path: Path,
    input_name: str,
    output_name: str,
    width: int,
    message: str,
) -> None:
    path = tmp_path / f"malformed-{input_name}-{output_name}-{width}.onnx"
    _write_identity_model(
        path,
        input_name=input_name,
        output_name=output_name,
        width=width,
    )

    with pytest.raises(PolicyError, match=message):
        load_policy(path)


def test_load_policy_rejects_invalid_onnx_bytes(tmp_path: Path) -> None:
    path = tmp_path / "invalid.onnx"
    path.write_bytes(b"not an ONNX model")

    with pytest.raises(PolicyError, match="loadable ONNX"):
        load_policy(path)


def test_load_policy_rejects_out_of_bounds_zero_smoke(tmp_path: Path) -> None:
    path = tmp_path / "out-of-bounds.onnx"
    _write_out_of_bounds_model(path)

    with pytest.raises(PolicyError, match="outside"):
        load_policy(path)
