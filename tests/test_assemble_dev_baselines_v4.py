from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import assemble_dev_baselines_v4 as assembly


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "step6-dev-baseline-table.json"
)
MARKDOWN = ROOT / "benchmarks" / "v4" / "development-baselines.md"


def test_committed_step6_receipt_has_all_planners_and_exact_pairings() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    solved = {
        planner["planner_id"]: planner["solved_count"]
        for planner in payload["planners"]
    }
    assert solved == {
        "reactive_heuristic": 17,
        "bc_teacher": 31,
        "tuned_constant_rule": 33,
        "bc_initialization": 32,
        "shipped_v3_ppo_onnx": 31,
        "v4_ppo_1m": 35,
        "causal_mpc_k1": 18,
        "causal_mpc_k3": 29,
        "causal_mpc_k5": 30,
        "clairvoyant_cem_oracle": 37,
    }
    assert all(payload["invariants"].values())
    assert payload["split"] == "dev"
    assert payload["final_split_used"] is False
    assert payload["training_performed"] is False

    expected_pairs = {
        "v4_ppo_1m_vs_tuned_constant_rule": (33, 2, 0, 5, 0.5),
        "v4_ppo_1m_vs_bc_teacher": (30, 5, 1, 4, 0.21875),
        "v4_ppo_1m_vs_shipped_v3_ppo_onnx": (28, 7, 3, 2, 0.34375),
        "v4_ppo_1m_vs_clairvoyant_cem_oracle": (35, 0, 2, 3, 0.5),
    }
    for key, expected in expected_pairs.items():
        pair = payload["paired_comparisons"][key]
        assert (
            pair["both_solved"],
            pair["left_only"],
            pair["right_only"],
            pair["neither"],
            pair["exact_mcnemar_p_two_sided"],
        ) == expected


def test_markdown_is_hash_bound_and_discloses_privileged_oracle() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    markdown_bytes = MARKDOWN.read_bytes()
    assert hashlib.sha256(markdown_bytes).hexdigest() == payload["markdown"]["sha256"]
    markdown = markdown_bytes.decode("utf-8")
    assert "privileged and clairvoyant" in markdown
    assert "not a submission baseline" in markdown
    assert "v4 PPO vs shipped v3 PPO" in markdown


def test_cli_refuses_to_overwrite_outputs(tmp_path: Path) -> None:
    receipt = tmp_path / "exists.json"
    markdown = tmp_path / "exists.md"
    receipt.write_text("existing", encoding="utf-8")
    assert (
        assembly.main(
            [
                "--developmental-nonauthorizing",
                "--receipt",
                str(receipt),
                "--markdown",
                str(markdown),
            ]
        )
        == 2
    )
