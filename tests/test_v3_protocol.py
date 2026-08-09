from __future__ import annotations

import unittest

from scripts.v3_protocol import (
    FINAL_STATE_PATHS,
    PROTOCOL_PATH,
    ROOT,
    SCIENTIFIC_SOURCE_PATHS,
    SOURCE_SEAL_PATH,
    canonical_hash,
    expected_protocol,
    expected_source_seal,
    verify_documents,
)


class V3ProtocolTests(unittest.TestCase):
    def test_frozen_documents_match_live_contracts(self) -> None:
        self.assertEqual(PROTOCOL_PATH.exists(), SOURCE_SEAL_PATH.exists())
        if PROTOCOL_PATH.exists():
            summary = verify_documents()
            self.assertEqual(summary["status"], "v3-protocol-verified")
            self.assertEqual(summary["final_case_count"], 40)
            final_unused = all(
                not (ROOT / relative).exists() for relative in FINAL_STATE_PATHS
            )
            self.assertEqual(summary["final_unused"], final_unused)
        else:
            self.assertEqual(
                expected_protocol()["status"],
                "preregistered_separate_authorizations",
            )

    def test_split_ranges_are_disjoint_and_final_is_exactly_forty(self) -> None:
        protocol = expected_protocol()
        splits = protocol["splits"]
        intervals = []
        for split_id in ("training", "development", "final"):
            interval = splits[split_id]["seed_interval"]
            values = set(range(interval["first"], interval["last"] + 1))
            self.assertEqual(len(values), interval["count"])
            intervals.append(values)
        self.assertFalse(intervals[0] & intervals[1])
        self.assertFalse(intervals[0] & intervals[2])
        self.assertFalse(intervals[1] & intervals[2])
        self.assertEqual(splits["final"]["cartesian_case_count"], 40)
        self.assertEqual(protocol["final_evaluation"]["case_count"], 40)

    def test_selection_cannot_consume_final_metrics(self) -> None:
        selection = expected_protocol()["checkpoint_selection"]
        self.assertEqual(selection["selection_dataset"], "development")
        self.assertFalse(selection["final_metrics_may_influence_selection"])
        self.assertTrue(selection["receipt_write_once"])
        self.assertEqual(selection["campaign_completion_required_transitions"], 645120)
        self.assertEqual(selection["individual_checkpoint_minimum_transitions"], 92160)
        self.assertEqual(selection["stage_count"], 7)
        self.assertEqual(
            selection["registered_stage_boundaries"],
            [92160, 184320, 276480, 368640, 460800, 552960, 645120],
        )
        self.assertTrue(selection["outage_resume_from_latest_complete_stage_only"])
        self.assertEqual(
            selection["deployment_staging"]["claim_name"],
            "selection-claim.json",
        )
        self.assertEqual(
            selection["deployment_staging"]["stage_receipt_name"],
            "selected-deployment-stage-receipt.json",
        )
        self.assertEqual(
            [item["metric"] for item in selection["ranking"]],
            [
                "candidate_solved_count",
                "candidate_mean_resilience_auc",
                "completed_transitions",
                "checkpoint_sha256",
            ],
        )

    def test_source_identity_binds_selection_and_final_materializer(self) -> None:
        self.assertIn("scripts/select_policy_v3.py", SCIENTIFIC_SOURCE_PATHS)
        self.assertIn("scripts/evaluate_policy_v3.py", SCIENTIFIC_SOURCE_PATHS)

    def test_authorization_is_separate_and_final_is_deferred(self) -> None:
        authorization = expected_protocol()["authorization"]
        self.assertFalse(authorization["protocol_itself_authorizes_execution"])
        self.assertEqual(
            authorization["training_and_development_selection"]["required_status"],
            "training_and_development_selection_authorized",
        )
        self.assertEqual(
            authorization["final_evaluation"]["required_status"],
            "final_evaluation_authorized",
        )
        self.assertTrue(
            authorization["final_evaluation"][
                "may_be_created_only_after_checkpoint_selection"
            ]
        )

    def test_final_terminal_contract_pins_selected_policy_and_hash_chain(self) -> None:
        final = expected_protocol()["final_evaluation"]
        self.assertEqual(
            final["selected_policy_id"], "city-recovery-sb3-ppo-v3-selected"
        )
        terminal = final["terminal_marker"]
        self.assertEqual(terminal["path"], "training/v3/final-terminal.json")
        self.assertEqual(terminal["status"], "final_evaluation_complete")
        self.assertTrue(terminal["write_once"])
        self.assertIn("final_result_sha256", terminal["required_bindings"])
        self.assertIn("final_ledger_sha256", terminal["required_bindings"])
        self.assertEqual(
            final["torn_eof_recovery"]["scope"],
            "last_non_newline_tail_only",
        )

    def test_protocol_tool_does_not_mutate_final_state(self) -> None:
        before = {
            relative: (ROOT / relative).exists() for relative in FINAL_STATE_PATHS
        }
        expected_protocol()
        after = {
            relative: (ROOT / relative).exists() for relative in FINAL_STATE_PATHS
        }
        self.assertEqual(after, before)

    def test_source_seal_chain_is_self_consistent(self) -> None:
        protocol = expected_protocol()
        seal = expected_source_seal(protocol)
        base = {key: value for key, value in seal.items() if key != "seal_sha256"}
        self.assertEqual(seal["seal_sha256"], canonical_hash(base))


if __name__ == "__main__":
    unittest.main()
