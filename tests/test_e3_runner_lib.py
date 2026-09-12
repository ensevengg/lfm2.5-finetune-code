from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "scripts"))

from e3_runner_lib import (  # noqa: E402
    ABORT_ROW,
    RECORD_OUTCOME,
    RETRY_FRESH_SERVER,
    build_prompt,
    classify_finish_reason,
    decide_retry,
    ensure_single_bos,
    generation_record,
    row_summary,
)


class RetryPolicyTests(unittest.TestCase):
    def test_infrastructure_failure_retries_once_on_a_fresh_server(self) -> None:
        for failure in ("server_death", "connection_error", "http_5xx"):
            with self.subTest(failure=failure):
                self.assertEqual(decide_retry(1, failure), RETRY_FRESH_SERVER)
                self.assertEqual(decide_retry(2, failure), ABORT_ROW)

    def test_model_outcomes_are_never_retried(self) -> None:
        for outcome in ("completed", "empty_completion", "length_limit"):
            with self.subTest(outcome=outcome):
                self.assertEqual(decide_retry(1, outcome), RECORD_OUTCOME)
                self.assertEqual(decide_retry(3, outcome), RECORD_OUTCOME)

    def test_identity_mismatch_never_retries(self) -> None:
        self.assertEqual(decide_retry(1, "model_identity_mismatch"), ABORT_ROW)

    def test_unknown_failure_class_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            decide_retry(1, "mystery")

    def test_attempts_are_one_based(self) -> None:
        with self.assertRaises(ValueError):
            decide_retry(0, "server_death")


class PromptTests(unittest.TestCase):
    TEMPLATE = "Instructions\n\n{{BENCHMARK_PROMPT}}\nEnd.\n"

    def test_benchmark_prompt_is_inserted_unmodified(self) -> None:
        self.assertEqual(
            build_prompt(self.TEMPLATE, "def foo():\n    ..."),
            "Instructions\n\ndef foo():\n    ...\nEnd.\n",
        )

    def test_template_needs_exactly_one_placeholder(self) -> None:
        with self.assertRaises(ValueError):
            build_prompt("no placeholder", "x")
        with self.assertRaises(ValueError):
            build_prompt("{{BENCHMARK_PROMPT}} {{BENCHMARK_PROMPT}}", "x")


class BosTests(unittest.TestCase):
    def test_exactly_one_leading_bos_passes(self) -> None:
        self.assertTrue(ensure_single_bos([1, 4, 5, 6]))
        self.assertTrue(ensure_single_bos([1]))

    def test_zero_duplicate_or_misplaced_bos_fail(self) -> None:
        self.assertFalse(ensure_single_bos([4, 5]))
        self.assertFalse(ensure_single_bos([1, 4, 1]))
        self.assertFalse(ensure_single_bos([]))
        self.assertFalse(ensure_single_bos([4, 1]))


class ClassificationTests(unittest.TestCase):
    def test_finish_reason_mapping(self) -> None:
        self.assertEqual(classify_finish_reason("eos", "def foo(): ..."), "completed")
        self.assertEqual(classify_finish_reason("eos", "   \n"), "empty_completion")
        self.assertEqual(classify_finish_reason("length", "def foo("), "length_limit")
        with self.assertRaises(ValueError):
            classify_finish_reason("cancelled", "x")


class RecordTests(unittest.TestCase):
    def base_kwargs(self) -> dict:
        return dict(
            schema_version=1,
            suite_id="e3-humanevalplus-pilot-v1",
            run_id="run-1",
            row_alias="merged-bf16",
            task_id="HumanEval/1",
            entry_point="foo",
            attempt=1,
            model_path="/model/model.gguf",
            model_sha256="abc",
            prompt_id="lfm-code-v1",
            prompt_sha256="def",
            templated_prompt="prompt",
            templated_prompt_token_ids=[1, 2, 3],
            raw_completion="answer",
            completion_token_ids=[9, 8],
            finish_reason="eos",
            timings={"predicted_ms": 1.0},
            started_at_utc="t0",
            finished_at_utc="t1",
            status="completed",
        )

    def test_record_carries_full_identity_and_evidence(self) -> None:
        record = generation_record(**self.base_kwargs())
        for field in (
            "schema_version", "suite_id", "run_id", "row_alias", "task_id",
            "attempt", "model_sha256", "prompt_id", "prompt_sha256",
            "templated_prompt_token_ids", "raw_completion", "finish_reason",
            "timings", "started_at_utc", "finished_at_utc", "status",
        ):
            self.assertIn(field, record)
        self.assertEqual(record["templated_prompt_token_count"], 3)
        self.assertEqual(record["completion_token_count"], 2)
        self.assertFalse(record["generated_program_executed"])
        self.assertFalse(record["scored"])

    def test_attempt_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            generation_record(**{**self.base_kwargs(), "attempt": 0})


class RowSummaryTests(unittest.TestCase):
    def base_kwargs(self) -> dict:
        return dict(
            schema_version=1,
            suite_id="s",
            run_id="r",
            row_alias="merged-bf16",
            model_path="/m",
            entry_point="",
            model_sha256="h",
            prompt_id="p",
            prompt_sha256="q",
            templated_prompt="t",
            templated_prompt_token_ids=[1],
            timings={},
            started_at_utc="t0",
            finished_at_utc="t1",
        )

    def test_summary_aggregates_retries_and_tokens(self) -> None:
        records = [
            generation_record(
                **self.base_kwargs(),
                task_id="HumanEval/1",
                attempt=1,
                raw_completion="a",
                completion_token_ids=[1, 2],
                finish_reason="eos",
                status="completed",
            ),
            generation_record(
                **self.base_kwargs(),
                task_id="HumanEval/6",
                attempt=1,
                raw_completion="",
                completion_token_ids=[],
                finish_reason="length",
                status="infrastructure_server_death",
            ),
            generation_record(
                **self.base_kwargs(),
                task_id="HumanEval/6",
                attempt=2,
                raw_completion="b",
                completion_token_ids=[3],
                finish_reason="eos",
                status="completed",
            ),
        ]
        summary = row_summary(records)
        self.assertEqual(summary["task_count"], 2)
        self.assertEqual(summary["generation_record_count"], 3)
        self.assertEqual(summary["terminal_record_count"], 2)
        self.assertEqual(summary["retried_task_count"], 1)
        self.assertEqual(summary["infrastructure_failure_count"], 1)
        self.assertEqual(summary["duplicate_generation_keys"], 0)
        self.assertEqual(summary["total_completion_tokens"], 3)
        self.assertEqual(summary["generated_programs_executed"], 0)


if __name__ == "__main__":
    unittest.main()
