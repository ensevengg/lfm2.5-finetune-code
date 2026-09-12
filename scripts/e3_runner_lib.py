#!/usr/bin/env python3
"""Pure logic for the E3 pilot runner: retry policy, prompts, records.

The retry policy is the one frozen in suite e3-humanevalplus-pilot-v1:

- Infrastructure failures (server death, connection error, HTTP 5xx) get one
  automatic retry on a fresh server process, recorded as attempt 2 with the
  original failure preserved.
- A second consecutive infrastructure failure aborts the row.
- Model outcomes (empty completion, token-limit finish, malformed answer) are
  never retried; they are terminal evidence.
- Wrong model identity or duplicate generation keys abort immediately.
"""

from __future__ import annotations

from typing import Any

BOS_TOKEN_ID = 1
PLACEHOLDER = "{{BENCHMARK_PROMPT}}"

INFRASTRUCTURE_FAILURES = frozenset(
    {"server_death", "connection_error", "http_5xx", "health_check_timeout"}
)
MODEL_OUTCOMES = frozenset({"completed", "empty_completion", "length_limit"})

RETRY_FRESH_SERVER = "retry_fresh_server"
ABORT_ROW = "abort_row"
RECORD_OUTCOME = "record_outcome"


def decide_retry(attempt: int, failure_class: str) -> str:
    """Frozen E3 retry decision for one request attempt (1-based)."""
    if attempt < 1:
        raise ValueError(f"Attempt numbers are 1-based; got {attempt}")
    if failure_class in MODEL_OUTCOMES:
        return RECORD_OUTCOME
    if failure_class == "model_identity_mismatch":
        return ABORT_ROW
    if failure_class in INFRASTRUCTURE_FAILURES:
        if attempt == 1:
            return RETRY_FRESH_SERVER
        return ABORT_ROW
    raise ValueError(f"Unknown failure class: {failure_class!r}")


def build_prompt(template_text: str, benchmark_prompt: str) -> str:
    """Insert the unmodified benchmark prompt into the frozen template."""
    if template_text.count(PLACEHOLDER) != 1:
        raise ValueError(
            "Prompt template must contain exactly one {{BENCHMARK_PROMPT}} placeholder"
        )
    return template_text.replace(PLACEHOLDER, benchmark_prompt)


def ensure_single_bos(token_ids: list[int]) -> bool:
    """True when the tokenized prompt starts with exactly one BOS token."""
    return (
        bool(token_ids)
        and token_ids[0] == BOS_TOKEN_ID
        and sum(1 for token_id in token_ids if token_id == BOS_TOKEN_ID) == 1
    )


def classify_finish_reason(finish_reason: str, raw_completion: str) -> str:
    """Map a raw model outcome to the E3 failure-class vocabulary."""
    if finish_reason == "eos":
        return "completed" if raw_completion.strip() else "empty_completion"
    if finish_reason == "length":
        return "length_limit"
    raise ValueError(f"Unusable finish reason: {finish_reason!r}")


def generation_record(
    *,
    schema_version: int,
    suite_id: str,
    run_id: str,
    row_alias: str,
    task_id: str,
    entry_point: str,
    attempt: int,
    case_kind: str = "humanevalplus_pilot",
    model_path: str,
    model_sha256: str,
    prompt_id: str,
    prompt_sha256: str,
    templated_prompt: str,
    templated_prompt_token_ids: list[int],
    raw_completion: str,
    completion_token_ids: list[int],
    finish_reason: str,
    timings: dict[str, Any],
    started_at_utc: str,
    finished_at_utc: str,
    status: str,
) -> dict[str, Any]:
    """One terminal generation record, schema-compatible with E1 evidence."""
    if attempt < 1:
        raise ValueError("Attempt numbers are 1-based")
    return {
        "schema_version": schema_version,
        "suite_id": suite_id,
        "run_id": run_id,
        "row_alias": row_alias,
        "case_kind": case_kind,
        "task_id": task_id,
        "entry_point": entry_point,
        "attempt": attempt,
        "model_alias": row_alias,
        "model_path": model_path,
        "model_sha256": model_sha256,
        "prompt_id": prompt_id,
        "prompt_sha256": prompt_sha256,
        "messages": [{"role": "user", "content": templated_prompt}],
        "templated_prompt": templated_prompt,
        "templated_prompt_token_ids": templated_prompt_token_ids,
        "templated_prompt_token_count": len(templated_prompt_token_ids),
        "completion_token_count": len(completion_token_ids),
        "completion_token_ids": completion_token_ids,
        "raw_completion": raw_completion,
        "finish_reason": finish_reason,
        "timings": timings,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "status": status,
        "generated_program_executed": False,
        "scored": False,
    }


def row_summary(generations: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one row's evidence into the pilot-report figures."""
    if not generations:
        raise ValueError("Row summary requires at least one generation record")
    attempts_by_task: dict[str, list[int]] = {}
    for record in generations:
        attempts_by_task.setdefault(record["task_id"], []).append(record["attempt"])
    duplicate_keys = sum(
        1 for attempts in attempts_by_task.values() if len(set(attempts)) != len(attempts)
    )
    retried_tasks = sum(1 for attempts in attempts_by_task.values() if len(attempts) > 1)
    terminal_records = sum(
        1 for record in generations if record["status"] in {"completed", "empty_completion", "length_limit"}
    )
    infra_failures = sum(1 for record in generations if record["status"].startswith("infrastructure_"))
    return {
        "task_count": len(attempts_by_task),
        "generation_record_count": len(generations),
        "terminal_record_count": terminal_records,
        "retried_task_count": retried_tasks,
        "infrastructure_failure_count": infra_failures,
        "duplicate_generation_keys": duplicate_keys,
        "total_completion_tokens": sum(
            record["completion_token_count"] for record in generations
        ),
        "generated_programs_executed": 0,
    }
