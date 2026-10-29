import json

import pytest
from pydantic import ValidationError

from qwen_repair.config import DemoConfig
from qwen_repair.data.fixtures import FIXTURE_NAMES, load_demo_scenario, load_fixture
from qwen_repair.evaluation import evaluate_private, summarize
from qwen_repair.executors import FakeExecutor, code_hash
from qwen_repair.harness.controller import EpisodeController
from qwen_repair.harness.parser import CandidateError, parse_candidate
from qwen_repair.policies import ScriptedPolicy
from qwen_repair.schemas import PromptView, TestResult, Trajectory


def run_scenario(name, responses=None):
    scenario = load_demo_scenario(name)
    policy = ScriptedPolicy(scenario.responses if responses is None else responses)
    trajectory = EpisodeController(DemoConfig()).run(
        scenario.task,
        policy,
        scenario.executor,
        run_id="test-run",
        episode_id=f"test-{name}",
        scenario=name,
    )
    return scenario, policy, trajectory


def test_recovery_second_prompt_contains_only_public_feedback():
    scenario, policy, trajectory = run_scenario("recovery")
    fixture = load_fixture("sum_positive")
    assert "REFERENCE_SENTINEL" in fixture.reference_code
    assert "PRIVATE_SENTINEL" in json.dumps(fixture.private_cases)
    assert len(policy.calls) == 2
    assert policy.calls[1][-1].content == trajectory.attempts[0].public_feedback
    assert policy.calls[1][-2].content == trajectory.attempts[0].response
    messages = json.dumps([[item.model_dump() for item in call] for call in policy.calls])
    for forbidden in ("PRIVATE_SENTINEL", "REFERENCE_SENTINEL", scenario.task.private_tests_ref,
                      scenario.task.reference_ref):
        assert forbidden not in messages
    assert all(bundle_id == scenario.task.public_tests_ref for bundle_id, _ in scenario.executor.calls)
    assert [attempt.public_result.status for attempt in trajectory.attempts] == ["failed", "passed"]


def test_prompt_view_is_an_explicit_allowlist():
    task = load_fixture("sum_positive").task
    view = PromptView.from_task(task).model_dump()
    assert set(view) == {"schema_version", "task_id", "statement", "signature", "buggy_code", "entry_point"}
    with pytest.raises(ValidationError):
        PromptView.model_validate(task.model_dump())


def test_public_pass_does_not_stop_or_select_the_previous_candidate():
    _, _, trajectory = run_scenario("regression")
    assert trajectory.stop_reason == "budget_exhausted"
    assert [attempt.public_result.status for attempt in trajectory.attempts] == ["passed", "failed"]
    assert len(trajectory.attempts) == 2


def test_invalid_format_consumes_an_attempt_without_execution():
    scenario, _, trajectory = run_scenario("invalid_format")
    assert len(trajectory.attempts) == 2
    assert trajectory.attempts[0].parse_error == "format_error"
    assert trajectory.attempts[0].public_result is None
    assert trajectory.attempts[1].public_result.status == "passed"
    assert len(scenario.executor.calls) == 1


def test_infrastructure_failure_stops_with_a_distinct_reason():
    scenario, policy, trajectory = run_scenario("infra_failure")
    assert trajectory.stop_reason == "infra_error"
    assert len(trajectory.attempts) == len(policy.calls) == len(scenario.executor.calls) == 1
    assert trajectory.attempts[0].public_result.status == "infra_error"


def test_last_invalid_candidate_loses_success_but_keeps_cumulative_success():
    good = load_demo_scenario("regression").responses[0]
    scenario, _, trajectory = run_scenario("regression", [good, "not a code block"])
    evaluation = evaluate_private(trajectory, scenario.task, scenario.executor)
    metrics = summarize([trajectory], [evaluation])
    assert len(trajectory.attempts) == 2
    assert evaluation.attempt_results[0].status == "passed"
    assert evaluation.attempt_results[1] is None
    assert evaluation.terminal_reward == 0
    assert metrics["final_success_rate"] == 0
    assert metrics["solved_by_attempt"] == {"1": 1, "2": 1}
    assert metrics["regression_rate"] == 1


def test_private_executor_failure_is_excluded_without_zero_reward():
    scenario, _, trajectory = run_scenario("recovery")
    evaluation = evaluate_private(trajectory, scenario.task, FakeExecutor({}))
    metrics = summarize([trajectory], [evaluation])
    assert evaluation.terminal_reward is None
    assert evaluation.excluded_reason == "private_infra_error"
    assert metrics["eligible_episodes"] == 0
    assert metrics["infrastructure_error_episodes"] == 1
    assert metrics["final_success_rate"] is None


def test_trajectory_roundtrip_preserves_ids_schema_and_events():
    _, _, trajectory = run_scenario("recovery")
    restored = Trajectory.model_validate_json(trajectory.model_dump_json())
    assert restored == trajectory
    assert restored.schema_version == "1.0"
    assert restored.run_id == "test-run"
    assert restored.episode_id == "test-recovery"
    with pytest.raises(ValidationError):
        Trajectory.model_validate({**trajectory.model_dump(), "schema_version": "999"})


@pytest.mark.parametrize("response, expected_error", [
    ("def solve(x):\n    return x\n", "format_error"),
    ("```python\ndef solve(x):\n    return x\n```\n```python\npass\n```", "format_error"),
    ("Here is a solution:\n```python\ndef solve(x):\n    return x\n```", "format_error"),
    ("```python\ndef solve(x)\n    return x\n```", "syntax_error"),
    ("```python\ndef other(x):\n    return x\n```", "entry_point_error"),
    ("```python\nasync def solve(x):\n    return x\n```", "entry_point_error"),
    ("```python\ndef solve(x, y):\n    return x + y\n```", "signature_error"),
])
def test_parser_rejects_invalid_candidates(response, expected_error):
    with pytest.raises(CandidateError) as raised:
        parse_candidate(response, "solve", "def solve(x)")
    assert raised.value.code == expected_error


def test_parser_and_fake_executor_never_execute_candidate(tmp_path):
    marker = tmp_path / "candidate-was-executed"
    code = f"from pathlib import Path\nPath({str(marker)!r}).touch()\ndef solve(x):\n    return x\n"
    parsed = parse_candidate(f"```python\n{code}```", "solve", "def solve(x)")
    result = TestResult(status="passed", tests_passed=1, tests_total=1)
    executor = FakeExecutor({("known-bundle", code_hash(parsed)): result})
    assert executor.run(parsed, "known-bundle").status == "passed"
    assert executor.run(parsed, "unknown-bundle").status == "infra_error"
    assert not marker.exists()


def test_zero_tests_cannot_be_reported_as_success():
    with pytest.raises(ValidationError):
        TestResult(status="passed", tests_passed=0, tests_total=0)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_authored_fixtures_have_separate_cases_and_reference(name):
    fixture = load_fixture(name)
    assert len(fixture.public_cases) >= 2
    assert len(fixture.private_cases) >= 2
    assert "REFERENCE_SENTINEL" in fixture.reference_code
    for visibility in ("public", "private"):
        assert fixture.outcomes["corrected"][visibility]["status"] == "passed"
        assert fixture.outcomes["buggy"][visibility]["status"] == "failed"
