from types import SimpleNamespace

import pytest

from qwen_repair.harness.controller import EpisodeController
from qwen_repair.harness.feedback import format_feedback, sanitize_diagnostic
from qwen_repair.schemas import (
    Attempt,
    Generation,
    TaskRecord,
    TestResult,
    count_message_tokens,
    count_tokens,
)


def task():
    return TaskRecord(
        task_id="edge", source_id="own", group_id="edge", split="demo",
        statement="Return the input unchanged.", signature="def identity(value):",
        entry_point="identity", buggy_code="def identity(value):\n    return None\n",
        bug_type="wrong_return", public_tests_ref="edge/public",
        private_tests_ref="edge/private", reference_ref="edge/reference",
    )


def config(**overrides):
    return SimpleNamespace(**{
        "max_attempts": 2,
        "max_new_tokens": 512,
        "max_context_tokens": 4096,
        "max_feedback_tokens": 256,
        **overrides,
    })


class Policy:
    policy_version = "fixed-v1"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, messages, generation_config):
        self.calls += 1
        response = next(self.responses)
        return Generation(
            text=response,
            input_tokens=count_message_tokens(messages),
            output_tokens=count_tokens(response),
            policy_version=self.policy_version,
        )


class Executor:
    def __init__(self):
        self.calls = []

    def run(self, candidate, test_bundle_id):
        self.calls.append((candidate, test_bundle_id))
        return TestResult(status="passed", tests_passed=1, tests_total=1)


def run(policy, executor, episode_config=None):
    return EpisodeController(episode_config or config()).run(
        task(), policy, executor,
        run_id="run", episode_id="episode", scenario="edge",
    )


def test_context_budget_stops_before_generation_without_truncating_task():
    policy, executor = Policy([]), Executor()
    trajectory = run(policy, executor, config(max_context_tokens=512))
    assert trajectory.stop_reason == "context_budget_exceeded"
    assert trajectory.attempts == []
    assert policy.calls == 0
    assert executor.calls == []


def test_output_over_budget_consumes_attempt_without_execution():
    policy = Policy(["x" * 513, "```python\ndef identity(value):\n    return value\n```"])
    executor = Executor()
    trajectory = run(policy, executor)
    assert len(trajectory.attempts) == 2
    assert trajectory.attempts[0].parse_error == "output_budget_exceeded"
    assert trajectory.attempts[0].code is None
    assert trajectory.attempts[1].public_result.status == "passed"
    assert len(executor.calls) == 1


def test_policy_cannot_change_both_reported_and_live_version():
    class MutatingPolicy(Policy):
        def generate(self, messages, generation_config):
            self.policy_version = "changed-v2"
            return super().generate(messages, generation_config)

    policy = MutatingPolicy(["```python\ndef identity(value):\n    return value\n```"])
    executor = Executor()
    with pytest.raises(ValueError, match="changed version"):
        run(policy, executor)
    assert executor.calls == []


def test_feedback_removes_frames_paths_controls_and_limits_utf8_bytes():
    diagnostic = (
        'Traceback (most recent call last):\n'
        '  File "/private/tmp/secret/driver.py", line 17, in internal_judge\n'
        '    internal_helper()\n'
        '    ^^^^^^^^^^^^^^^^^\n'
        '\x1b[31mValueError: /private/tmp/secret/value.txt\x1b[0m\n'
        'Example: C:\\Users\\person\\secret.txt\n'
        'Do not include this third line'
    )
    sanitized = sanitize_diagnostic(diagnostic)
    assert sanitized == "ValueError: [path]\nExample: [path]"
    result = TestResult(
        status="failed", tests_total=2, tests_passed=1, tests_failed=1,
        feedback=diagnostic + "я" * 500,
    )
    feedback = format_feedback(result, max_tokens=64)
    assert len(feedback.encode("utf-8")) <= 64
    assert feedback.startswith("Public status: failed; passed 1/2.")
    assert not any(value in feedback for value in ["Traceback", "internal_", "/private", "person", "\x1b"])


def test_zero_collected_tests_cannot_be_a_pass():
    with pytest.raises(ValueError, match="at least one test"):
        TestResult(status="passed")


@pytest.mark.parametrize("changes", [
    {"code": None, "public_result": None, "parse_error": None},
    {"code": None},
    {"public_result": None},
    {"parse_error": "format_error"},
    {"code": None, "public_result": None, "parse_error": " "},
])
def test_inconsistent_imported_attempts_are_rejected(changes):
    response = "```python\ndef identity(value):\n    return value\n```"
    attempt = run(Policy([response, response]), Executor()).attempts[0].model_dump()
    attempt.update(changes)
    with pytest.raises(ValueError):
        Attempt.model_validate(attempt)
