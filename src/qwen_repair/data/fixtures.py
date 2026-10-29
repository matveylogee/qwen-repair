import json
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from typing import Any

from qwen_repair.executors import FakeExecutor, code_hash
from qwen_repair.schemas import TaskRecord, TestResult

SCENARIO_NAMES = (
    "recovery",
    "invalid_format",
    "repeated_error",
    "infra_failure",
    "regression",
)
FIXTURE_NAMES = ("sum_positive", "last_or_none", "count_even", "is_palindrome")
_SCENARIO_FIXTURE = {
    "recovery": "sum_positive",
    "invalid_format": "last_or_none",
    "repeated_error": "count_even",
    "infra_failure": "is_palindrome",
    "regression": "is_palindrome",
}


@dataclass(frozen=True)
class DemoFixture:
    task: TaskRecord
    reference_code: str
    corrected_code: str
    public_cases: tuple[dict[str, Any], ...]
    private_cases: tuple[dict[str, Any], ...]
    outcomes: dict[str, dict[str, dict[str, Any]]]


@dataclass(frozen=True)
class DemoScenario:
    name: str
    task: TaskRecord
    responses: list[str]
    executor: FakeExecutor


def _fixture_directory():
    return files("qwen_repair.data").joinpath("fixtures")


def fixture_hashes() -> dict[str, str]:
    return {
        f"{name}.json": sha256(
            _fixture_directory().joinpath(f"{name}.json").read_bytes()
        ).hexdigest()
        for name in FIXTURE_NAMES
    }


def load_fixture(name: str) -> DemoFixture:
    if name not in FIXTURE_NAMES:
        raise ValueError(f"Unknown fixture: {name}")
    payload = json.loads(
        _fixture_directory().joinpath(f"{name}.json").read_text(encoding="utf-8")
    )
    if payload["schema_version"] != "1.0" or payload["synthetic_demo"] is not True:
        raise ValueError("Unsupported fixture schema or non-synthetic fixture")
    return DemoFixture(
        task=TaskRecord.model_validate(payload["task"]),
        reference_code=payload["reference_code"].strip(),
        corrected_code=payload["corrected_code"].strip(),
        public_cases=tuple(payload["public_cases"]),
        private_cases=tuple(payload["private_cases"]),
        outcomes=payload["outcomes"],
    )


def _prepared_results(fixture: DemoFixture) -> dict[tuple[str, str], TestResult]:
    results = {}
    variants = {
        "buggy": fixture.task.buggy_code,
        "corrected": fixture.corrected_code,
        "reference": fixture.reference_code,
    }
    bundles = {
        "public": fixture.task.public_tests_ref,
        "private": fixture.task.private_tests_ref,
    }
    for variant, candidate in variants.items():
        outcome_variant = "corrected" if variant == "reference" else variant
        for visibility, bundle_id in bundles.items():
            outcome = fixture.outcomes[outcome_variant][visibility]
            result = TestResult.model_validate(
                {**outcome, "duration_seconds": 0.0, "synthetic_demo": True}
            )
            # The parser emits a final newline; raw fixture strings omit it.
            for serialized_candidate in (candidate, candidate + "\n"):
                results[(bundle_id, code_hash(serialized_candidate))] = result
    return results


def load_demo_scenario(name: str) -> DemoScenario:
    if name not in SCENARIO_NAMES:
        raise ValueError(f"Unknown demo scenario: {name}")
    fixture = load_fixture(_SCENARIO_FIXTURE[name])
    results = _prepared_results(fixture)
    bad = f"```python\n{fixture.task.buggy_code}\n```"
    good = f"```python\n{fixture.corrected_code}\n```"
    responses = {
        "recovery": [bad, good],
        "invalid_format": [fixture.corrected_code, good],
        "repeated_error": [bad, bad],
        "infra_failure": [bad, good],
        "regression": [good, bad],
    }[name]
    if name == "infra_failure":
        failure = TestResult(
            status="infra_error",
            tests_passed=0,
            tests_failed=0,
            tests_total=0,
            feedback="Synthetic executor unavailable; no tests were run.",
            duration_seconds=0.0,
            synthetic_demo=True,
        )
        for candidate in (fixture.task.buggy_code, fixture.task.buggy_code + "\n"):
            results[(fixture.task.public_tests_ref, code_hash(candidate))] = failure
    return DemoScenario(
        name=name,
        task=fixture.task,
        responses=responses,
        executor=FakeExecutor(results),
    )
