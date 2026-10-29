from collections.abc import Mapping
from hashlib import sha256

from qwen_repair.schemas import TestResult


def code_hash(candidate: str) -> str:
    return sha256(candidate.encode("utf-8")).hexdigest()


class FakeExecutor:
    """Look up authored outcomes without evaluating or importing candidate code."""

    def __init__(self, results: Mapping[tuple[str, str], TestResult]) -> None:
        self._results = {
            key: value.model_copy(deep=True) for key, value in results.items()
        }
        self.calls: list[tuple[str, str]] = []

    def run(self, candidate: str, test_bundle_id: str) -> TestResult:
        key = (test_bundle_id, code_hash(candidate))
        self.calls.append(key)
        result = self._results.get(key)
        if result is None:
            return TestResult(
                status="infra_error",
                feedback="No synthetic outcome is registered for this code and test bundle.",
                tests_passed=0,
                tests_failed=0,
                tests_total=0,
                duration_seconds=0.0,
                synthetic_demo=True,
            )
        return result.model_copy(deep=True)
