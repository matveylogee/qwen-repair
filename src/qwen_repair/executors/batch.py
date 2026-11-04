from collections.abc import Sequence

from qwen_repair.schemas import TestResult

from .filequeue import FileQueueExecutor, _infra

MAX_BATCH_CHECKS = 32


def run_checks(checks: Sequence[tuple[object, str, str]]) -> list[TestResult]:
    """Publish bounded queue checks together and return results in input order."""
    if len(checks) > MAX_BATCH_CHECKS:
        raise ValueError(f"A check batch cannot exceed {MAX_BATCH_CHECKS} entries")
    queued = {backend: set(backend.pending) for backend, _, _ in checks if isinstance(backend, FileQueueExecutor)}
    requests = {}
    results = []

    def incomplete():
        return results + [_infra("Check batch cancelled after an infrastructure failure.")
                          for _ in range(len(checks) - len(results))]

    try:
        for backend in queued:
            if not backend.preflight()["passed"]:
                return incomplete()
        for index, (backend, candidate, bundle) in enumerate(checks):
            if isinstance(backend, FileQueueExecutor):
                requests[index] = backend.submit(candidate, bundle)
        for index, (backend, candidate, bundle) in enumerate(checks):
            result = backend.wait(requests[index]) if index in requests else backend.run(candidate, bundle)
            results.append(result)
            if result.status == "infra_error":
                return incomplete()
        return results
    except (OSError, ValueError, TypeError):
        return incomplete()
    finally:
        for backend, existing in queued.items():
            for request_id in tuple(backend.pending - existing):
                try:
                    backend.cancel(request_id)
                except (OSError, ValueError, TypeError):
                    pass
                finally:
                    backend.pending.discard(request_id)
