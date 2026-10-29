from qwen_repair.schemas import PrivateEvaluation, TaskRecord, Trajectory
from qwen_repair.harness.controller import Executor


def evaluate_private(
    trajectory: Trajectory, task: TaskRecord, executor: Executor
) -> PrivateEvaluation:
    if trajectory.task_id != task.task_id:
        raise ValueError("Task ID does not match trajectory")
    results = []
    excluded = None
    if trajectory.stop_reason != "budget_exhausted":
        excluded = trajectory.stop_reason
        results = [None] * len(trajectory.attempts)
    else:
        for attempt in trajectory.attempts:
            result = (
                executor.run(attempt.code, task.private_tests_ref)
                if attempt.code is not None and attempt.parse_error is None else None
            )
            results.append(result)
        if any(result is not None and result.status == "infra_error" for result in results):
            excluded = "private_infra_error"
    final_passed = bool(results and results[-1] is not None and results[-1].status == "passed")
    return PrivateEvaluation(
        run_id=trajectory.run_id,
        episode_id=trajectory.episode_id,
        task_id=trajectory.task_id,
        attempt_results=results,
        terminal_reward=None if excluded else int(final_passed),
        excluded_reason=excluded,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(
    trajectories: list[Trajectory], evaluations: list[PrivateEvaluation]
) -> dict:
    if not trajectories:
        raise ValueError("No trajectories to summarize")
    by_id = {(item.run_id, item.episode_id): item for item in evaluations}
    trajectory_ids = {(item.run_id, item.episode_id) for item in trajectories}
    if len(by_id) != len(evaluations) or len(trajectory_ids) != len(trajectories):
        raise ValueError("Duplicate episode IDs")
    if trajectory_ids != set(by_id):
        raise ValueError("Private evaluation and trajectory IDs must match exactly")
    if len({item.run_id for item in trajectories}) != 1:
        raise ValueError("Summarize one run at a time")

    successes = []
    eligible_attempts = []
    excluded_reasons: dict[str, int] = {}
    for trajectory in trajectories:
        evaluation = by_id[(trajectory.run_id, trajectory.episode_id)]
        if evaluation.task_id != trajectory.task_id:
            raise ValueError("Private evaluation task ID mismatch")
        if len(evaluation.attempt_results) != len(trajectory.attempts):
            raise ValueError("Private evaluation attempt count mismatch")
        if trajectory.stop_reason != "budget_exhausted" and not evaluation.excluded_reason:
            raise ValueError("Incomplete episodes must be excluded")
        if evaluation.excluded_reason:
            reason = evaluation.excluded_reason
            excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
            continue
        values = []
        for attempt, result in zip(trajectory.attempts, evaluation.attempt_results, strict=True):
            if attempt.public_result is not None and attempt.public_result.status == "infra_error":
                raise ValueError("Public infrastructure failure must exclude the episode")
            invalid = attempt.parse_error is not None or attempt.code is None
            if invalid != (result is None):
                raise ValueError("Only invalid candidates may omit private results")
            values.append(result is not None and result.status == "passed")
        if evaluation.terminal_reward != int(values[-1]):
            raise ValueError("Terminal reward must reflect the last candidate")
        successes.append(values)
        eligible_attempts.extend(trajectory.attempts)

    eligible = len(successes)
    final_successes = sum(values[-1] for values in successes)
    recovery_base = [values for values in successes if not values[0]]
    regression_base = [values for values in successes if values[0]]
    attempt_denominators = {
        str(index + 1): sum(len(values) > index for values in successes)
        for index in range(max(item.max_attempts for item in trajectories))
    }
    attempt_success = {}
    solved_by_attempt = {}
    for key, denominator in attempt_denominators.items():
        index = int(key) - 1
        attempt_success[key] = _rate(
            sum(values[index] for values in successes if len(values) > index), denominator
        )
        solved_by_attempt[key] = _rate(
            sum(any(values[:index + 1]) for values in successes if len(values) > index), denominator
        )
    all_attempts = [attempt for item in trajectories for attempt in item.attempts]
    return {
        "schema_version": "1.0",
        "synthetic_demo": True,
        "aggregation": "demo_episode",
        "total_episodes": len(trajectories),
        "eligible_episodes": eligible,
        "excluded_episodes": len(trajectories) - eligible,
        "excluded_reasons": excluded_reasons,
        "infrastructure_error_episodes": sum(
            value for key, value in excluded_reasons.items() if "infra_error" in key
        ),
        "context_budget_exceeded_episodes": excluded_reasons.get("context_budget_exceeded", 0),
        "final_successes": final_successes,
        "final_success_rate": _rate(final_successes, eligible),
        "attempt_success": attempt_success,
        "solved_by_attempt": solved_by_attempt,
        "attempt_denominators": attempt_denominators,
        "recovery_denominator": len(recovery_base),
        "regression_denominator": len(regression_base),
        "recovery_rate": _rate(sum(values[-1] for values in recovery_base), len(recovery_base)),
        "regression_rate": _rate(sum(not values[-1] for values in regression_base), len(regression_base)),
        "invalid_candidate_rate": _rate(
            sum(attempt.parse_error is not None for attempt in eligible_attempts),
            len(eligible_attempts),
        ),
        "total_attempts": len(all_attempts),
        "public_executor_calls": sum(attempt.public_result is not None for attempt in all_attempts),
        "private_executor_calls": sum(
            result is not None for evaluation in evaluations for result in evaluation.attempt_results
        ),
        "input_tokens": sum(attempt.input_tokens for attempt in all_attempts),
        "output_tokens": sum(attempt.output_tokens for attempt in all_attempts),
        "token_count_method": "utf8_bytes_proxy",
    }
