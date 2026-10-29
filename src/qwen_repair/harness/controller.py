from time import perf_counter
from typing import Protocol

from qwen_repair.schemas import (
    Attempt,
    Generation,
    GenerationConfig,
    Message,
    PromptView,
    TaskRecord,
    TestResult,
    Trajectory,
    count_message_tokens,
    count_tokens,
)

from .feedback import format_feedback, format_parse_error, sanitize_diagnostic, truncate_to_budget
from .parser import CandidateError, parse_candidate
from .prompts import build_initial_messages


class EpisodeConfig(Protocol):
    max_attempts: int
    max_new_tokens: int
    max_context_tokens: int
    max_feedback_tokens: int


class Policy(Protocol):
    policy_version: str

    def generate(self, messages: list[Message], generation_config: GenerationConfig) -> Generation: ...


class Executor(Protocol):
    def run(self, candidate: str, test_bundle_id: str) -> TestResult: ...


class EpisodeController:
    def __init__(self, config: EpisodeConfig):
        self.config = config

    def run(
        self,
        task: TaskRecord,
        policy: Policy,
        executor: Executor,
        *,
        run_id: str,
        episode_id: str,
        scenario: str,
    ) -> Trajectory:
        messages = build_initial_messages(PromptView.from_task(task))
        policy_version = policy.policy_version
        attempts = []
        stop_reason = "budget_exhausted"
        generation_config = GenerationConfig(max_new_tokens=self.config.max_new_tokens)
        for number in range(1, self.config.max_attempts + 1):
            input_tokens = count_message_tokens(messages)
            if input_tokens + self.config.max_new_tokens > self.config.max_context_tokens:
                stop_reason = "context_budget_exceeded"
                break
            started = perf_counter()
            generated = policy.generate(list(messages), generation_config)
            if generated.policy_version != policy_version or policy.policy_version != policy_version:
                raise ValueError("Policy changed version within an episode")
            output_tokens = count_tokens(generated.text)
            if generated.input_tokens != input_tokens or generated.output_tokens != output_tokens:
                raise ValueError("Policy counts must match the M0 UTF-8 byte budget")
            code = None
            parse_error = None
            result = None
            try:
                if output_tokens > self.config.max_new_tokens:
                    raise CandidateError("output_budget_exceeded", "Response exceeded the output budget.")
                code = parse_candidate(generated.text, task.entry_point, task.signature)
            except CandidateError as error:
                parse_error = error.code
                feedback = format_parse_error(error, self.config.max_feedback_tokens)
            else:
                result = executor.run(code, task.public_tests_ref)
                result = result.model_copy(update={
                    "feedback": truncate_to_budget(
                        sanitize_diagnostic(result.feedback), self.config.max_feedback_tokens
                    )
                })
                feedback = format_feedback(result, self.config.max_feedback_tokens)

            attempts.append(Attempt(
                number=number,
                response=generated.text,
                code=code,
                parse_error=parse_error,
                public_result=result,
                public_feedback=feedback,
                messages=list(messages),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_seconds=perf_counter() - started,
            ))
            if result is not None and result.status == "infra_error":
                stop_reason = "infra_error"
                break
            messages.extend([
                Message(role="assistant", content=generated.text),
                Message(role="user", content=feedback),
            ])

        return Trajectory(
            run_id=run_id,
            episode_id=episode_id,
            task_id=task.task_id,
            group_id=task.group_id,
            scenario=scenario,
            policy_version=policy_version,
            max_attempts=self.config.max_attempts,
            attempts=attempts,
            stop_reason=stop_reason,
        )
