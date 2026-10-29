from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.0"
TOKEN_COUNT_METHOD = "utf8_bytes_proxy"
TokenCountMethod = Literal["utf8_bytes_proxy", "tokenizer"]


class VersionedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = SCHEMA_VERSION


class TaskRecord(VersionedModel):
    task_id: str
    source_id: str
    group_id: str
    split: Literal["train", "validation", "test", "demo"]
    statement: str
    signature: str
    entry_point: str
    buggy_code: str
    bug_type: str
    public_tests_ref: str
    private_tests_ref: str
    reference_ref: str


class PromptView(VersionedModel):
    task_id: str
    statement: str
    signature: str
    buggy_code: str
    entry_point: str

    @classmethod
    def from_task(cls, task: TaskRecord) -> "PromptView":
        return cls(
            task_id=task.task_id,
            statement=task.statement,
            signature=task.signature,
            buggy_code=task.buggy_code,
            entry_point=task.entry_point,
        )


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


def count_tokens(text: str) -> int:
    """M0 budget units are UTF-8 bytes, not a model tokenizer's tokens."""
    return len(text.encode("utf-8"))


def count_message_tokens(messages: list[Message]) -> int:
    return sum(count_tokens(f"{message.role}\n{message.content}\n") for message in messages)


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_new_tokens: int = Field(default=512, ge=1)


class Generation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    policy_version: str
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    token_count_method: TokenCountMethod = TOKEN_COUNT_METHOD
    duration_seconds: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    synthetic_demo: bool = True

    @model_validator(mode="after")
    def validate_token_ids(self) -> "Generation":
        _validate_token_ids(self.prompt_token_ids, self.completion_token_ids,
                            self.input_tokens, self.output_tokens, self.token_count_method)
        return self


def _validate_token_ids(prompt_ids, completion_ids, input_count, output_count, method):
    if method == "tokenizer" and (prompt_ids is None or completion_ids is None):
        raise ValueError("Tokenizer counts require exact prompt and completion token IDs")
    for ids, count in ((prompt_ids, input_count), (completion_ids, output_count)):
        if ids is not None and (len(ids) != count or any(token < 0 for token in ids)):
            raise ValueError("Token IDs must be nonnegative and match the recorded counts")


class TestResult(VersionedModel):
    __test__ = False

    status: Literal[
        "passed", "failed", "syntax_error", "runtime_error",
        "timeout", "resource_limit", "infra_error",
    ]
    tests_passed: int = Field(default=0, ge=0)
    tests_failed: int = Field(default=0, ge=0)
    tests_total: int = Field(default=0, ge=0)
    feedback: str = ""
    duration_seconds: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    synthetic_demo: bool = True

    @model_validator(mode="after")
    def validate_counts(self) -> "TestResult":
        if self.tests_passed + self.tests_failed > self.tests_total:
            raise ValueError("Completed test counts exceed tests_total")
        if self.status in {"passed", "failed"}:
            if self.tests_total == 0:
                raise ValueError("A completed verdict requires at least one test")
            if self.tests_passed + self.tests_failed != self.tests_total:
                raise ValueError("A completed verdict must account for every test")
        if self.status == "passed" and self.tests_failed:
            raise ValueError("A passing verdict cannot contain failed tests")
        if self.status == "failed" and not self.tests_failed:
            raise ValueError("A failed verdict requires a failed test")
        return self


class Attempt(VersionedModel):
    number: int = Field(ge=1)
    response: str
    code: str | None = None
    parse_error: str | None = None
    public_result: TestResult | None = None
    public_feedback: str
    messages: list[Message]
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    duration_seconds: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    token_count_method: TokenCountMethod = TOKEN_COUNT_METHOD
    generation_duration_seconds: float = Field(default=0.0, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_candidate_state(self) -> "Attempt":
        _validate_token_ids(self.prompt_token_ids, self.completion_token_ids,
                            self.input_tokens, self.output_tokens, self.token_count_method)
        if self.parse_error is not None:
            if not self.parse_error.strip():
                raise ValueError("A parse error must have a nonempty reason")
            if self.code is not None or self.public_result is not None:
                raise ValueError("An invalid candidate cannot have parsed code or an execution result")
        elif self.code is None or self.public_result is None:
            raise ValueError("A parsed candidate requires code and a public execution result")
        return self


class Trajectory(VersionedModel):
    run_id: str
    episode_id: str
    task_id: str
    group_id: str
    scenario: str
    policy_version: str
    max_attempts: int = Field(ge=1)
    attempts: list[Attempt]
    stop_reason: Literal["budget_exhausted", "infra_error", "context_budget_exceeded"]
    token_count_method: TokenCountMethod = TOKEN_COUNT_METHOD
    synthetic_demo: bool = True

    @model_validator(mode="after")
    def validate_attempt_order(self) -> "Trajectory":
        if any(attempt.token_count_method != self.token_count_method for attempt in self.attempts):
            raise ValueError("Attempt token counting methods must match the trajectory")
        if not self.synthetic_demo and any(
            attempt.public_result is not None and attempt.public_result.synthetic_demo
            for attempt in self.attempts
        ):
            raise ValueError("A real trajectory cannot contain synthetic public results")
        if [attempt.number for attempt in self.attempts] != list(range(1, len(self.attempts) + 1)):
            raise ValueError("Attempts must be contiguous and start at one")
        if len(self.attempts) > self.max_attempts:
            raise ValueError("Attempts exceed the declared budget")
        if self.stop_reason == "budget_exhausted" and len(self.attempts) != self.max_attempts:
            raise ValueError("A completed episode must consume its full attempt budget")
        return self


class PrivateEvaluation(VersionedModel):
    run_id: str
    episode_id: str
    task_id: str
    attempt_results: list[TestResult | None]
    terminal_reward: Literal[0, 1] | None
    excluded_reason: str | None = None
    synthetic_demo: bool = True

    @model_validator(mode="after")
    def validate_reward(self) -> "PrivateEvaluation":
        if not self.synthetic_demo and any(result is not None and result.synthetic_demo for result in self.attempt_results):
            raise ValueError("A real private evaluation cannot contain synthetic results")
        if self.excluded_reason is not None:
            if self.terminal_reward is not None:
                raise ValueError("Excluded episodes cannot carry a reward")
        elif self.terminal_reward is None:
            raise ValueError("An absent reward requires an exclusion reason")
        if any(result is not None and result.status == "infra_error" for result in self.attempt_results):
            if self.excluded_reason is None:
                raise ValueError("Infrastructure failures must exclude the episode")
        return self


class RunManifest(VersionedModel):
    run_id: str
    status: Literal["running", "completed", "completed_with_exclusions", "failed"]
    synthetic_demo: bool = True
    created_at: str
    completed_at: str | None = None
    package_version: str
    git_sha: str | None
    git_dirty: bool | None
    config: dict
    seed: int
    policy_version: str
    prompt_version: str
    runner_version: str
    token_count_method: TokenCountMethod = TOKEN_COUNT_METHOD
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    fixture_hashes: dict[str, str]
    source_hashes: dict[str, str]
    dataset_hashes: dict[str, str] = Field(default_factory=dict)
    executor_config: dict = Field(default_factory=dict)
    performance: dict = Field(default_factory=dict)
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    error_type: str | None = None
