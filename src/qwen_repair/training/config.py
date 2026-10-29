from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ML_VERSIONS = {
    "torch": "2.10.0", "transformers": "5.20.2", "peft": "0.21.2",
    "trl": "1.16.0", "accelerate": "1.11.0", "datasets": "4.7.0",
}


class SmokeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"] = "1.0"
    model_id: Literal["Qwen/Qwen3-0.6B"] = "Qwen/Qwen3-0.6B"
    model_revision: str = QWEN_REVISION
    device: Literal["cuda", "cpu"] = "cuda"
    dtype: Literal["auto", "float32", "bfloat16", "float16"] = "auto"
    fixture_names: list[Literal["sum_positive", "last_or_none", "count_even", "is_palindrome"]] = Field(
        default_factory=lambda: ["sum_positive", "last_or_none"]
    )
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    max_steps: int = Field(default=2, ge=1, le=20)
    stop_after_steps: int | None = Field(default=None, ge=1, le=20)
    learning_rate: float = Field(default=5e-5, gt=0, le=0.01)
    lora_rank: int = Field(default=8, ge=1, le=32)
    max_context_tokens: int = Field(default=4096, ge=64, le=4096)
    max_new_tokens: int = Field(default=512, ge=1, le=512)
    max_feedback_tokens: int = Field(default=256, ge=1, le=256)
    gradient_checkpointing: bool = True

    @field_validator("fixture_names")
    @classmethod
    def unique_fixtures(cls, value):
        if not value or len(value) != len(set(value)):
            raise ValueError("fixture_names must be nonempty and unique")
        return value

    @model_validator(mode="after")
    def valid_run(self):
        if self.stop_after_steps is not None and self.stop_after_steps > self.max_steps:
            raise ValueError("stop_after_steps cannot exceed the planned max_steps")
        if len(self.model_revision) != 40 or any(c not in "0123456789abcdef" for c in self.model_revision):
            raise ValueError("model_revision must be an immutable full commit SHA")
        return self


class SFTConfig(SmokeConfig):
    include_recovery: bool = True


class GRPOConfig(SmokeConfig):
    sft_adapter: str
    num_generations: Literal[2, 4] = 2
    max_attempts: Literal[2] = 2
    beta: Literal[0.0] = 0.0
