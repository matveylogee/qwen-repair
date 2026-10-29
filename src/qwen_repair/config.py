from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DemoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"] = "1.0"
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    max_attempts: int = Field(default=2, ge=1, le=3)
    max_new_tokens: int = Field(default=512, ge=1)
    max_context_tokens: int = Field(default=4096, ge=1)
    max_feedback_tokens: int = Field(default=256, ge=1)
    scenarios: list[Literal[
        "recovery", "invalid_format", "repeated_error", "infra_failure", "regression"
    ]] = Field(default_factory=lambda: [
        "recovery", "invalid_format", "repeated_error", "infra_failure", "regression"
    ])

    @field_validator("scenarios")
    @classmethod
    def unique_scenarios(cls, value: list[str]) -> list[str]:
        if not value or len(set(value)) != len(value):
            raise ValueError("scenarios must be nonempty and unique")
        return value


def load_config(path: str | Path) -> DemoConfig:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise ValueError(f"Invalid YAML syntax{location}") from None
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    return DemoConfig.model_validate(data)
