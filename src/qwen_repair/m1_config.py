from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from qwen_repair.policies.transformers import QWEN_MODEL_ID, QWEN_REVISION


class ExecutorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["docker", "file_queue"]
    image: str | None = None
    queue_dir: str | None = None
    deadline_seconds: float = Field(default=120, gt=0, le=3600)
    poll_interval: float = Field(default=0.5, ge=0.05, le=10)
    timeout_seconds: float = Field(default=5, gt=0, le=60)

    @model_validator(mode="after")
    def backend_fields(self):
        if not self.image:
            raise ValueError("Executor profile requires the worker image pinned by digest")
        if self.backend == "file_queue" and not self.queue_dir:
            raise ValueError("File queue executor requires queue_dir")
        return self


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["1.0"] = "1.0"
    model_id: Literal["Qwen/Qwen3-0.6B"] = QWEN_MODEL_ID
    model_revision: str = QWEN_REVISION
    tokenizer_revision: str = QWEN_REVISION
    device: Literal["cuda", "cpu"] = "cuda"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    seed: int = Field(default=42, ge=0)
    max_attempts: Literal[2, 3] = 2
    max_new_tokens: int = Field(default=512, ge=1, le=512)
    max_context_tokens: int = Field(default=4096, ge=64, le=4096)
    max_feedback_tokens: int = Field(default=256, ge=1, le=256)
    do_sample: bool = True
    temperature: float = Field(default=0.7, gt=0)
    top_p: float = Field(default=0.8, gt=0, le=1)
    top_k: int = Field(default=20, ge=0)
    fixture_names: list[str] = Field(default_factory=lambda: ["sum_positive", "last_or_none"])
    pilot_dir: str | None = None
    adapter_path: str | None = None
    max_tasks: int = Field(default=2, ge=1, le=100)


def read_yaml_model(path: Path, model):
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        raise ValueError("Invalid YAML configuration") from None
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return model.model_validate(payload)
