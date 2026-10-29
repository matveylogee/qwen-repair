from collections.abc import Sequence

from qwen_repair.schemas import (
    Generation,
    GenerationConfig,
    Message,
    count_message_tokens,
    count_tokens,
)


class ScriptedPolicy:
    """Replay supplied responses; token counts are the M0 UTF-8 byte proxy."""

    def __init__(
        self, responses: Sequence[str], policy_version: str = "scripted-v1"
    ) -> None:
        self._responses = tuple(responses)
        self._position = 0
        self.policy_version = policy_version
        self.calls: list[list[Message]] = []

    def generate(
        self, messages: list[Message], generation_config: GenerationConfig
    ) -> Generation:
        if self._position >= len(self._responses):
            raise RuntimeError("Scripted policy has no response for this attempt")
        self.calls.append([message.model_copy(deep=True) for message in messages])
        response = self._responses[self._position]
        self._position += 1
        # The controller records budget violations; truncating would alter the script.
        return Generation(
            text=response,
            input_tokens=count_message_tokens(messages),
            output_tokens=count_tokens(response),
            policy_version=self.policy_version,
        )
