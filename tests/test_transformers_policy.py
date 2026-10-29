from contextlib import nullcontext
import subprocess
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from qwen_repair.config import DemoConfig
from qwen_repair.data.fixtures import load_fixture
from qwen_repair.harness.controller import EpisodeController
from qwen_repair.policies.transformers import (
    QWEN_MODEL_ID,
    QWEN_REVISION,
    TokenConversation,
    TransformersPolicy,
    build_feedback_suffix,
)
from qwen_repair.schemas import Generation, GenerationConfig, Message, TestResult


class TinyTokenizer:
    eos_token = "<|im_end|>"
    eos_token_id = 2_000_001
    pad_token_id = 2_000_002
    alias_id = 2_000_003

    def __init__(self):
        self.template_calls = []

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        ids = []
        while text:
            if text.startswith(self.eos_token):
                ids.append(self.eos_token_id)
                text = text[len(self.eos_token):]
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return ids

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        pieces = []
        for token in ids:
            if token == self.eos_token_id:
                pieces.append("" if skip_special_tokens else self.eos_token)
            elif token == self.alias_id:
                pieces.append("alias")
            else:
                pieces.append(chr(token))
        return "".join(pieces)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking, return_dict=False):
        assert add_generation_prompt is True
        assert enable_thinking is False
        assert return_dict is False
        self.template_calls.append(messages)
        text = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}{self.eos_token}\n"
            for message in messages
        ) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return self.encode(text) if tokenize else text


class TinyTensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return TinyTensor(self.values[index])

    def tolist(self):
        return self.values


class TinyTorch:
    long = "long"
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"

    def __init__(self):
        self.seeds = []
        self.random = SimpleNamespace(fork_rng=lambda devices: nullcontext())
        self.cuda = SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False)

    def manual_seed(self, seed):
        self.seeds.append(seed)

    def inference_mode(self):
        return nullcontext()

    def tensor(self, values, **kwargs):
        return TinyTensor(values)

    def ones_like(self, tensor):
        return TinyTensor([[1] * len(row) for row in tensor.values])


class TinyModel:
    def __init__(self, completions):
        self.completions = iter(completions)
        self.device = SimpleNamespace(type="cpu", index=None)
        self.calls = []

    def eval(self):
        return self

    def to(self, device):
        self.device = SimpleNamespace(type=device, index=None)
        return self

    def generate(self, *, input_ids, attention_mask, **kwargs):
        self.calls.append(kwargs)
        return TinyTensor([input_ids.tolist()[0] + next(self.completions)])


def policy_with_responses(responses):
    tokenizer = TinyTokenizer()
    torch = TinyTorch()
    model = TinyModel([tokenizer.encode(text) + [tokenizer.eos_token_id] for text in responses])
    policy = TransformersPolicy(model, tokenizer, torch_module=torch)
    return policy, tokenizer, torch


def test_import_is_lazy_and_does_not_require_gpu_packages():
    script = "import sys; import qwen_repair.policies.transformers; assert not {'torch', 'transformers', 'peft'} & sys.modules.keys()"
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_generation_records_exact_ids_and_disables_thinking():
    policy, tokenizer, torch = policy_with_responses(["é"])
    messages = [Message(role="user", content="Исправь функцию.")]
    count = policy.count_messages(messages)
    generation = policy.generate(messages, GenerationConfig(max_new_tokens=16))
    assert generation.input_tokens == count == len(generation.prompt_token_ids)
    assert generation.output_tokens == len(generation.completion_token_ids) == 2
    assert generation.completion_token_ids == [ord("é"), tokenizer.eos_token_id]
    assert generation.text == "é"
    assert generation.token_count_method == "tokenizer"
    assert generation.synthetic_demo is False
    assert torch.seeds == [42]
    assert policy.model.calls[0]["temperature"] == 0.7
    assert policy.model.calls[0]["eos_token_id"] == tokenizer.eos_token_id
    assert tokenizer.template_calls


def test_second_prompt_preserves_original_ids_even_when_decode_is_not_roundtrip():
    tokenizer = TinyTokenizer()
    torch = TinyTorch()
    first_ids = [tokenizer.alias_id, tokenizer.eos_token_id]
    model = TinyModel([first_ids, tokenizer.encode("done") + [tokenizer.eos_token_id]])
    policy = TransformersPolicy(model, tokenizer, torch_module=torch)
    messages = [Message(role="user", content="repair")]
    first = policy.generate(messages, GenerationConfig(max_new_tokens=32))
    messages.extend([Message(role="assistant", content=first.text), Message(role="user", content="try again")])
    second = policy.generate(messages, GenerationConfig(max_new_tokens=32))
    prefix = first.prompt_token_ids + first.completion_token_ids
    assert second.prompt_token_ids[:len(prefix)] == prefix
    assert tokenizer.alias_id in second.prompt_token_ids
    assert torch.seeds == [42, 43]


@pytest.mark.parametrize("with_eos", [False, True])
def test_feedback_mask_preserves_actions_and_marks_only_appended_environment(with_eos):
    tokenizer = TinyTokenizer()
    conversation = TokenConversation.start(tokenizer, [Message(role="user", content="repair")])
    initial = list(conversation.token_ids)
    action = [tokenizer.alias_id] + ([tokenizer.eos_token_id] if with_eos else [])
    conversation.append_action(action)
    suffix = conversation.append_feedback("failed")
    assert conversation.token_ids == initial + action + suffix
    assert conversation.action_mask == [0] * len(initial) + [1] * len(action) + [0] * len(suffix)
    assert suffix.count(tokenizer.eos_token_id) == (1 if with_eos else 2)
    assert suffix == build_feedback_suffix(tokenizer, "failed", with_eos)


def test_context_overflow_is_rejected_without_generation_or_truncation():
    policy, _, _ = policy_with_responses(["unused"])
    messages = [Message(role="user", content="x" * 300)]
    prompt_count = policy.count_messages(messages)
    policy.max_context_tokens = prompt_count + 7
    with pytest.raises(ValueError, match="context budget"):
        policy.generate(messages, GenerationConfig(max_new_tokens=8))
    assert policy.model.calls == []


def test_feedback_is_bounded_by_tokenizer_instead_of_utf8_bytes():
    policy, _, _ = policy_with_responses([])
    assert policy.truncate_feedback("é" * 10, 3) == "é" * 3
    assert policy.count_text(policy.truncate_feedback("Привет", 4)) == 4


def test_real_controller_keeps_fixed_turns_and_exact_token_prefix():
    fixture = load_fixture("sum_positive")
    good = f"```python\n{fixture.corrected_code}\n```"
    policy, _, _ = policy_with_responses([good, good])

    class RealExecutor:
        synthetic_demo = False

        def run(self, candidate, test_bundle_id):
            return TestResult(status="passed", tests_passed=2, tests_total=2, synthetic_demo=False)

    trajectory = EpisodeController(DemoConfig()).run(
        fixture.task, policy, RealExecutor(), run_id="real", episode_id="one", scenario="inference",
    )
    assert len(trajectory.attempts) == 2
    assert trajectory.synthetic_demo is False
    assert trajectory.token_count_method == "tokenizer"
    first, second = trajectory.attempts
    prefix = first.prompt_token_ids + first.completion_token_ids
    assert second.prompt_token_ids[:len(prefix)] == prefix
    assert first.public_result.synthetic_demo is False


def test_loader_pins_both_revisions_and_disables_remote_code(monkeypatch):
    tokenizer = TinyTokenizer()
    model = TinyModel([])
    calls = []

    class TokenizerFactory:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("tokenizer", name, kwargs))
            return tokenizer

    class ModelFactory:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("model", name, kwargs))
            return model

    monkeypatch.setitem(sys.modules, "torch", TinyTorch())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=TokenizerFactory, AutoModelForCausalLM=ModelFactory,
    ))
    policy = TransformersPolicy.from_pretrained(device="cpu")
    assert policy.tokenizer_revision == policy.model_revision == QWEN_REVISION
    assert len(calls) == 2
    for _, name, kwargs in calls:
        assert name == QWEN_MODEL_ID
        assert kwargs["revision"] == QWEN_REVISION
        assert kwargs["trust_remote_code"] is False


def test_loader_rejects_unpinned_revisions_before_optional_imports():
    with pytest.raises(ValueError, match="commit SHAs"):
        TransformersPolicy.from_pretrained(model_revision="main")


def test_tokenizer_generation_requires_ids_matching_counts():
    fields = dict(text="answer", input_tokens=2, output_tokens=1, policy_version="v1", token_count_method="tokenizer")
    with pytest.raises(ValidationError, match="exact prompt"):
        Generation(**fields)
    with pytest.raises(ValidationError, match="recorded counts"):
        Generation(**fields, prompt_token_ids=[1], completion_token_ids=[2])
