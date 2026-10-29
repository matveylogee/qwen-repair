import os

import pytest

from qwen_repair.data.fixtures import load_fixture
from qwen_repair.harness.prompts import build_initial_messages
from qwen_repair.policies.transformers import QWEN_MODEL_ID, QWEN_REVISION, TokenConversation
from qwen_repair.schemas import Message, PromptView


def test_pinned_qwen_tokenizer_preserves_actions_and_inserts_one_feedback_turn():
    cache = os.environ.get("QREPAIR_TOKENIZER_CACHE")
    if not cache:
        pytest.skip("Set QREPAIR_TOKENIZER_CACHE to an already downloaded pinned tokenizer; this test never downloads")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        QWEN_MODEL_ID, revision=QWEN_REVISION, cache_dir=cache,
        local_files_only=True, trust_remote_code=False,
    )
    fixture = load_fixture("sum_positive")
    messages = build_initial_messages(PromptView.from_task(fixture.task))
    conversation = TokenConversation.start(tokenizer, messages)
    initial = list(conversation.token_ids)
    response = f"```python\n{fixture.corrected_code}\n```"
    action = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
    conversation.append_action(action)
    feedback = "Public status: passed; passed 3/3."
    suffix = conversation.append_feedback(feedback)
    assert conversation.token_ids[:len(initial + action)] == initial + action
    assert conversation.action_mask == [0] * len(initial) + [1] * len(action) + [0] * len(suffix)
    decoded = tokenizer.decode(conversation.token_ids, skip_special_tokens=False)
    assert decoded.count("<|im_start|>system\n") == 1
    assert decoded.count("<|im_start|>user\n") == 2
    assert decoded.count("<|im_start|>assistant\n") == 2
    assert decoded.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert tokenizer.decode(suffix, skip_special_tokens=False).startswith("\n<|im_start|>user\n")
    retemplated = tokenizer.apply_chat_template(
        [message.model_dump() for message in messages] + [
            Message(role="assistant", content=response).model_dump(),
            Message(role="user", content=feedback).model_dump(),
        ],
        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False,
    )
    # Qwen removes the old empty thinking prefix when it re-renders history.
    assert list(retemplated)[:len(initial + action)] != initial + action
