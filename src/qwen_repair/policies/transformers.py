from hashlib import sha256
from pathlib import Path
import re
from time import perf_counter

from qwen_repair.schemas import Generation, GenerationConfig, Message


QWEN_MODEL_ID = "Qwen/Qwen3-0.6B"
QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def encode_messages(tokenizer, messages: list[Message]) -> list[int]:
    return list(tokenizer.apply_chat_template(
        [message.model_dump() for message in messages],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ))


def build_feedback_suffix(tokenizer, feedback: str, ends_with_eos: bool) -> list[int]:
    """Append framing without re-tokenizing any previously generated action."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": feedback}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    closing = "" if ends_with_eos else tokenizer.eos_token
    return list(tokenizer.encode(closing + "\n" + rendered, add_special_tokens=False))


class TokenConversation:
    def __init__(self, tokenizer, messages: list[Message]):
        self.tokenizer = tokenizer
        self.token_ids = encode_messages(tokenizer, messages)
        self.action_mask = [0] * len(self.token_ids)
        self._awaiting_feedback = False

    @classmethod
    def start(cls, tokenizer, messages: list[Message]) -> "TokenConversation":
        return cls(tokenizer, messages)

    def append_action(self, token_ids: list[int]) -> None:
        if self._awaiting_feedback:
            raise ValueError("Feedback must separate generated actions")
        self.token_ids.extend(token_ids)
        self.action_mask.extend([1] * len(token_ids))
        self._awaiting_feedback = True

    def append_feedback(self, feedback: str) -> list[int]:
        if not self._awaiting_feedback:
            raise ValueError("Feedback requires a preceding generated action")
        suffix = build_feedback_suffix(
            self.tokenizer, feedback,
            bool(self.token_ids and self.token_ids[-1] == self.tokenizer.eos_token_id),
        )
        self.token_ids.extend(suffix)
        self.action_mask.extend([0] * len(suffix))
        self._awaiting_feedback = False
        return suffix


class TransformersPolicy:
    synthetic_demo = False
    token_count_method = "tokenizer"

    def __init__(
        self, model, tokenizer, *,
        model_revision: str = QWEN_REVISION,
        tokenizer_revision: str = QWEN_REVISION,
        policy_version: str | None = None,
        seed: int = 42,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        max_context_tokens: int = 4096,
        torch_module=None,
    ):
        if temperature <= 0 or not 0 < top_p <= 1 or top_k < 0:
            raise ValueError("Invalid decoding settings")
        if max_context_tokens <= 0 or seed < 0:
            raise ValueError("Context limit must be positive and seed nonnegative")
        self.model = model
        self.tokenizer = tokenizer
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.policy_version = policy_version or f"{QWEN_MODEL_ID}@{model_revision}"
        self.seed = seed
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        self._torch = torch_module
        self._generation_index = 0
        self.model.eval()
        self.begin_episode()

    @classmethod
    def from_pretrained(
        cls, *, model_id: str = QWEN_MODEL_ID,
        model_revision: str = QWEN_REVISION,
        tokenizer_revision: str = QWEN_REVISION,
        device: str = "cuda", dtype: str = "auto",
        adapter_path: str | Path | None = None,
        policy_version: str | None = None,
        **kwargs,
    ) -> "TransformersPolicy":
        if model_id != QWEN_MODEL_ID:
            raise ValueError("M1 supports only the specified Qwen3-0.6B model")
        for revision in (model_revision, tokenizer_revision):
            if re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
                raise ValueError("Model and tokenizer revisions must be full commit SHAs")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run GPU preflight in the selected runtime")
        if dtype == "auto":
            dtype = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if device.startswith("cuda") else "float32"
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be auto, float32, float16 or bfloat16")
        if dtype == "bfloat16" and device.startswith("cuda") and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected GPU does not support the requested bfloat16 dtype")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=tokenizer_revision, trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=model_revision, dtype=getattr(torch, dtype), trust_remote_code=False,
        ).to(device)
        if adapter_path is not None:
            from peft import PeftModel

            adapter_dir = Path(adapter_path)
            if not adapter_dir.is_dir():
                raise ValueError("adapter_path must name an existing local adapter directory")
            adapter_files = sorted(
                path for path in adapter_dir.iterdir()
                if path.name == "adapter_config.json" or path.suffix in {".safetensors", ".bin"}
            )
            if not adapter_files:
                raise ValueError("Adapter directory contains no adapter files")
            digest = sha256()
            for path in adapter_files:
                digest.update(path.name.encode())
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
            policy_version = policy_version or f"{model_id}@{model_revision}+adapter:{digest.hexdigest()}"
        return cls(
            model, tokenizer, model_revision=model_revision,
            tokenizer_revision=tokenizer_revision, policy_version=policy_version,
            torch_module=torch, **kwargs,
        )

    def begin_episode(self) -> None:
        self._conversation = None
        self._previous_messages = None
        self._previous_text = None
        self._prepared_messages = None
        self._prepared_ids = None

    def _prepare_prompt(self, messages: list[Message]) -> list[int]:
        if messages == self._prepared_messages:
            return list(self._prepared_ids)
        if self._conversation is None:
            self._conversation = TokenConversation.start(self.tokenizer, messages)
        elif (
            self._previous_messages is not None
            and messages[:-2] == self._previous_messages
            and len(messages) == len(self._previous_messages) + 2
            and messages[-2].role == "assistant"
            and messages[-2].content == self._previous_text
            and messages[-1].role == "user"
        ):
            self._conversation.append_feedback(messages[-1].content)
        else:
            raise ValueError("Message history changed; call begin_episode for a new episode")
        self._prepared_messages = list(messages)
        self._prepared_ids = list(self._conversation.token_ids)
        return list(self._prepared_ids)

    def count_messages(self, messages: list[Message]) -> int:
        return len(self._prepare_prompt(messages))

    def count_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate_feedback(self, text: str, max_tokens: int) -> str:
        if max_tokens < 1:
            return ""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        shortened = self.tokenizer.decode(ids[:max_tokens], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
        while shortened and self.count_text(shortened) > max_tokens:
            shortened = shortened[:-1]
        return shortened

    def generate(self, messages: list[Message], generation_config: GenerationConfig) -> Generation:
        prompt_ids = self._prepare_prompt(messages)
        if len(prompt_ids) + generation_config.max_new_tokens > self.max_context_tokens:
            raise ValueError("Prompt and output reservation exceed the model context budget")
        if self._torch is None:
            import torch
            self._torch = torch
        torch = self._torch
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=self.model.device)
        kwargs = {
            "max_new_tokens": generation_config.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if self.do_sample:
            kwargs.update(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        device = self.model.device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        started = perf_counter()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed + self._generation_index)
            with torch.inference_mode():
                output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **kwargs)
        self._generation_index += 1
        all_ids = output[0].tolist()
        if all_ids[:len(prompt_ids)] != prompt_ids:
            raise RuntimeError("Model generation changed its input token prefix")
        completion_ids = all_ids[len(prompt_ids):]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
        self._conversation.append_action(completion_ids)
        self._previous_messages = list(messages)
        self._previous_text = text
        return Generation(
            text=text, input_tokens=len(prompt_ids), output_tokens=len(completion_ids),
            policy_version=self.policy_version, prompt_token_ids=prompt_ids,
            completion_token_ids=completion_ids, token_count_method="tokenizer",
            duration_seconds=perf_counter() - started, synthetic_demo=False,
        )
