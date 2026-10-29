from pathlib import Path

from qwen_repair.artifacts import write_json
from qwen_repair.data.fixtures import load_fixture
from qwen_repair.harness.feedback import format_feedback
from qwen_repair.harness.prompts import build_initial_messages
from qwen_repair.policies.transformers import TokenConversation
from qwen_repair.schemas import PromptView

from .common import (
    add_lora, begin_run, checked_executor, checkpoint_callback, fail_run,
    finish_run, load_base, require_real_result,
    record_executors, seal_training_run,
)


def bounded_feedback(tokenizer, result, max_tokens: int) -> str:
    text = format_feedback(result, max_tokens=100_000)
    return bounded_text(tokenizer, text, max_tokens)


def bounded_text(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    text = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    while text and len(tokenizer.encode(text, add_special_tokens=False)) > max_tokens:
        text = text[:-1]
    return text


def supervised_record(tokenizer, prompt_ids: list[int], target: str, max_context_tokens: int) -> dict:
    target_ids = list(tokenizer.encode(target, add_special_tokens=False)) + [tokenizer.eos_token_id]
    if len(prompt_ids) + len(target_ids) > max_context_tokens:
        raise ValueError("SFT example exceeds the context budget; no implicit truncation is allowed")
    return {
        "input_ids": prompt_ids + target_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def build_sft_records(tokenizer, fixtures, executors, config):
    records, audit = [], []
    for fixture, executor in zip(fixtures, executors, strict=True):
        task = fixture.task
        for bundle in (task.public_tests_ref, task.private_tests_ref):
            reference_result = require_real_result(executor.run(fixture.reference_code, bundle))
            if reference_result.status != "passed":
                raise ValueError(f"Authored reference does not pass its trusted tests: {task.task_id}")
            result = require_real_result(executor.run(fixture.corrected_code, bundle))
            if result.status != "passed":
                raise ValueError(f"Authored target does not pass its trusted tests: {task.task_id}")
        conversation = TokenConversation.start(tokenizer, build_initial_messages(PromptView.from_task(task)))
        target = f"```python\n{fixture.corrected_code}\n```"
        one_shot = supervised_record(tokenizer, list(conversation.token_ids), target, config.max_context_tokens)
        records.append(one_shot)
        audit.append({"task_id": task.task_id, "kind": "one_shot", **one_shot})
        if config.include_recovery:
            result = require_real_result(executor.run(task.buggy_code, task.public_tests_ref))
            if result.status == "passed":
                raise ValueError("Recovery supervision requires an actually failing public attempt")
            bad_answer = f"```python\n{task.buggy_code}\n```"
            conversation.append_action(list(tokenizer.encode(bad_answer, add_special_tokens=False)) + [tokenizer.eos_token_id])
            feedback = bounded_feedback(tokenizer, result, config.max_feedback_tokens)
            conversation.append_feedback(feedback)
            recovery = supervised_record(tokenizer, list(conversation.token_ids), target, config.max_context_tokens)
            records.append(recovery)
            audit.append({
                "task_id": task.task_id, "kind": "recovery", "public_feedback": feedback,
                "public_result": result.model_dump(), **recovery,
            })
    return records, audit


class SupervisedCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        import torch

        width = max(len(item["input_ids"]) for item in examples)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in examples:
            pad = width - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
            batch["labels"].append(item["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def make_sft_trainer(model, tokenizer, records, config, output: Path, dtype_name="float32", *, allow_test_checkpoint=False):
    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(
        output_dir=str(output), max_steps=config.max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        learning_rate=config.learning_rate, seed=config.seed, data_seed=config.seed,
        save_strategy="steps", save_steps=1, save_total_limit=2,
        logging_steps=1, report_to=[], disable_tqdm=True,
        use_cpu=config.device == "cpu", bf16=dtype_name == "bfloat16", fp16=dtype_name == "float16",
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False, remove_unused_columns=False,
        optim="adamw_torch",
    )
    callback = checkpoint_callback(config, allow_test_checkpoint=allow_test_checkpoint)
    trainer = Trainer(
        model=model, args=args, train_dataset=records, processing_class=tokenizer,
        data_collator=SupervisedCollator(tokenizer.pad_token_id),
        callbacks=[callback],
    )
    callback.scaler_getter = lambda: getattr(trainer.accelerator, "scaler", None) is not None
    return trainer


def verify_adapter_roundtrip(model, adapter_path: Path, probe_ids: list[int]) -> dict:
    import torch

    original_mode = model.training
    model.eval()
    inputs = torch.tensor([probe_ids], device=model.device, dtype=torch.long)
    with torch.no_grad():
        original = model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits.detach().float()
    model.load_adapter(str(adapter_path), adapter_name="roundtrip", is_trainable=False)
    model.set_adapter("roundtrip")
    try:
        with torch.no_grad():
            restored = model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits.detach().float()
        maximum = (original - restored).abs().max().item()
        if not torch.allclose(original, restored, rtol=1e-4, atol=1e-5):
            raise RuntimeError("Saved LoRA adapter changed probe logits after reloading")
        return {"adapter_roundtrip_verified": True, "adapter_roundtrip_max_abs_diff": maximum}
    finally:
        model.set_adapter("default")
        model.delete_adapter("roundtrip")
        model.train(original_mode)


def run_sft_smoke(config, output, resume_from=None, executor=None):
    output = Path(output)
    started = begin_run(config, output, resume_from, "sft")
    try:
        fixtures = [load_fixture(name) for name in config.fixture_names]
        executors = [checked_executor(executor, fixture.task) for fixture in fixtures]
        record_executors(output, fixtures, executors)
        model, tokenizer, dtype_name = load_base(config)
        model = add_lora(model, config.lora_rank)
        records, audit = build_sft_records(tokenizer, fixtures, executors, config)
        write_json(output / "supervision_audit.json", {"synthetic_demo": False, "examples": audit})
        trainer = make_sft_trainer(model, tokenizer, records, config, output, dtype_name)
        trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
        result = finish_run(config, output, trainer, started, dtype_name, supervised_examples=len(records))
        result.update(verify_adapter_roundtrip(model, output / "adapter", records[0]["input_ids"][:32]))
        write_json(output / "training_result.json", result)
        seal_training_run(output, result)
        return result
    except Exception as error:
        fail_run(output, error)
        raise
