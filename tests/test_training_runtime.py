import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch", reason="Optional tiny CPU integration needs the pinned GPU extra")
pytest.importorskip("trl")


def tiny_model():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, set_seed

    from qwen_repair.data.fixtures import load_fixture
    from qwen_repair.training.common import add_lora

    set_seed(42)
    torch.set_num_threads(1)
    fixture = load_fixture("sum_positive")
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        [fixture.task.statement, fixture.task.buggy_code, fixture.corrected_code, "```python\n\n```", "Public status: failed; passed 0/1."],
        trainers.BpeTrainer(vocab_size=384, special_tokens=["<unk>", "<pad>", "<|im_start|>", "<|im_end|>"],
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>",
        eos_token="<|im_end|>", padding_side="left",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=2048, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, tie_word_embeddings=True,
    ))
    return add_lora(model, rank=2), tokenizer, fixture


def child_sft(output, resume=False):
    from qwen_repair.training.common import verify_checkpoint
    from qwen_repair.training.config import SFTConfig
    from qwen_repair.training.sft import make_sft_trainer, supervised_record, verify_adapter_roundtrip

    output = Path(output)
    output.mkdir(exist_ok=True)
    model, tokenizer, fixture = tiny_model()
    config = SFTConfig(device="cpu", gradient_checkpointing=False, lora_rank=2,
                       max_steps=2, stop_after_steps=None if resume else 1)
    records = [supervised_record(tokenizer, tokenizer.encode("Repair the function:\n"),
                                 f"```python\n{fixture.corrected_code}\n```", 512)] * 2
    trainer = make_sft_trainer(model, tokenizer, records, config, output, allow_test_checkpoint=True)
    checkpoint = output / "checkpoint-1" if resume else None
    if checkpoint:
        verify_checkpoint(checkpoint, config, allow_test_checkpoint=True)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    trainer.save_model(str(output / "adapter"))
    evidence = verify_adapter_roundtrip(model, output / "adapter", records[0]["input_ids"][:10])
    (output / "child_result.json").write_text(json.dumps({"step": trainer.state.global_step, **evidence}))


def test_tiny_cpu_sft_checkpoint_resume_in_new_process_and_adapter_roundtrip(tmp_path):
    test_file = str(Path(__file__).resolve())
    code = "import runpy,sys; mod=runpy.run_path(sys.argv[1]); mod['child_sft'](sys.argv[2], sys.argv[3]=='resume')"
    run_dir = tmp_path / "sft"
    for stage, expected_step in (("initial", 1), ("resume", 2)):
        result = subprocess.run([sys.executable, "-c", code, test_file, str(run_dir), stage],
                                capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stderr + result.stdout
        evidence = json.loads((run_dir / "child_result.json").read_text())
        assert evidence["step"] == expected_step
        assert evidence["adapter_roundtrip_verified"] is True
        assert evidence["adapter_roundtrip_max_abs_diff"] == 0.0
    assert (run_dir / "checkpoint-1/checkpoint_complete.json").exists()
    assert (run_dir / "checkpoint-2/checkpoint_complete.json").exists()


def test_tiny_cpu_standard_trl_runs_two_updates_with_action_masks(tmp_path):
    from qwen_repair.schemas import TestResult
    from qwen_repair.training.config import GRPOConfig
    from qwen_repair.training.grpo import make_grpo_trainer

    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "test-training-run"}))
    model, tokenizer, fixture = tiny_model()

    class SimulatedJudge:
        def run(self, candidate, bundle):
            passed = candidate.strip() == fixture.corrected_code
            return TestResult(status="passed" if passed else "failed", tests_total=1,
                              tests_passed=int(passed), tests_failed=int(not passed), synthetic_demo=False)

    config = GRPOConfig(sft_adapter="unused-in-test", device="cpu", gradient_checkpointing=False,
                        lora_rank=2, max_steps=2, max_new_tokens=64, max_context_tokens=1024)
    trainer, rollout = make_grpo_trainer(model, tokenizer, [fixture], [SimulatedJudge()], config, tmp_path,
                                       allow_test_checkpoint=True)
    original = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}
    samples = []
    retained = []
    original_loss = trainer._compute_loss
    original_logps = trainer._get_per_token_logps_and_entropies

    def traced_loss(model, inputs):
        trainer._test_loss_mask = (inputs["completion_mask"] * inputs["tool_mask"]).detach().clone()
        trainer._test_completion_ids = inputs["completion_ids"].detach().clone()
        return original_loss(model, inputs)

    def traced_logps(*args, **kwargs):
        result = original_logps(*args, **kwargs)
        if result[0].requires_grad:
            result[0].retain_grad()
            retained.append((result[0], trainer._test_loss_mask, trainer._test_completion_ids))
        return result

    trainer._compute_loss = traced_loss
    trainer._get_per_token_logps_and_entropies = traced_logps

    def controlled_sample(model, prompt_ids):
        # This test fixes actions to exercise both rewards; production uses sampled model actions.
        code = fixture.corrected_code if len(samples) % 4 == 3 else fixture.task.buggy_code
        action_ids = tokenizer.encode(f"```python\n{code}\n```", add_special_tokens=False) + [tokenizer.eos_token_id]
        samples.append((list(prompt_ids), list(action_ids)))
        with torch.no_grad():
            ids = torch.tensor([prompt_ids + action_ids])
            logits = model(ids).logits[0, len(prompt_ids) - 1:-1].float()
            logps = logits.log_softmax(-1).gather(-1, torch.tensor(action_ids).unsqueeze(-1)).squeeze(-1).tolist()
        return action_ids, logps

    rollout._sample = controlled_sample
    trainer.train()
    assert trainer.state.global_step == 2
    assert rollout.groups == rollout.nonzero_variance_groups == 2
    assert len(samples) == 8
    assert any(not torch.equal(original[name], value) for name, value in model.named_parameters() if name in original)
    public = [json.loads(line) for line in (tmp_path / "rollouts.jsonl").read_text().splitlines()]
    private = [json.loads(line) for line in (tmp_path / "private_evaluation.jsonl").read_text().splitlines()]
    public_ids = {(row["run_id"], row["episode_id"]) for row in public}
    private_ids = {(row["run_id"], row["episode_id"]) for row in private}
    assert len(public_ids) == len(public) == 4
    assert public_ids == private_ids
    assert {row["run_id"] for row in public} == {"test-training-run"}
    assert all("reward" not in row and "private_result" not in row for row in public)
    assert all(0 in row["env_mask"] and 1 in row["env_mask"] for row in public)
    for row in public:
        assert sum(row["env_mask"]) == sum(len(attempt["action_ids"]) for attempt in row["attempts"])
    assert (tmp_path / "checkpoint-2/checkpoint_complete.json").exists()
    assert retained
    assert all(torch.count_nonzero(logps.grad[mask == 0]).item() == 0 for logps, mask, _ in retained)
    for logps, mask, completion_ids in retained:
        for row in range(mask.shape[0]):
            first_eos = torch.where(completion_ids[row] == tokenizer.eos_token_id)[0][0].item()
            second_action = mask[row].bool() & (torch.arange(mask.shape[1]) > first_eos)
            assert second_action.any()
            assert torch.count_nonzero(logps.grad[row][second_action]).item() > 0


def child_grpo(output, resume=False):
    from qwen_repair.schemas import TestResult
    from qwen_repair.training.common import verify_checkpoint
    from qwen_repair.training.config import GRPOConfig
    from qwen_repair.training.grpo import make_grpo_trainer

    output = Path(output)
    output.mkdir(exist_ok=True)
    model, tokenizer, fixture = tiny_model()

    class SimulatedJudge:
        def run(self, candidate, bundle):
            return TestResult(status="failed", tests_total=1, tests_failed=1, synthetic_demo=False)

    config = GRPOConfig(sft_adapter="unused-in-test", device="cpu", gradient_checkpointing=False,
                        lora_rank=2, max_steps=2, stop_after_steps=None if resume else 1,
                        max_new_tokens=4, max_context_tokens=1024)
    trainer, rollout = make_grpo_trainer(model, tokenizer, [fixture], [SimulatedJudge()], config, output,
                                       allow_test_checkpoint=True)
    checkpoint = output / "checkpoint-1" if resume else None
    if checkpoint:
        verify_checkpoint(checkpoint, config, allow_test_checkpoint=True)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    (output / "child_result.json").write_text(json.dumps({"step": trainer.state.global_step, "groups": rollout.groups}))


def test_tiny_cpu_grpo_sampled_rollouts_resume_in_new_process(tmp_path):
    test_file = str(Path(__file__).resolve())
    code = "import runpy,sys; mod=runpy.run_path(sys.argv[1]); mod['child_grpo'](sys.argv[2], sys.argv[3]=='resume')"
    run_dir = tmp_path / "grpo-resume"
    for stage, expected_step in (("initial", 1), ("resume", 2)):
        result = subprocess.run([sys.executable, "-c", code, test_file, str(run_dir), stage],
                                capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stderr + result.stdout
        evidence = json.loads((run_dir / "child_result.json").read_text())
        assert evidence["step"] == expected_step
        assert evidence["groups"] == 1
    groups = [json.loads(line) for line in (run_dir / "groups.jsonl").read_text().splitlines()]
    assert [row["policy_version"] for row in groups] == ["step-0", "step-1"]
    assert len({row["group"] for row in groups}) == 2


def test_resumed_training_summary_labels_process_segment_metrics(tmp_path):
    from qwen_repair.training.common import finish_run
    from qwen_repair.training.config import SFTConfig

    (tmp_path / "training_status.json").write_text(json.dumps({"segment_start_step": 1}))
    trainer = SimpleNamespace(
        state=SimpleNamespace(global_step=2, log_history=[{"step": 1, "loss": 3.0}, {"step": 2, "loss": 2.0}]),
        save_model=lambda path: Path(path).mkdir(),
        processing_class=SimpleNamespace(save_pretrained=lambda path: None),
    )
    result = finish_run(SFTConfig(device="cpu"), tmp_path, trainer, perf_counter(), "float32", reward_groups=1)
    assert result["metrics_scope"] == "current_process_segment"
    assert result["global_step_scope"] == "cumulative"
    assert result["global_step"] == 2
    assert result["optimizer_steps_in_segment"] == result["reward_groups"] == 1
    assert result["training_log"] == [{"step": 2, "loss": 2.0}]
