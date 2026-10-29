import json
from types import SimpleNamespace

import pytest

from qwen_repair.data.fixtures import load_fixture
from qwen_repair.schemas import TestResult
from qwen_repair.training.common import (
    TrainingInfrastructureError, checked_executor, file_hashes, require_real_result,
    verify_checkpoint,
)
from qwen_repair.training.config import GRPOConfig, SFTConfig
from qwen_repair.training.grpo import MultiTurnRollout
from qwen_repair.training.sft import build_sft_records, supervised_record


class ToyTokenizer:
    eos_token = "~"
    eos_token_id = 126
    pad_token_id = 0

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return bytes(token for token in ids if token != self.eos_token_id).decode()

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        value = "".join(f"<{message['role']}>{message['content']}~\n" for message in messages) + "<assistant>"
        return self.encode(value) if tokenize else value


class TestJudge:
    __test__ = False
    synthetic_demo = False

    def __init__(self, fixture, infra=False):
        self.fixture = fixture
        self.calls = []
        self.infra = infra

    def preflight(self):
        return {"passed": True}

    def run(self, code, bundle):
        self.calls.append((code, bundle))
        if self.infra:
            return TestResult(status="infra_error", synthetic_demo=False)
        passed = code.strip() in {self.fixture.corrected_code, self.fixture.reference_code}
        return TestResult(
            status="passed" if passed else "failed", tests_total=1,
            tests_passed=int(passed), tests_failed=int(not passed),
            feedback="public example" if bundle == self.fixture.task.public_tests_ref else "HIDDEN_ONLY",
            synthetic_demo=False,
        )


def test_sft_only_target_tokens_receive_labels_including_recovery():
    fixture, tokenizer = load_fixture("sum_positive"), ToyTokenizer()
    records, audit = build_sft_records(tokenizer, [fixture], [TestJudge(fixture)], SFTConfig(device="cpu"))
    assert [item["kind"] for item in audit] == ["one_shot", "recovery"]
    assert audit[1]["public_result"]["status"] == "failed"
    for record in records:
        first = next(index for index, label in enumerate(record["labels"]) if label != -100)
        assert all(label == -100 for label in record["labels"][:first])
        assert record["labels"][first:] == record["input_ids"][first:]
        assert tokenizer.decode(record["labels"][first:]).startswith("```python\n")
    assert "public example" in tokenizer.decode(records[1]["input_ids"])
    assert "HIDDEN_ONLY" not in tokenizer.decode(records[1]["input_ids"])
    with pytest.raises(ValueError, match="context budget"):
        supervised_record(tokenizer, [1] * 100, "target", 50)


def test_real_training_rejects_synthetic_results_and_missing_isolation():
    with pytest.raises(TrainingInfrastructureError, match="Synthetic"):
        require_real_result(TestResult(status="passed", tests_total=1, tests_passed=1))
    with pytest.raises(TrainingInfrastructureError, match="non-synthetic"):
        checked_executor(SimpleNamespace(synthetic_demo=True), None)
    with pytest.raises(TrainingInfrastructureError, match="infrastructure"):
        require_real_result(TestResult(status="infra_error", synthetic_demo=False))


def test_two_turn_rollout_masks_feedback_preserves_prefix_and_keeps_private_separate(tmp_path):
    fixture, tokenizer = load_fixture("sum_positive"), ToyTokenizer()
    judge = TestJudge(fixture)
    config = GRPOConfig(sft_adapter="unused", device="cpu")
    rollout = MultiTurnRollout(tokenizer, [fixture], [judge], config, tmp_path)
    samples = []

    def sample(model, ids):
        samples.append(list(ids))
        code = fixture.task.buggy_code if len(samples) % 2 else fixture.corrected_code
        action = tokenizer.encode(f"```python\n{code}\n```") + [tokenizer.eos_token_id]
        return action, [-0.5] * len(action)

    rollout._sample = sample
    episode = rollout._episode(None, rollout.rows[0]["prompt"], fixture, judge, "step-0")
    first_ids = episode["attempts"][0]["action_ids"]
    assert samples[1][:len(samples[0]) + len(first_ids)] == samples[0] + first_ids
    assert episode["reward"] == 1
    assert episode["private_result"]["feedback"] == "HIDDEN_ONLY"
    assert "HIDDEN_ONLY" not in json.dumps(episode["attempts"])
    mask = episode["env_mask"]
    assert len(mask) == len(episode["completion_ids"]) == len(episode["logprobs"])
    assert sum(mask) == sum(len(item["action_ids"]) for item in episode["attempts"])
    assert all(lp == 0 for lp, is_action in zip(episode["logprobs"], mask) if not is_action)
    assert [bundle for _, bundle in judge.calls] == [fixture.task.public_tests_ref] * 2 + [fixture.task.private_tests_ref]


def test_infra_failure_cancels_episode_before_reward(tmp_path):
    fixture, tokenizer = load_fixture("sum_positive"), ToyTokenizer()
    judge = TestJudge(fixture, infra=True)
    rollout = MultiTurnRollout(tokenizer, [fixture], [judge], GRPOConfig(sft_adapter="unused", device="cpu"), tmp_path)
    ids = tokenizer.encode(f"```python\n{fixture.corrected_code}\n```")
    rollout._sample = lambda model, prompt: (ids, [-1.0] * len(ids))
    with pytest.raises(TrainingInfrastructureError):
        rollout._episode(None, rollout.rows[0]["prompt"], fixture, judge, "step-0")
    assert len(judge.calls) == 1


def test_checkpoint_rejects_incomplete_tampered_or_changed_training_state(tmp_path):
    config = SFTConfig(device="cpu")
    with pytest.raises(FileNotFoundError):
        verify_checkpoint(tmp_path, config)
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                 "adapter_config.json", "adapter_model.safetensors"):
        (tmp_path / name).write_bytes(b"test checkpoint bytes")
    payload = {"config": config.model_dump(exclude={"stop_after_steps"}), "files": file_hashes(tmp_path)}
    marker = tmp_path / "checkpoint_complete.json"
    marker.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="complete provenance"):
        verify_checkpoint(tmp_path, config)
    assert verify_checkpoint(tmp_path, config, allow_test_checkpoint=True) == payload
    with pytest.raises(ValueError, match="same resolved"):
        verify_checkpoint(tmp_path, config.model_copy(update={"max_steps": 3}), allow_test_checkpoint=True)
    assert verify_checkpoint(tmp_path, config.model_copy(update={"stop_after_steps": 1}), allow_test_checkpoint=True) == payload
    (tmp_path / "rng_state.pth").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint(tmp_path, config)


def test_checkpoint_requires_active_fp16_scaler_state(tmp_path):
    config = SFTConfig(device="cuda", dtype="float16")
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                 "adapter_config.json", "adapter_model.safetensors"):
        (tmp_path / name).write_bytes(b"test checkpoint bytes")
    payload = {
        "config": config.model_dump(exclude={"stop_after_steps"}), "files": file_hashes(tmp_path),
        "scaler_required": True, "resolved_dtype": "float16",
    }
    marker = tmp_path / "checkpoint_complete.json"
    marker.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing"):
        verify_checkpoint(tmp_path, config, allow_test_checkpoint=True)
    (tmp_path / "scaler.pt").write_bytes(b"test scaler state")
    payload["files"] = file_hashes(tmp_path)
    marker.write_text(json.dumps(payload))
    assert verify_checkpoint(tmp_path, config, allow_test_checkpoint=True) == payload
    with pytest.raises(ValueError, match="dtype differs"):
        verify_checkpoint(tmp_path, config, allow_test_checkpoint=True, expected_dtype="bfloat16")


def stub_trainer():
    class Model:
        training = True

        def eval(self):
            self.training = False

        def train(self, flag):
            self.training = flag

        def parameters(self):
            return []

    return SimpleNamespace(
        accelerator=SimpleNamespace(num_processes=1, unwrap_model=lambda model: model),
        model=Model(), state=SimpleNamespace(global_step=0),
    )


def test_grpo_rejects_mixed_tasks_before_generating(tmp_path):
    fixtures = [load_fixture("sum_positive"), load_fixture("last_or_none")]
    judges = [TestJudge(fixture) for fixture in fixtures]
    rollout = MultiTurnRollout(ToyTokenizer(), fixtures, judges, GRPOConfig(sft_adapter="unused", device="cpu"), tmp_path)
    with pytest.raises(RuntimeError, match="mixes different task"):
        rollout([row["prompt"] for row in rollout.rows], stub_trainer())
    assert all(not judge.calls for judge in judges)


def test_grpo_infra_cancels_whole_group_without_emitting_rewards(tmp_path):
    fixture, tokenizer = load_fixture("sum_positive"), ToyTokenizer()
    judge = TestJudge(fixture, infra=True)
    rollout = MultiTurnRollout(tokenizer, [fixture], [judge], GRPOConfig(sft_adapter="unused", device="cpu"), tmp_path)
    ids = tokenizer.encode(f"```python\n{fixture.corrected_code}\n```")
    rollout._sample = lambda model, prompt: (ids, [-1.0] * len(ids))
    with pytest.raises(TrainingInfrastructureError):
        rollout([rollout.rows[0]["prompt"]] * 2, stub_trainer())
    assert rollout.groups == 0
    assert not (tmp_path / "private_evaluation.jsonl").exists()
    assert not (tmp_path / "rollouts.jsonl").exists()
    record = json.loads((tmp_path / "groups.jsonl").read_text())
    assert record["status"] == "cancelled_infrastructure"
    assert "rewards" not in record


def test_recreated_rollout_at_same_step_uses_new_group_and_episode_ids(tmp_path):
    fixture, tokenizer = load_fixture("sum_positive"), ToyTokenizer()
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "resumed-run"}))
    ids = tokenizer.encode(f"```python\n{fixture.corrected_code}\n```")
    for _ in range(2):
        rollout = MultiTurnRollout(tokenizer, [fixture], [TestJudge(fixture)],
                                   GRPOConfig(sft_adapter="unused", device="cpu"), tmp_path)
        rollout._sample = lambda model, prompt: (ids, [-1.0] * len(ids))
        rollout([rollout.rows[0]["prompt"]] * 2, stub_trainer())
    groups = [json.loads(line) for line in (tmp_path / "groups.jsonl").read_text().splitlines()]
    episodes = [json.loads(line) for line in (tmp_path / "rollouts.jsonl").read_text().splitlines()]
    assert len({row["group"] for row in groups}) == 2
    assert len({(row["run_id"], row["episode_id"]) for row in episodes}) == 4
