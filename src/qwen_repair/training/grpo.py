import json
import math
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from qwen_repair.artifacts import write_json
from qwen_repair.data.fixtures import load_fixture
from qwen_repair.harness.feedback import format_parse_error
from qwen_repair.harness.parser import CandidateError, parse_candidate
from qwen_repair.harness.prompts import build_initial_messages
from qwen_repair.policies.transformers import TokenConversation
from qwen_repair.schemas import Message, PromptView

from .common import (
    TrainingInfrastructureError, begin_run, checked_executor, checkpoint_callback,
    fail_run, finish_run, load_base, require_real_result,
    record_executors, seal_training_run, file_hashes,
)
from .sft import bounded_feedback, bounded_text, verify_adapter_roundtrip


def prompt_key(messages) -> str:
    return json.dumps(messages, sort_keys=True, ensure_ascii=False)


def terminal_reward(*, episode_reward, **kwargs):
    return [float(value) for value in episode_reward]


class MultiTurnRollout:
    """Return full two-turn streams to TRL; its env_mask excludes injected tokens."""

    def __init__(self, tokenizer, fixtures, executors, config, output: Path):
        self.tokenizer = tokenizer
        self.config = config
        self.output = output
        manifest_path = output / "manifest.json"
        self.run_id = json.loads(manifest_path.read_text())["run_id"] if manifest_path.exists() else uuid4().hex
        self.by_prompt = {}
        self.rows = []
        for fixture, executor in zip(fixtures, executors, strict=True):
            messages = [message.model_dump() for message in build_initial_messages(PromptView.from_task(fixture.task))]
            self.by_prompt[prompt_key(messages)] = (fixture, executor)
            self.rows.append({"prompt": messages})
        self.groups = 0
        self.nonzero_variance_groups = 0
        self.generated_tokens = 0
        self.generation_seconds = 0.0
        self.executor_seconds = 0.0

    def _sample(self, model, prompt_ids):
        import torch
        from transformers import GenerationConfig

        if len(prompt_ids) + self.config.max_new_tokens > self.config.max_context_tokens:
            raise TrainingInfrastructureError("Context budget exceeded; cancel the complete rollout group")
        inputs = torch.tensor([prompt_ids], device=model.device, dtype=torch.long)
        settings = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens, do_sample=True,
            temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True, output_scores=True, use_cache=True,
        )
        started = perf_counter()
        with torch.no_grad():
            result = model.generate(inputs, attention_mask=torch.ones_like(inputs), generation_config=settings)
        self.generation_seconds += perf_counter() - started
        ids = result.sequences[0].tolist()
        if ids[:len(prompt_ids)] != prompt_ids:
            raise RuntimeError("Generation changed the original token prefix")
        action_ids = ids[len(prompt_ids):]
        if not action_ids or len(action_ids) != len(result.scores):
            raise RuntimeError("Generation returned incomplete token/logprob evidence")
        logprobs = [
            float(torch.log_softmax(score[0].float(), dim=-1)[token].item())
            for token, score in zip(action_ids, result.scores, strict=True)
        ]
        if not all(math.isfinite(value) for value in logprobs):
            raise RuntimeError("Nonfinite rollout logprobs; abort before a policy update")
        self.generated_tokens += len(action_ids)
        return action_ids, logprobs

    def _execute(self, executor, code, bundle):
        started = perf_counter()
        result = require_real_result(executor.run(code, bundle))
        self.executor_seconds += perf_counter() - started
        return result

    def _episode(self, model, messages, fixture, executor, policy_version):
        task = fixture.task
        conversation = TokenConversation.start(self.tokenizer, [Message.model_validate(item) for item in messages])
        prompt_ids = list(conversation.token_ids)
        logprobs, attempts = [], []
        final_code = None
        for number in (1, 2):
            input_ids = list(conversation.token_ids)
            action_ids, action_logps = self._sample(model, input_ids)
            conversation.append_action(action_ids)
            logprobs.extend(action_logps)
            response = self.tokenizer.decode(action_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            final_code, public_result, parse_error = None, None, None
            try:
                final_code = parse_candidate(response, task.entry_point, task.signature)
            except CandidateError as error:
                parse_error = error.code
                feedback = format_parse_error(error, max_tokens=100_000)
                feedback = bounded_text(self.tokenizer, feedback, self.config.max_feedback_tokens)
            else:
                public_result = self._execute(executor, final_code, task.public_tests_ref)
                feedback = bounded_feedback(self.tokenizer, public_result, self.config.max_feedback_tokens)
            attempts.append({
                "number": number, "response": response, "code": final_code,
                "parse_error": parse_error, "public_feedback": feedback,
                "public_result": public_result.model_dump() if public_result else None,
                "prompt_ids": input_ids, "action_ids": action_ids,
                "action_logprobs": action_logps,
            })
            if number == 1:
                inserted = conversation.append_feedback(feedback)
                logprobs.extend([0.0] * len(inserted))
        private_result = self._execute(executor, final_code, task.private_tests_ref) if final_code is not None else None
        reward = int(private_result is not None and private_result.status == "passed")
        completion_ids = conversation.token_ids[len(prompt_ids):]
        action_mask = conversation.action_mask[len(prompt_ids):]
        if len(completion_ids) != len(action_mask) or len(completion_ids) != len(logprobs):
            raise RuntimeError("Completion, action mask and rollout logprobs must align exactly")
        return {
            "task_id": task.task_id, "policy_version": policy_version,
            "prompt_ids": prompt_ids, "completion_ids": completion_ids,
            "env_mask": action_mask, "logprobs": logprobs, "attempts": attempts,
            "reward": reward,
            "private_result": private_result.model_dump() if private_result else None,
        }

    def __call__(self, prompts, trainer):
        if trainer.accelerator.num_processes != 1:
            raise ValueError("M1 GRPO supports one process; distributed grouping has not been validated")
        group_size = self.config.num_generations
        if len(prompts) % group_size:
            raise RuntimeError("TRL did not supply complete reward groups")
        model = trainer.accelerator.unwrap_model(trainer.model)
        previous_mode = model.training
        model.eval()
        outputs = []
        version = f"step-{trainer.state.global_step}"
        parameter_versions = [parameter._version for parameter in model.parameters() if parameter.requires_grad]
        try:
            for begin in range(0, len(prompts), group_size):
                group_prompts = prompts[begin:begin + group_size]
                key = prompt_key(group_prompts[0])
                if any(prompt_key(prompt) != key for prompt in group_prompts):
                    raise RuntimeError("A reward group mixes different task states")
                fixture, executor = self.by_prompt[key]
                if hasattr(executor, "policy_version"):
                    executor.policy_version = version
                group = [self._episode(model, prompt, fixture, executor, version) for prompt in group_prompts]
                if f"step-{trainer.state.global_step}" != version or parameter_versions != [
                    parameter._version for parameter in model.parameters() if parameter.requires_grad
                ]:
                    raise RuntimeError("The policy changed before all group rollouts completed")
                self.groups += 1
                group_id = f"{version}-group-{uuid4().hex}"
                has_variance = len({item["reward"] for item in group}) > 1
                self.nonzero_variance_groups += int(has_variance)
                for index, item in enumerate(group, 1):
                    episode_id = f"{group_id}-episode-{index:03}"
                    item["episode_id"] = episode_id
                    private = {"group": group_id, "episode_id": episode_id, "task_id": item["task_id"], "policy_version": version,
                               "terminal_reward": item["reward"], "result": item.pop("private_result")}
                    self._append("private_evaluation.jsonl", private)
                    self._append("rollouts.jsonl", {"group": group_id, **{key: value for key, value in item.items() if key != "reward"}})
                self._append("groups.jsonl", {
                    "group": group_id, "task_id": fixture.task.task_id, "policy_version": version,
                    "rewards": [item["reward"] for item in group], "nonzero_reward_variance": has_variance,
                })
                outputs.extend(group)
        except TrainingInfrastructureError:
            self._append("groups.jsonl", {"policy_version": version, "status": "cancelled_infrastructure"})
            raise
        finally:
            model.train(previous_mode)
        return {
            "prompt_ids": [item["prompt_ids"] for item in outputs],
            "completion_ids": [item["completion_ids"] for item in outputs],
            "logprobs": [item["logprobs"] for item in outputs],
            "env_mask": [item["env_mask"] for item in outputs],
            "episode_reward": [item["reward"] for item in outputs],
        }

    def _append(self, filename, value):
        with (self.output / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"schema_version": "1.0", "run_id": self.run_id, "synthetic_demo": False, **value}, ensure_ascii=False) + "\n")


def make_grpo_trainer(model, tokenizer, fixtures, executors, config, output: Path, dtype_name="float32", *, allow_test_checkpoint=False):
    from datasets import Dataset
    from trl import GRPOConfig as TRLConfig, GRPOTrainer

    rollout = MultiTurnRollout(tokenizer, fixtures, executors, config, output)
    args = TRLConfig(
        output_dir=str(output), max_steps=config.max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=config.num_generations,
        generation_batch_size=config.num_generations, num_generations=config.num_generations,
        num_iterations=1, beta=0.0, loss_type="grpo", scale_rewards="group",
        temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
        max_completion_length=config.max_context_tokens,
        mask_truncated_completions=False, disable_dropout=True,
        learning_rate=config.learning_rate, seed=config.seed, data_seed=config.seed,
        save_strategy="steps", save_steps=1, save_total_limit=2,
        logging_steps=1, report_to=[], disable_tqdm=True,
        use_cpu=config.device == "cpu", bf16=dtype_name == "bfloat16", fp16=dtype_name == "float16",
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False, use_vllm=False,
        chat_template_kwargs={"enable_thinking": False}, optim="adamw_torch",
    )
    callback = checkpoint_callback(config, allow_test_checkpoint=allow_test_checkpoint)
    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, args=args,
        train_dataset=Dataset.from_list(rollout.rows),
        reward_funcs=terminal_reward, rollout_func=rollout,
        callbacks=[callback],
    )
    callback.scaler_getter = lambda: getattr(trainer.accelerator, "scaler", None) is not None
    return trainer, rollout


def run_grpo_smoke(config, output, executor, resume_from=None):
    output = Path(output)
    started = begin_run(config, output, resume_from, "grpo_fixed_two_turn")
    try:
        from peft import PeftModel

        adapter = Path(config.sft_adapter)
        metadata = json.loads((adapter / "repair_training.json").read_text())
        if metadata["model_id"] != config.model_id or metadata["model_revision"] != config.model_revision:
            raise ValueError("SFT adapter base does not match the selected model revision")
        manifest = json.loads((output / "manifest.json").read_text())
        manifest["initial_sft_adapter_hashes"] = file_hashes(adapter, ignored=frozenset())
        manifest["generation"] = {
            "temperature": 1.0, "top_p": 1.0, "top_k": 0,
            "repetition_penalty": 1.0, "enable_thinking": False,
            "num_generations": config.num_generations, "attempts": 2,
        }
        write_json(output / "manifest.json", manifest)
        fixtures = [load_fixture(name) for name in config.fixture_names]
        executors = [checked_executor(executor, fixture.task) for fixture in fixtures]
        record_executors(output, fixtures, executors)
        for fixture, backend in zip(fixtures, executors, strict=True):
            for bundle in (fixture.task.public_tests_ref, fixture.task.private_tests_ref):
                if require_real_result(backend.run(fixture.reference_code, bundle)).status != "passed":
                    raise ValueError("Authored GRPO reference must pass both trusted test bundles")
        model, tokenizer, dtype_name = load_base(config)
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
        trainer, rollout = make_grpo_trainer(model, tokenizer, fixtures, executors, config, output, dtype_name)
        trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
        result = finish_run(
            config, output, trainer, started, dtype_name,
            reward_groups=rollout.groups,
            nonzero_reward_variance_groups=rollout.nonzero_variance_groups,
            nonzero_reward_variance_fraction=rollout.nonzero_variance_groups / rollout.groups if rollout.groups else None,
            generated_tokens=rollout.generated_tokens,
            generation_seconds=rollout.generation_seconds,
            tokens_per_second=rollout.generated_tokens / rollout.generation_seconds if rollout.generation_seconds else None,
            executor_seconds=rollout.executor_seconds,
            kl_coefficient=0.0,
        )
        probe = TokenConversation.start(tokenizer, [Message.model_validate(item) for item in rollout.rows[0]["prompt"]])
        result.update(verify_adapter_roundtrip(model, output / "adapter", probe.token_ids[:32]))
        write_json(output / "training_result.json", result)
        seal_training_run(output, result)
        return result
    except Exception as error:
        fail_run(output, error)
        raise
