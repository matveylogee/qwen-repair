import json

import pytest

from qwen_repair.artifact_transfer import digest
from qwen_repair.artifacts import write_json
from qwen_repair.training_audit import audit_training


def evidence_run(tmp_path, *, grpo=False, cpu=False, skipped=False, changed_resume=False):
    run = tmp_path / "reports"
    run.mkdir()
    (run / "segments").mkdir()
    identity = "a" * 32
    for start in (0, 1):
        updates = {"applied": int(not skipped), "skipped": int(skipped), "nonfinite_gradient_steps": 0,
                   "nonzero_gradient_steps": int(not grpo), "trainable_parameters_changed": not grpo,
                   "before_sha256": f"weights-{start}" if not (changed_resume and start) else "unrelated",
                   "after_sha256": f"weights-{start + 1}"}
        result = {"global_step": start + 1, "planned_steps": 2, "segment_start_step": start,
                  "optimizer_updates": updates, "adapter_roundtrip_verified": True,
                  "adapter_roundtrip_max_abs_diff": 0.0, "duration_seconds": 1.0,
                  "peak_vram_bytes": None if cpu else 100, "learning_signal_observed": False}
        manifest = {"kind": "training", "run_id": identity, "synthetic_demo": False,
                    "status": "completed", "method": "grpo_fixed_two_turn" if grpo else "sft",
                    "process_id": 100 + start,
                    "config": {"device": "cpu" if cpu else "cuda"}, "resume_from": "checkpoint-1" if start else None}
        environment = {"gpu": {"status": "cpu_profile" if cpu else "available"}}
        write_json(run / f"segments/{start}.json", {"result": result, "manifest": manifest, "environment": environment,
                   "synthetic_demo": False, "run_id": identity})
        checkpoint = run / f"checkpoint-{start + 1}"
        checkpoint.mkdir()
        for filename in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json", "adapter_model.safetensors"):
            (checkpoint / filename).write_bytes(b"fixture only; must never deserialize")
        write_json(checkpoint / "trainer_state.json", {"global_step": start + 1})
        write_json(checkpoint / "checkpoint_complete.json", {
            "schema_version": "1.0", "run_id": identity, "global_step": start + 1,
            "scaler_required": False, "resolved_dtype": "float32", "optimizer_updates": updates,
            "provenance": {key: {"fixture": True} for key in ("source_hashes", "fixture_hashes", "versions", "executors")},
            "files": {path.name: digest(path) for path in checkpoint.iterdir()},
        })
    write_json(run / "manifest.json", manifest)
    write_json(run / "training_result.json", result)
    write_json(run / "environment.json", environment)
    write_json(run / "training_complete.json", {"schema_version": "1.0", "status": "completed", "run_id": identity,
               "files": {path.relative_to(run).as_posix(): digest(path) for path in run.rglob("*") if path.is_file()}})
    return run


def test_audit_distinguishes_update_signal_from_completed_plumbing(tmp_path):
    report = audit_training(evidence_run(tmp_path, grpo=True))
    assert not report["passed"]
    assert report["plumbing_passed"]
    assert report["status"] == "plumbing_only"
    assert not report["learning_signal_observed"]
    assert not report["research_training_ready"]
    assert not report["runtime_loss_recovery_verified"]


@pytest.mark.parametrize("option, failed_check", [("cpu", "gpu_segments"), ("skipped", "actual_optimizer_updates"),
                                                   ("changed_resume", "resume_parameter_match")])
def test_audit_rejects_cpu_skipped_updates_and_wrong_restored_weights(tmp_path, option, failed_check):
    report = audit_training(evidence_run(tmp_path, **{option: True}))
    assert not report["passed"]
    assert not report["checks"][failed_check]


def test_audit_requires_original_checksums_and_does_not_import_torch(tmp_path):
    import subprocess
    import sys

    run = evidence_run(tmp_path)
    result = subprocess.run([sys.executable, "-c", "import sys; from qwen_repair.training_audit import audit_training; "
                             "assert audit_training(sys.argv[1])['passed']; assert 'torch' not in sys.modules", str(run)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    (run / "segments/0.json").write_text(json.dumps({"tampered": True}))
    with pytest.raises(ValueError, match="checksum"):
        audit_training(run)
