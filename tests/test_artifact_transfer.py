import json
import zipfile

import pytest

from qwen_repair.artifact_transfer import digest, publish_training, restore_training, verify_training_seal
from qwen_repair.artifacts import write_json


def sealed_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "manifest.json", {"kind": "training", "run_id": "a" * 32, "status": "paused_at_checkpoint"})
    write_json(run / "environment.json", {"gpu": {"status": "cpu_profile"}})
    write_json(run / "training_result.json", {"global_step": 1})
    checkpoint = run / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").write_bytes(b"not a pickle; transfer never deserializes optimizer state")
    write_json(run / "training_complete.json", {"schema_version": "1.0", "run_id": "a" * 32,
               "status": "paused_at_checkpoint", "files": {
                   path.relative_to(run).as_posix(): digest(path) for path in run.rglob("*") if path.is_file()
               }})
    return run


def test_publish_and_restore_preserve_every_file_and_previous_checkpoint(tmp_path):
    run = sealed_run(tmp_path)
    first = publish_training(run, tmp_path / "persistent")
    second = publish_training(run, tmp_path / "persistent")
    assert first["bundle"] != second["bundle"]
    restored = tmp_path / "new-runtime" / "run"
    report = restore_training(first["bundle"], restored)
    assert report["passed"] and not report["resume_executed"]
    assert verify_training_seal(restored) == verify_training_seal(run)
    assert (restored / "checkpoint-1/optimizer.pt").read_bytes() == (run / "checkpoint-1/optimizer.pt").read_bytes()
    with pytest.raises(ValueError, match="new directory"):
        restore_training(first["bundle"], restored)


@pytest.mark.parametrize("damage", ["missing", "changed", "extra", "symlink"])
def test_export_rejects_unsealed_or_changed_inventory(tmp_path, damage):
    run = sealed_run(tmp_path)
    if damage == "missing":
        (run / "environment.json").unlink()
    elif damage == "changed":
        (run / "environment.json").write_text("{}")
    elif damage == "extra":
        (run / "unexpected.txt").write_text("extra")
    else:
        (run / "link").symlink_to(run / "environment.json")
    with pytest.raises(ValueError):
        publish_training(run, tmp_path / "persistent")
    assert not (tmp_path / "persistent").exists()


def test_restore_requires_ready_and_verifies_payload(tmp_path):
    from pathlib import Path

    record = publish_training(sealed_run(tmp_path), tmp_path / "persistent")
    bundle = Path(record["bundle"])
    ready = (bundle / "READY").read_text()
    (bundle / "READY").unlink()
    with pytest.raises(FileNotFoundError):
        restore_training(bundle, tmp_path / "restored")
    (bundle / "READY").write_text(ready)
    with (bundle / "training.zip").open("ab") as stream:
        stream.write(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        restore_training(bundle, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize("name", ["../escape", "/escape", "a/../../escape", "a\\escape"])
def test_restore_rejects_traversal_even_with_matching_archive_hash(tmp_path, name):
    from pathlib import Path

    record = publish_training(sealed_run(tmp_path), tmp_path / "persistent")
    bundle = Path(record["bundle"])
    archive = bundle / "training.zip"
    with zipfile.ZipFile(archive, "a") as stream:
        stream.writestr(name, "untrusted")
    metadata = json.loads((bundle / "transfer.json").read_text())
    metadata.update(sha256=digest(archive), bytes=archive.stat().st_size)
    write_json(bundle / "transfer.json", metadata)
    (bundle / "READY").write_text(digest(bundle / "transfer.json"))
    with pytest.raises(ValueError, match="relative"):
        restore_training(bundle, tmp_path / "restored")
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "restored").exists()


def test_archive_expansion_is_bounded_and_destination_cannot_modify_source(tmp_path):
    run = sealed_run(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        publish_training(run, run / "backups")
    record = publish_training(run, tmp_path / "persistent")
    with pytest.raises(ValueError, match="limits"):
        restore_training(record["bundle"], tmp_path / "restored", max_bytes=1)
