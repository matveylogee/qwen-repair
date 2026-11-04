"""Copy sealed training runs without loading model or optimizer objects."""

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from uuid import uuid4

from qwen_repair.artifacts import write_json


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def relative_file(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or any(
        part in {"", ".", ".."} for part in name.split("/")
    ):
        raise ValueError("Artifact names must be relative file paths without traversal")
    return path


def verify_training_seal(run: Path) -> dict:
    run = Path(run)
    marker = run / "training_complete.json"
    if run.is_symlink() or marker.is_symlink():
        raise ValueError("Training artifacts cannot contain symlinks")
    seal = json.loads(marker.read_text())
    if seal.get("schema_version") != "1.0" or seal.get("status") not in {"completed", "paused_at_checkpoint"}:
        raise ValueError("Training run is incomplete")
    if not isinstance(seal.get("run_id"), str) or re.fullmatch(r"[0-9a-f]{32}", seal["run_id"]) is None:
        raise ValueError("Invalid training run identity")
    expected = seal.get("files")
    if not isinstance(expected, dict) or not {"manifest.json", "training_result.json", "environment.json"} <= expected.keys():
        raise ValueError("Training seal is missing required reports")
    for name in expected:
        relative_file(name)
    files = {}
    for path in run.rglob("*"):
        if path.is_symlink():
            raise ValueError("Training artifacts cannot contain symlinks")
        if path.is_file() and path != marker:
            files[path.relative_to(run).as_posix()] = digest(path)
    if files != expected:
        raise ValueError("Training artifact inventory or checksum mismatch")
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("kind") != "training" or manifest.get("run_id") != seal.get("run_id"):
        raise ValueError("Training seal identity mismatch")
    if manifest.get("status") != seal["status"]:
        raise ValueError("Training seal status mismatch")
    return seal


def publish_training(run: Path, destination: Path) -> dict:
    run, destination = Path(run).resolve(), Path(destination).resolve()
    if destination.is_relative_to(run):
        raise ValueError("Persistent destination must be outside the sealed run")
    seal = verify_training_seal(run)
    destination.mkdir(parents=True, exist_ok=True)
    bundle = destination / f"{seal['run_id']}-{uuid4().hex}"
    bundle.mkdir()
    # Only the archive and small completion records are copied through Drive.
    with tempfile.TemporaryDirectory(prefix="qrepair-export-") as staging:
        archive = Path(staging) / "training.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as stream:
            for name in sorted([*seal["files"], "training_complete.json"]):
                stream.write(run / name, name)
        if verify_training_seal(run) != seal:
            raise ValueError("Training run changed while the archive was being created")
        checksum = digest(archive)
        shutil.copyfile(archive, bundle / "training.zip")
        if digest(bundle / "training.zip") != checksum:
            raise ValueError("Persistent archive failed readback verification")
    record = {"schema_version": "1.0", "run_id": seal["run_id"], "status": seal["status"],
              "archive": "training.zip", "sha256": checksum, "bytes": (bundle / "training.zip").stat().st_size}
    write_json(bundle / "transfer.json", record)
    (bundle / "READY").write_text(digest(bundle / "transfer.json") + "\n")
    return {"passed": True, "bundle": str(bundle), **record,
            "verification_scope": "local persistent-path readback; cloud upload is asynchronous"}


def restore_training(bundle: Path, output: Path, *, max_bytes: int = 4 * 1024**3) -> dict:
    bundle, output = Path(bundle), Path(output)
    if output.exists():
        raise ValueError("Restore output must be a new directory")
    record_path = bundle / "transfer.json"
    if (bundle / "READY").read_text().strip() != digest(record_path):
        raise ValueError("Transfer is incomplete or its metadata was modified")
    record = json.loads(record_path.read_text())
    if record.get("schema_version") != "1.0" or record.get("archive") != "training.zip":
        raise ValueError("Unsupported training transfer")
    archive = bundle / "training.zip"
    if archive.stat().st_size != record["bytes"] or digest(archive) != record["sha256"]:
        raise ValueError("Archive is incomplete or its checksum does not match")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qrepair-restore-", dir=output.parent) as staging:
        unpacked = Path(staging) / "run"
        unpacked.mkdir()
        with zipfile.ZipFile(archive) as stream:
            entries = stream.infolist()
            names = [entry.filename for entry in entries]
            if len(entries) > 10_000 or len(names) != len(set(names)) or sum(e.file_size for e in entries) > max_bytes:
                raise ValueError("Archive exceeds bounded restore limits or has duplicate entries")
            for entry in entries:
                relative_file(entry.filename)
                mode = (entry.external_attr >> 16) & 0o170000
                if entry.is_dir() or mode not in {0, 0o100000}:
                    raise ValueError("Archive contains a non-regular file")
                target = unpacked / entry.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.open(entry) as source, target.open("xb") as dest:
                    shutil.copyfileobj(source, dest, length=1024 * 1024)
        seal = verify_training_seal(unpacked)
        if seal["run_id"] != record["run_id"] or seal["status"] != record["status"]:
            raise ValueError("Archive run identity mismatch")
        unpacked.rename(output)
    return {"passed": True, "output": str(output), "run_id": seal["run_id"],
            "status": seal["status"], "resume_executed": False}
