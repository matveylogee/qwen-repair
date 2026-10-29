import hashlib
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from qwen_repair.schemas import PrivateEvaluation, Trajectory


Record = TypeVar("Record", bound=BaseModel)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict | BaseModel) -> None:
    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[Record]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(record.model_dump_json() + "\n")


def read_jsonl(path: Path, model: type[Record]) -> list[Record]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Empty JSONL record in {path.name}:{number}")
            records.append(model.model_validate_json(line))
    return records


def read_run(directory: Path) -> tuple[list[Trajectory], list[PrivateEvaluation]]:
    checksums = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
    hashes = checksums.get("files", {})
    required = {"trajectories.jsonl", "private_evaluation.jsonl", "manifest.json", "config.yaml", "environment.json", "metrics.json"}
    if set(hashes) != required:
        raise ValueError("Run checksum inventory is incomplete or unexpected")
    for name, expected in hashes.items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    from qwen_repair.schemas import RunManifest

    manifest = RunManifest.model_validate_json((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.status not in {"completed", "completed_with_exclusions"}:
        raise ValueError("Run is not complete")
    trajectories = read_jsonl(directory / "trajectories.jsonl", Trajectory)
    evaluations = read_jsonl(directory / "private_evaluation.jsonl", PrivateEvaluation)
    if any(item.run_id != manifest.run_id for item in [*trajectories, *evaluations]):
        raise ValueError("Record run ID differs from manifest")
    return trajectories, evaluations
