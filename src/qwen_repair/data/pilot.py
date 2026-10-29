import ast
import copy
import difflib
import hashlib
import json
import math
from collections import Counter
from contextlib import closing
from itertools import islice
from pathlib import Path
from typing import Any

from qwen_repair.artifacts import write_json
from qwen_repair.schemas import TaskRecord


DATASET_ID = "KodCode/KodCode-Light-RL-10K"
DATASET_REVISION = "dcf78a8bbba9a613b596ce993c4921a38687dfcc"


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


class RejectedTask(ValueError):
    pass


def _json_value(node: ast.AST, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Name) and node.id in names:
        return copy.deepcopy(names[node.id])
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as error:
        raise RejectedTask("nonliteral_test_value") from error

    def normalize(item):
        if item is None or type(item) in (bool, int, str):
            return item
        if type(item) is float and math.isfinite(item):
            return item
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            return {key: normalize(child) for key, child in item.items()}
        raise RejectedTask("non_json_test_value")

    return normalize(value)


def extract_cases(test_source: str, entry_point: str) -> list[dict]:
    """Accept every assertion in a restricted literal-call test file or reject it whole."""
    try:
        tree = ast.parse(test_source)
    except SyntaxError as error:
        raise RejectedTask("test_syntax") from error
    cases = []
    test_names = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "solution":
            if node.level or any(alias.name != entry_point or alias.asname for alias in node.names):
                raise RejectedTask("test_import")
            continue
        if isinstance(node, ast.Import) and all(alias.name == "pytest" and not alias.asname for alias in node.names):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            raise RejectedTask("test_helpers_or_toplevel_code")
        if node.name in test_names:
            raise RejectedTask("duplicate_test_name")
        test_names.add(node.name)
        if node.decorator_list or node.args.posonlyargs or node.args.args or node.args.kwonlyargs or node.args.vararg or node.args.kwarg:
            raise RejectedTask("test_fixture_or_parameterization")
        names = {}
        mutable_uses = Counter()
        for statement in node.body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                continue
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                if statement.targets[0].id == entry_point:
                    raise RejectedTask("test_shadows_entry_point")
                if (isinstance(statement.value, ast.Name)
                        and isinstance(names.get(statement.value.id), (list, dict))):
                    raise RejectedTask("mutable_test_alias")
                names[statement.targets[0].id] = _json_value(statement.value, names)
                mutable_uses[statement.targets[0].id] = 0
                continue
            if not isinstance(statement, ast.Assert) or statement.msg is not None:
                raise RejectedTask("unsupported_test_statement")
            comparison = statement.test
            if not isinstance(comparison, ast.Compare) or len(comparison.ops) != 1 or not isinstance(comparison.ops[0], ast.Eq):
                raise RejectedTask("non_equality_assertion")
            call = comparison.left
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != entry_point:
                raise RejectedTask("not_direct_function_assertion")
            if any(keyword.arg is None for keyword in call.keywords):
                raise RejectedTask("unpacked_test_arguments")
            for value in ast.walk(comparison):
                if isinstance(value, ast.Name) and isinstance(names.get(value.id), (list, dict)):
                    mutable_uses[value.id] += 1
                    if mutable_uses[value.id] > 1:
                        raise RejectedTask("reused_mutable_test_value")
            cases.append({
                "args": [_json_value(arg, names) for arg in call.args],
                "kwargs": {keyword.arg: _json_value(keyword.value, names) for keyword in call.keywords},
                "expected": _json_value(comparison.comparators[0], names),
            })
    unique = {content_hash(case): case for case in cases}
    if len(unique) < 4:
        raise RejectedTask("fewer_than_four_independent_cases")
    return list(unique.values())


def inspect_reference(source: str) -> tuple[ast.Module, ast.FunctionDef]:
    if len(source.encode()) > 12000:
        raise RejectedTask("reference_too_long")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise RejectedTask("reference_syntax") from error
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise RejectedTask("not_one_function")
    entry = functions[0]
    allowed_imports = {"math", "string", "itertools", "functools", "collections", "operator", "bisect", "heapq", "re"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Global, ast.Nonlocal)):
            raise RejectedTask("unsupported_reference_structure")
        if isinstance(node, ast.Import) and any(alias.name not in allowed_imports for alias in node.names):
            raise RejectedTask("reference_import")
        if isinstance(node, ast.ImportFrom) and (node.level or node.module not in allowed_imports):
            raise RejectedTask("reference_import")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"open", "eval", "exec", "compile", "input", "print", "__import__"}:
            raise RejectedTask("reference_io_or_dynamic_code")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        raise RejectedTask("reference_toplevel_code")
    if entry.decorator_list or entry.args.vararg or entry.args.kwarg:
        raise RejectedTask("reference_signature")
    return tree, entry


def mutation_candidates(source: str, seed: int, limit: int = 8) -> list[dict]:
    tree = ast.parse(source)
    replacements = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
                    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Add: ast.Sub, ast.Sub: ast.Add,
                    ast.Mult: ast.Add, ast.FloorDiv: ast.Mult}
    positions = []
    for index, node in enumerate(ast.walk(tree)):
        if isinstance(node, ast.Compare):
            positions.extend((index, position, type(operator)) for position, operator in enumerate(node.ops) if type(operator) in replacements)
        elif isinstance(node, ast.BinOp) and type(node.op) in replacements:
            positions.append((index, None, type(node.op)))
    positions.sort(key=lambda value: content_hash([seed, value[0], value[1]]))
    candidates, seen = [], set()
    for index, position, old_type in positions:
        mutated = copy.deepcopy(tree)
        node = list(ast.walk(mutated))[index]
        new_type = replacements[old_type]
        if position is None:
            node.op = new_type()
        else:
            node.ops[position] = new_type()
        code = ast.unparse(mutated) + "\n"
        digest = content_hash(code)
        if digest in seen:
            continue
        seen.add(digest)
        candidates.append({
            "code": code, "candidate_hash": digest,
            "mutation": {"operator": f"{old_type.__name__}_to_{new_type.__name__}",
                         "line": node.lineno, "column": node.col_offset, "seed": seed,
                         "patch": "".join(difflib.unified_diff(source.splitlines(True), code.splitlines(True), fromfile="reference.py", tofile="buggy.py"))},
        })
        if len(candidates) == limit:
            break
    if not candidates:
        raise RejectedTask("no_supported_mutation")
    return candidates


def fetch_pilot(output: Path, *, limit: int = 100, revision: str = DATASET_REVISION) -> dict:
    if not 1 <= limit <= 100:
        raise ValueError("M1 fetch limit must be between 1 and 100 rows")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("A full dataset revision SHA is required")
    output.mkdir(parents=True, exist_ok=False)
    keys = ("question_id", "question", "solution", "test", "test_info", "metadata")
    count = 0
    with closing(_iter_pilot_rows(revision, limit, keys)) as rows, (output / "raw.jsonl").open("x", encoding="utf-8") as stream:
        for row in islice(rows, limit):
            if any(key not in row for key in keys[:4]):
                raise ValueError("KodCode source schema changed")
            stream.write(json.dumps({key: row.get(key) for key in keys}, ensure_ascii=False) + "\n")
            count += 1
    metadata = {"schema_version": "1.0", "dataset_id": DATASET_ID, "revision": revision,
                "rows": count, "source_split": "train", "license": "CC-BY-NC-4.0",
                "reader": "synchronous_parquet_batches",
                "source_url": f"https://huggingface.co/datasets/{DATASET_ID}/tree/{revision}",
                "raw_sha256": hashlib.sha256((output / "raw.jsonl").read_bytes()).hexdigest(),
                "status": "downloaded_not_executed"}
    write_json(output / "source.json", metadata)
    return metadata


def _iter_pilot_rows(revision: str, batch_size: int, columns: tuple[str, ...]):
    from huggingface_hub import HfApi, HfFileSystem
    import pyarrow.parquet as parquet

    paths = sorted(
        path for path in HfApi().list_repo_files(DATASET_ID, repo_type="dataset", revision=revision)
        if path.startswith("data/train-") and path.endswith(".parquet")
    )
    if not paths:
        raise ValueError("Pinned dataset has no supported train Parquet files")
    filesystem = HfFileSystem()
    for path in paths:
        url = f"datasets/{DATASET_ID}@{revision}/{path}"
        with filesystem.open(url, "rb") as stream:
            # Dataset scanners leave asynchronous Python I/O alive during interpreter exit.
            with parquet.ParquetFile(stream, pre_buffer=False) as source:
                available = source.schema_arrow.names
                if any(name not in available for name in columns[:4]):
                    raise ValueError("KodCode source schema changed")
                selected = [name for name in columns if name in available]
                for batch in source.iter_batches(batch_size=batch_size, columns=selected, use_threads=False):
                    yield from batch.to_pylist()


def prepare_pilot(source: Path, output: Path, *, seed: int = 42, max_tasks: int = 10) -> dict:
    from qwen_repair.executors.bundles import TestBundle, TestCase

    if not 1 <= max_tasks <= 100:
        raise ValueError("M1 accepts at most 100 tasks")
    metadata = json.loads((source / "source.json").read_text())
    raw = (source / "raw.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata["raw_sha256"]:
        raise ValueError("Raw data hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    records, bundles, reasons = [], {}, Counter()
    seen_groups, seen_questions = set(), set()
    for line in raw.decode().splitlines():
        row = json.loads(line)
        if len(records) >= max_tasks:
            reasons["pilot_limit"] += 1
            continue
        try:
            if len(row["question"].encode()) > 6000:
                raise RejectedTask("question_too_long")
            tree, entry = inspect_reference(row["solution"])
            normalized = copy.deepcopy(tree)
            for function in [node for node in ast.walk(normalized) if isinstance(node, ast.FunctionDef)]:
                if function.body and isinstance(function.body[0], ast.Expr) and isinstance(function.body[0].value, ast.Constant) and isinstance(function.body[0].value.value, str):
                    function.body.pop(0)
            group_id = "pilot-" + content_hash(ast.dump(normalized, include_attributes=False))[:24]
            question_key = " ".join(row["question"].lower().split())
            if group_id in seen_groups or question_key in seen_questions:
                raise RejectedTask("exact_duplicate")
            seen_groups.add(group_id)
            seen_questions.add(question_key)
            # M1 uses only the provisional train area; M2 must audit near duplicates globally.
            if int(content_hash([seed, group_id])[:8], 16) % 100 >= 80:
                raise RejectedTask("reserved_non_train_group")
            cases = extract_cases(row["test"], entry.name)
            cases.sort(key=lambda case: content_hash([seed, case]))
            public, private = cases[:len(cases)//2], cases[len(cases)//2:]
            mutations = mutation_candidates(row["solution"], seed)
            task_id = "kodcode-" + content_hash(row["question_id"])[:16]
            signature_node = copy.deepcopy(entry)
            signature_node.body = [ast.Pass()]
            signature = ast.unparse(signature_node).splitlines()[0]
            task = TaskRecord(
                task_id=task_id, source_id=row["question_id"], group_id=group_id, split="train",
                statement=row["question"], signature=signature, entry_point=entry.name,
                buggy_code=mutations[0]["code"], bug_type=mutations[0]["mutation"]["operator"],
                public_tests_ref=f"{task_id}.public", private_tests_ref=f"{task_id}.private",
                reference_ref=f"{task_id}.reference",
            )
            for visibility, subset in (("public", public), ("private", private)):
                bundle = TestBundle(bundle_id=f"{task_id}.{visibility}", entry_point=entry.name,
                                    cases=[TestCase(**case) for case in subset], visibility=visibility)
                bundles[bundle.bundle_id] = bundle.model_dump(mode="json")
            records.append({"task": task.model_dump(mode="json"), "reference_code": row["solution"],
                            "candidates": mutations, "source_test_hash": content_hash(row["test"]),
                            "status": "pending_execution_validation"})
        except (RejectedTask, SyntaxError) as error:
            reasons[str(error) if isinstance(error, RejectedTask) else "syntax_error"] += 1
    with (output / "tasks.jsonl").open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(output / "registry.json", {"schema_version": "1.0", "bundles": list(bundles.values())})
    report = {"schema_version": "1.0", "status": "pending_execution_validation", "source": metadata,
              "seed": seed, "tasks": len(records), "rejected": dict(reasons),
              "limitations": ["Strict literal assertions only; unsupported test files rejected whole.",
                              "Provisional train-only pilot; global near-duplicate and related-ID audit belongs to M2."]}
    report["artifact_hashes"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ("tasks.jsonl", "registry.json")}
    write_json(output / "manifest.json", report)
    return report


def load_pilot(directory: Path, *, require_validated: bool = True) -> tuple[list[dict], dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise ValueError("Unsupported pilot manifest schema")
    if require_validated and manifest.get("status") != "validated":
        raise ValueError("Pilot has not passed real isolated execution validation")
    if manifest.get("status") == "validated":
        if manifest.get("synthetic_demo") is not False:
            raise ValueError("Validated pilot must record real execution")
        validation_hash = manifest.get("validation_sha256")
        if (not validation_hash or not (directory / "validation.json").is_file()
                or hashlib.sha256((directory / "validation.json").read_bytes()).hexdigest() != validation_hash):
            raise ValueError("Pilot validation checksum mismatch")
    hashes = manifest.get("artifact_hashes", {})
    if set(hashes) != {"tasks.jsonl", "registry.json"}:
        raise ValueError("Invalid pilot artifact inventory")
    for name, expected in hashes.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pilot checksum mismatch: {name}")
    records = [json.loads(line) for line in (directory / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != manifest["tasks"]:
        raise ValueError("Pilot record count differs from manifest")
    seen = set()
    for record in records:
        task = TaskRecord.model_validate(record["task"])
        if task.task_id in seen or task.split != "train":
            raise ValueError("Pilot requires unique train tasks")
        seen.add(task.task_id)
    return records, manifest


def validate_pilot(source: Path, output: Path, executor_factory) -> dict:
    records, manifest = load_pilot(source, require_validated=False)
    accepted, reasons, results = [], Counter(), []
    for record in records:
        task = TaskRecord.model_validate(record["task"])
        executor = executor_factory(task)
        if getattr(executor, "synthetic_demo", True):
            raise ValueError("Pilot validation requires a real isolated executor")
        task_results = {"task_id": task.task_id, "reference": [], "candidates": []}

        def run(code, bundle):
            result = executor.run(code, bundle)
            if result.synthetic_demo or result.status == "infra_error":
                raise RuntimeError("Executor unavailable during pilot validation; no data verdict recorded")
            return result

        reference_passed = True
        for _ in range(2):
            for bundle in (task.public_tests_ref, task.private_tests_ref):
                result = run(record["reference_code"], bundle)
                reference_passed = reference_passed and result.status == "passed"
                task_results["reference"].append(result.model_dump(mode="json"))
        if not reference_passed:
            reasons["reference_failed_or_unstable"] += 1
            results.append(task_results)
            continue
        for candidate in record["candidates"]:
            public = run(candidate["code"], task.public_tests_ref)
            private = run(candidate["code"], task.private_tests_ref)
            task_results["candidates"].append({"candidate_hash": candidate["candidate_hash"],
                                               "public": public.model_dump(mode="json"), "private": private.model_dump(mode="json")})
            if public.status == "failed" and private.status == "failed":
                task = task.model_copy(update={"buggy_code": candidate["code"], "bug_type": candidate["mutation"]["operator"]})
                accepted.append({"task": task.model_dump(mode="json"), "reference_code": record["reference_code"],
                                 "mutation": candidate["mutation"], "candidate_hash": candidate["candidate_hash"],
                                 "source_test_hash": record["source_test_hash"], "status": "validated"})
                break
        else:
            reasons["no_public_and_private_failure"] += 1
        results.append(task_results)
    if not accepted:
        raise ValueError(f"No validated pilot tasks: {dict(reasons)}")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "tasks.jsonl").open("x", encoding="utf-8") as stream:
        for record in accepted:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output / "registry.json").write_bytes((source / "registry.json").read_bytes())
    report = {**manifest, "status": "validated", "synthetic_demo": False, "tasks": len(accepted),
              "validation_rejected": dict(reasons), "reference_repetitions": 2}
    report["artifact_hashes"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ("tasks.jsonl", "registry.json")}
    write_json(output / "validation.json", {"synthetic_demo": False, "results": results})
    report["validation_sha256"] = hashlib.sha256((output / "validation.json").read_bytes()).hexdigest()
    write_json(output / "manifest.json", report)
    return report
