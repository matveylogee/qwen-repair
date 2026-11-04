import hashlib
import json

import pytest

from qwen_repair.data.fixtures import load_fixture
from qwen_repair.data import pilot
from qwen_repair.data.pilot import RejectedTask, content_hash, extract_cases, inspect_reference, load_pilot
from qwen_repair.executors.bundles import BundleRegistry, registry_from_fixtures
from qwen_repair.schemas import TestResult


def test_extracts_literal_cases_without_executing_reference_or_tests(tmp_path):
    marker = tmp_path / "must-not-exist"
    source = f"def solve(x):\n    open({str(marker)!r}, 'w').close()\n    return x\n"
    with pytest.raises(RejectedTask, match="reference_io_or_dynamic_code"):
        inspect_reference(source)
    test_source = "from solution import solve\ndef test_values():\n" + "".join(
        f"    assert solve({value}) == {value}\n" for value in range(4)
    )
    assert len(extract_cases(test_source, "solve")) == 4
    assert not marker.exists()


@pytest.mark.parametrize("literal", ["(1, 2)", "{1, 2}", "b'bytes'", "[{'nested': (1,)}]"])
def test_non_json_python_types_are_rejected_without_changing_test_semantics(literal):
    source = f"def test_values():\n    assert solve(1) == {literal}\n"
    with pytest.raises(RejectedTask, match="non_json_test_value"):
        extract_cases(source, "solve")


def test_positional_only_pytest_arguments_are_rejected():
    source = "def test_values(fixture, /):\n    assert solve(1) == 1\n"
    with pytest.raises(RejectedTask, match="test_fixture_or_parameterization"):
        extract_cases(source, "solve")


def test_entry_point_shadowing_is_rejected():
    source = "def test_values():\n    solve = 1\n    assert solve(1) == 1\n"
    with pytest.raises(RejectedTask, match="test_shadows_entry_point"):
        extract_cases(source, "solve")


def test_duplicate_test_names_are_rejected_instead_of_collecting_shadowed_assertions():
    source = "def test_values():\n    assert solve(1) == 1\ndef test_values():\n    assert solve(2) == 2\n"
    with pytest.raises(RejectedTask, match="duplicate_test_name"):
        extract_cases(source, "solve")


def test_relative_solution_import_is_rejected():
    source = "from .solution import solve\ndef test_values():\n    assert solve(1) == 1\n"
    with pytest.raises(RejectedTask, match="test_import"):
        extract_cases(source, "solve")


def test_duplicate_keyword_arguments_are_not_silently_overwritten():
    source = "def test_values():\n    assert solve(x=1, x=2) == 2\n"
    with pytest.raises(RejectedTask, match="duplicate_test_keyword"):
        extract_cases(source, "solve")


@pytest.mark.parametrize("body", [
    "    values = [1]\n    assert solve(values) == 1\n    assert solve(values) == 2\n",
    "    values = [1]\n    assert solve(values, values) == 1\n",
    "    values = [1]\n    assert solve(values) == values\n",
])
def test_reused_mutable_values_cannot_change_case_semantics(body):
    with pytest.raises(RejectedTask, match="reused_mutable_test_value"):
        extract_cases("def test_values():\n" + body, "solve")


def test_mutable_alias_assignment_is_rejected():
    source = "def test_values():\n    values = [1]\n    alias = values\n    assert solve(alias) == 1\n"
    with pytest.raises(RejectedTask, match="mutable_test_alias"):
        extract_cases(source, "solve")


def test_independent_literal_lists_are_kept_as_independent_cases():
    source = "def test_values():\n" + "".join(
        f"    assert solve([{value}]) == {value}\n" for value in range(4)
    )
    cases = extract_cases(source, "solve")
    assert [case["args"] for case in cases] == [[[0]], [[1]], [[2]], [[3]]]


def _pilot_bundle(tmp_path, *, validated=True):
    fixture = load_fixture("sum_positive")
    task = fixture.task.model_copy(update={"split": "train"})
    registry = registry_from_fixtures()
    bundles = {key: registry.get(key) for key in (task.public_tests_ref, task.private_tests_ref)}
    BundleRegistry(bundles).save(tmp_path / "registry.json")
    status = "validated" if validated else "pending_execution_validation"
    candidate_hash = content_hash(task.buggy_code)
    record = {"task": task.model_dump(), "status": status, "reference_code": fixture.reference_code,
              "candidate_hash": candidate_hash, "source_test_hash": "test-fixture",
              "candidates": [{"code": task.buggy_code, "candidate_hash": candidate_hash,
                              "mutation": {"operator": task.bug_type}}]}
    (tmp_path / "tasks.jsonl").write_text(json.dumps(record) + "\n")
    def result(bundle_id, passed):
        count = len(bundles[bundle_id].cases)
        return TestResult(status="passed" if passed else "failed", tests_total=count,
                          tests_passed=count if passed else 0, tests_failed=0 if passed else count,
                          synthetic_demo=False).model_dump()
    evidence = {"task_id": task.task_id,
                "reference": [result(key, True) for _ in range(2) for key in bundles],
                "candidates": [{"candidate_hash": candidate_hash, "public": result(task.public_tests_ref, False),
                                "private": result(task.private_tests_ref, False)}]}
    (tmp_path / "validation.json").write_text(json.dumps({"synthetic_demo": False, "results": [evidence]}))
    manifest = {
        "schema_version": "1.0", "status": status, "synthetic_demo": False, "tasks": 1,
        "artifact_hashes": {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                            for name in ("tasks.jsonl", "registry.json")},
        "validation_sha256": hashlib.sha256((tmp_path / "validation.json").read_bytes()).hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return task


def _refresh_hash(tmp_path, filename):
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    digest = hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
    if filename == "validation.json":
        manifest["validation_sha256"] = digest
    else:
        manifest["artifact_hashes"][filename] = digest
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))


def test_validated_pilot_requires_unchanged_execution_evidence(tmp_path):
    _pilot_bundle(tmp_path)
    assert len(load_pilot(tmp_path)[0]) == 1
    (tmp_path / "validation.json").write_text("changed evidence")
    with pytest.raises(ValueError, match="validation checksum"):
        load_pilot(tmp_path)


@pytest.mark.parametrize("defect", ["empty", "synthetic", "candidate", "reference_failed", "wrong_count"])
def test_validated_pilot_checks_evidence_semantics_even_with_matching_hash(tmp_path, defect):
    _pilot_bundle(tmp_path)
    path = tmp_path / "validation.json"
    evidence = json.loads(path.read_text())
    result = evidence["results"][0]
    if defect == "empty":
        evidence["results"] = []
    elif defect == "synthetic":
        result["reference"][0]["synthetic_demo"] = True
    elif defect == "candidate":
        result["candidates"][0]["candidate_hash"] = "unrelated"
    elif defect == "reference_failed":
        result["reference"][0] = result["candidates"][0]["public"]
    else:
        result["reference"][0]["tests_total"] += 1
        result["reference"][0]["tests_passed"] += 1
    path.write_text(json.dumps(evidence))
    _refresh_hash(tmp_path, "validation.json")
    with pytest.raises(ValueError):
        load_pilot(tmp_path)


def test_public_private_overlap_rejected_even_with_matching_artifact_hash(tmp_path):
    _pilot_bundle(tmp_path)
    path = tmp_path / "registry.json"
    registry = json.loads(path.read_text())
    registry["bundles"][1]["cases"][0] = registry["bundles"][0]["cases"][0]
    path.write_text(json.dumps(registry))
    _refresh_hash(tmp_path, "registry.json")
    with pytest.raises(ValueError, match="disjoint independent inputs"):
        load_pilot(tmp_path)


def test_equivalent_keyword_positional_calls_do_not_inflate_independent_case_count():
    _, entry = inspect_reference("def solve(x, y=1):\n    return x+y\n")
    source = "def test_values():\n" + "".join(f"    assert {call} == {expected}\n" for call, expected in [
        ("solve(1)", 2), ("solve(x=1, y=1)", 2), ("solve(2)", 3), ("solve(3)", 4)
    ])
    with pytest.raises(RejectedTask, match="fewer_than_four_independent_cases"):
        extract_cases(source, "solve", entry)


def test_conflicting_expectations_for_same_input_are_rejected():
    source = "def test_values():\n    assert solve(1) == 1\n    assert solve(1) == 2\n"
    with pytest.raises(RejectedTask, match="conflicting_test_expectations"):
        extract_cases(source, "solve")


def test_stateful_mutable_default_is_rejected():
    with pytest.raises(RejectedTask, match="mutable_reference_default"):
        inspect_reference("def solve(x, cache=[]):\n    cache.append(x)\n    return len(cache)\n")


def test_review_requires_all_ids_and_preserves_prevalidation_status(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    task = _pilot_bundle(source, validated=False)
    with pytest.raises(ValueError, match="every source ID"):
        pilot.review_pilot(source, tmp_path / "bad", decisions={}, review_id="review-1")
    report = pilot.review_pilot(source, tmp_path / "reviewed", review_id="review-1", decisions={
        task.source_id: {"verdict": "include", "reason": "Manually checked independent cases and contract."},
    })
    assert report["status"] == "pending_execution_validation"
    assert len(load_pilot(tmp_path / "reviewed", require_validated=False)[0]) == 1
    with pytest.raises(ValueError, match="not passed real isolated"):
        load_pilot(tmp_path / "reviewed")
    (tmp_path / "reviewed" / "review.json").write_text("changed review")
    with pytest.raises(ValueError, match="review checksum"):
        load_pilot(tmp_path / "reviewed", require_validated=False)


def test_fetch_stops_at_limit_and_closes_stream_before_returning(tmp_path, monkeypatch):
    observed = {"consumed": 0, "closed": False}

    def source(revision, batch_size, columns):
        try:
            for index in range(10):
                observed["consumed"] += 1
                yield {"question_id": str(index), "question": "q", "solution": "s", "test": "t"}
        finally:
            observed["closed"] = True

    monkeypatch.setattr(pilot, "_iter_pilot_rows", source)
    destination = tmp_path / "fetched"
    report = pilot.fetch_pilot(destination, limit=3)
    assert observed == {"consumed": 3, "closed": True}
    assert report["rows"] == len((destination / "raw.jsonl").read_text().splitlines()) == 3
    assert report["status"] == "downloaded_not_executed"
