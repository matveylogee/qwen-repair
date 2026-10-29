import hashlib
import json

import pytest

from qwen_repair.data.fixtures import load_fixture
from qwen_repair.data import pilot
from qwen_repair.data.pilot import RejectedTask, extract_cases, inspect_reference, load_pilot


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


def test_validated_pilot_requires_unchanged_execution_evidence(tmp_path):
    task = load_fixture("sum_positive").task.model_copy(update={"split": "train"})
    (tmp_path / "tasks.jsonl").write_text(json.dumps({"task": task.model_dump(), "status": "validated"}) + "\n")
    (tmp_path / "registry.json").write_text('{"bundles": []}')
    (tmp_path / "validation.json").write_text('{"synthetic_demo": false, "results": []}')
    manifest = {
        "schema_version": "1.0", "status": "validated", "synthetic_demo": False, "tasks": 1,
        "artifact_hashes": {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                            for name in ("tasks.jsonl", "registry.json")},
        "validation_sha256": hashlib.sha256((tmp_path / "validation.json").read_bytes()).hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert len(load_pilot(tmp_path)[0]) == 1
    (tmp_path / "validation.json").write_text("changed evidence")
    with pytest.raises(ValueError, match="validation checksum"):
        load_pilot(tmp_path)


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
