import json
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


class TestCase(BaseModel):
    __test__ = False
    model_config = ConfigDict(extra="forbid", frozen=True)

    args: list[JsonValue] = Field(default_factory=list)
    kwargs: dict[str, JsonValue] = Field(default_factory=dict)
    expected: JsonValue

    @model_validator(mode="after")
    def require_finite_json(self) -> "TestCase":
        if len(canonical_json(self.model_dump(mode="json"))) > 1048576:
            raise ValueError("Test case exceeds the JSON byte limit")
        return self


class TestBundle(BaseModel):
    __test__ = False
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    bundle_id: str = Field(min_length=1)
    entry_point: str
    cases: list[TestCase] = Field(min_length=1, max_length=1000)
    visibility: Literal["public", "private"]

    @field_validator("entry_point")
    @classmethod
    def validate_entry_point(cls, value: str) -> str:
        if not value.isidentifier() or value.startswith("__"):
            raise ValueError("Expected a named Python function")
        return value

    @property
    def bundle_hash(self) -> str:
        return sha256(canonical_json(self.model_dump(mode="json"))).hexdigest()


class BundleRegistry:
    def __init__(self, bundles: dict[str, TestBundle]):
        if any(key != bundle.bundle_id for key, bundle in bundles.items()):
            raise ValueError("Registry keys must match bundle IDs")
        self._bundles = {
            key: bundle.model_copy(deep=True) for key, bundle in bundles.items()
        }

    def get(self, bundle_id: str) -> TestBundle:
        try:
            return self._bundles[bundle_id].model_copy(deep=True)
        except KeyError as error:
            raise ValueError("Test bundle is not in the trusted local registry") from error

    def hashes(self) -> dict[str, str]:
        return {key: bundle.bundle_hash for key, bundle in self._bundles.items()}

    def save(self, path: Path) -> None:
        payload = {
            "schema_version": "1.0",
            "bundles": [bundle.model_dump(mode="json") for bundle in self._bundles.values()],
        }
        path.write_bytes(canonical_json(payload) + b"\n")


def load_registry(path: str | Path) -> BundleRegistry:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "bundles"} or payload["schema_version"] != "1.0":
        raise ValueError("Expected a versioned registry with bundles")
    if not isinstance(payload["bundles"], list):
        raise ValueError("Registry bundles must be a list")
    bundles = [TestBundle.model_validate(item) for item in payload["bundles"]]
    if len({bundle.bundle_id for bundle in bundles}) != len(bundles):
        raise ValueError("Duplicate bundle IDs")
    return BundleRegistry({bundle.bundle_id: bundle for bundle in bundles})


def export_registry(registry: BundleRegistry, path: str | Path) -> None:
    registry.save(Path(path))


def registry_from_fixtures() -> BundleRegistry:
    from qwen_repair.data.fixtures import FIXTURE_NAMES, load_fixture

    bundles = {}
    for name in FIXTURE_NAMES:
        fixture = load_fixture(name)
        for visibility in ("public", "private"):
            bundle_id = getattr(fixture.task, f"{visibility}_tests_ref")
            cases = getattr(fixture, f"{visibility}_cases")
            bundles[bundle_id] = TestBundle(
                bundle_id=bundle_id,
                entry_point=fixture.task.entry_point,
                visibility=visibility,
                cases=[TestCase(args=case["args"], expected=case["expected"]) for case in cases],
            )
    return BundleRegistry(bundles)
