import ast
from copy import deepcopy
import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from qwen_repair.schemas import TestResult

from .bundles import BundleRegistry, TestBundle, TestCase, canonical_json


class DockerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    image: str
    timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    startup_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    memory_mb: int = Field(default=512, ge=64, le=4096)
    pids_limit: int = Field(default=32, ge=8, le=128)
    cpus: float = Field(default=1.0, gt=0, le=4)
    tmpfs_mb: int = Field(default=32, ge=1, le=128)
    max_output_bytes: int = Field(default=65536, ge=1024, le=1048576)
    max_candidate_bytes: int = Field(default=65536, ge=1024, le=1048576)

    @field_validator("image")
    @classmethod
    def require_digest(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[0-9a-f]{64}", value) is None:
            raise ValueError("Docker image must be pinned by a complete sha256 digest")
        return value


@dataclass
class ContainerOutput:
    status: str
    stdout: bytes = b""
    duration_seconds: float = 0.0


def _json_equal(actual: object, expected: object) -> bool:
    canonical_json(actual)
    canonical_json(expected)
    return actual == expected


class DockerExecutor:
    synthetic_demo = False
    runner_version = "docker-json-v1"

    def __init__(self, registry: BundleRegistry, config: DockerConfig):
        self.registry = registry
        self.config = config
        self._preflight_passed = False
        self._cancelled = threading.Event()
        self.preflight_report: dict | None = None

    def _create_command(self, name: str, inputs: Path) -> list[str]:
        config = self.config
        return [
            "docker", "create", "--name", name, "--pull", "never", "--interactive",
            "--init", "--network", "none", "--read-only", "--user", "65532:65532",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            "--pids-limit", str(config.pids_limit), "--cpus", str(config.cpus),
            "--memory", f"{config.memory_mb}m", "--memory-swap", f"{config.memory_mb}m",
            "--tmpfs", f"/tmp:rw,noexec,nosuid,nodev,size={config.tmpfs_mb}m",
            "--env", "NVIDIA_VISIBLE_DEVICES=void",
            "--shm-size", "1m", "--ulimit", "nofile=64:64",
            "--ulimit", f"fsize={config.max_output_bytes}:{config.max_output_bytes}",
            "--no-healthcheck", "--log-driver", "none", "--workdir", "/tmp",
            "--mount", f"type=bind,source={inputs},target=/input,readonly",
            "--entrypoint", "python", config.image,
            "-I", "-B", "/input/runner.py",
        ]

    def _remove_container(self, name: str) -> bool:
        try:
            result = subprocess.run(
                ["docker", "rm", "--force", "--volumes", name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=self.config.startup_timeout_seconds, check=False,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _container_state(self, name: str) -> dict | None:
        try:
            inspection = subprocess.run(
                ["docker", "inspect", "--format", "{{json .State}}", name],
                capture_output=True, text=True, timeout=self.config.startup_timeout_seconds,
                check=False,
            )
            if inspection.returncode:
                return None
            return json.loads(inspection.stdout)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return None

    def _capture(self, command: list[str], payload: bytes) -> ContainerOutput:
        started = time.monotonic()
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, start_new_session=True,
        )
        selector = selectors.DefaultSelector()
        stdout = bytearray()
        consumed = 0
        status = "ok"
        try:
            os.set_blocking(process.stdin.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            position = 0
            while selector.get_map():
                if self._cancelled.is_set():
                    status = "infra_error"
                    break
                remaining = self.config.timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    status = "timeout"
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    if key.data == "stdin":
                        try:
                            position += os.write(key.fileobj.fileno(), payload[position:position + 4096])
                        except BrokenPipeError:
                            position = len(payload)
                        if position == len(payload):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                        continue
                    chunk = os.read(key.fileobj.fileno(), 4096)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    consumed += len(chunk)
                    if consumed > self.config.max_output_bytes:
                        status = "resource_limit"
                        break
                    if key.data == "stdout":
                        stdout.extend(chunk)
                if status != "ok":
                    break
            if status == "ok":
                try:
                    returncode = process.wait(timeout=max(0.01, self.config.timeout_seconds - (time.monotonic() - started)))
                except subprocess.TimeoutExpired:
                    status = "timeout"
                else:
                    if returncode in (125, 126, 127):
                        status = "infra_error"
                    elif returncode == 137:
                        status = "resource_limit"
                    elif returncode:
                        status = "runtime_error"
        finally:
            selector.close()
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            for stream in (process.stdin, process.stdout, process.stderr):
                if not stream.closed:
                    stream.close()
        return ContainerOutput(status, bytes(stdout), time.monotonic() - started)

    def _execute_case(self, candidate: str, entry_point: str, case: TestCase) -> ContainerOutput:
        name = f"qrepair-{uuid4().hex}"
        created = False
        result = ContainerOutput("infra_error")
        with tempfile.TemporaryDirectory(prefix="qrepair-input-") as directory:
            inputs = Path(directory)
            inputs.chmod(0o755)
            (inputs / "solution.py").write_text(candidate, encoding="utf-8")
            shutil.copyfile(Path(__file__).with_name("_candidate_runner.py"), inputs / "runner.py")
            for path in inputs.iterdir():
                path.chmod(0o444)
            try:
                creation = subprocess.run(
                    self._create_command(name, inputs), stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, timeout=self.config.startup_timeout_seconds,
                    check=False,
                )
                created = creation.returncode == 0
                if created:
                    payload = canonical_json({
                        "entry_point": entry_point, "args": case.args, "kwargs": case.kwargs,
                    })
                    result = self._capture(["docker", "start", "--attach", "--interactive", name], payload)
                    if result.status in ("ok", "runtime_error"):
                        state = self._container_state(name)
                        if not isinstance(state, dict) or state.get("Status") != "exited" or state.get("Error"):
                            result.status = "infra_error"
                        elif state.get("OOMKilled"):
                            result.status = "resource_limit"
                        elif state.get("ExitCode") != 0:
                            result.status = "runtime_error"
                        elif result.status != "ok":
                            result.status = "infra_error"
            except (OSError, subprocess.TimeoutExpired):
                result = ContainerOutput("infra_error")
            finally:
                # Removal also covers a container created just before the CLI timed out.
                removed = self._remove_container(name)
                if created and not removed:
                    self._preflight_passed = False
                    if self.preflight_report is not None:
                        self.preflight_report["passed"] = False
                    result = ContainerOutput("infra_error")
        return result

    def _run_bundle(self, candidate: str, bundle: TestBundle) -> TestResult:
        started = time.monotonic()
        passed = failed = 0
        status = "passed"
        diagnostic = "All cases passed."
        for index, case in enumerate(bundle.cases, start=1):
            output = self._execute_case(candidate, bundle.entry_point, case)
            if output.status != "ok":
                status = output.status
                diagnostic = f"Case {index}: {status}."
                break
            try:
                response = json.loads(output.stdout)
                if not isinstance(response, dict) or set(response) != {"value"}:
                    raise ValueError("Expected a single JSON return value")
                success = _json_equal(response["value"], case.expected)
            except (UnicodeError, ValueError, TypeError):
                status = "runtime_error"
                diagnostic = f"Case {index}: invalid return-value protocol."
                break
            if success:
                passed += 1
            else:
                failed += 1
                if failed == 1:
                    diagnostic = f"Case {index}: returned an incorrect value."
        if status == "passed" and failed:
            status = "failed"
        return TestResult(
            status=status, tests_passed=passed, tests_failed=failed,
            tests_total=len(bundle.cases), feedback=diagnostic,
            duration_seconds=time.monotonic() - started, synthetic_demo=False,
        )

    def run(self, candidate: str, test_bundle_id: str) -> TestResult:
        self._cancelled.clear()
        if not self._preflight_passed:
            return TestResult(status="infra_error", feedback="Docker preflight has not passed in this process.", synthetic_demo=False)
        try:
            bundle = self.registry.get(test_bundle_id)
        except ValueError:
            return TestResult(status="infra_error", feedback="Unknown trusted test bundle.", synthetic_demo=False)
        if len(candidate.encode("utf-8")) > self.config.max_candidate_bytes:
            return TestResult(status="resource_limit", feedback="Candidate exceeds the byte limit.", synthetic_demo=False)
        try:
            ast.parse(candidate, filename="solution.py")
        except (SyntaxError, ValueError, RecursionError):
            return TestResult(status="syntax_error", feedback="Invalid Python syntax.", synthetic_demo=False)
        return self._run_bundle(candidate, bundle)

    def cancel_current(self) -> None:
        self._cancelled.set()

    def preflight(self) -> dict:
        if self._preflight_passed and self.preflight_report and self.preflight_report.get("passed"):
            return deepcopy(self.preflight_report)
        self._preflight_passed = False
        self._cancelled.clear()
        report = {
            "schema_version": "1.0", "backend": self.runner_version,
            "synthetic_demo": False, "image": self.config.image,
            "config": self.config.model_dump(), "passed": False, "checks": {},
            "limitations": "Probe results do not prove absolute container isolation.",
        }
        self.preflight_report = report
        if shutil.which("docker") is None:
            report["checks"]["docker_available"] = False
            return report
        try:
            daemon = subprocess.run(
                ["docker", "info", "--format", "{{.OSType}}"], capture_output=True,
                text=True, timeout=self.config.startup_timeout_seconds, check=False,
            )
            image = subprocess.run(
                ["docker", "image", "inspect", self.config.image, "--format", "{{.Id}}"],
                capture_output=True, text=True, timeout=self.config.startup_timeout_seconds, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            report["checks"]["docker_available"] = False
            return report
        report["checks"]["docker_available"] = daemon.returncode == 0 and daemon.stdout.strip() == "linux"
        report["checks"]["pinned_image_available"] = image.returncode == 0 and image.stdout.strip().startswith("sha256:")
        if not all(report["checks"].values()):
            return report
        with tempfile.TemporaryDirectory(prefix="qrepair-host-probe-") as host_directory:
            host_file = Path(host_directory) / "host-canary"
            host_file.write_text(uuid4().hex, encoding="utf-8")
            probes = {
                "normal": ("def probe():\n    return 7\n", 7, "passed"),
                "timeout": ("def probe():\n    while True:\n        pass\n", None, "timeout"),
                "output_limit": ("def probe():\n    import os\n    while True:\n        os.write(1, b'x' * 8192)\n", None, "resource_limit"),
                "network": (
                    "def probe():\n    import socket\n    from pathlib import Path\n    active = [path.name for path in Path('/sys/class/net').iterdir() if (path / 'flags').is_file() and int((path / 'flags').read_text(), 16) & 1]\n    routes = Path('/proc/net/route').read_text().splitlines()[1:]\n    sock = socket.socket()\n    sock.settimeout(0.25)\n    try:\n        sock.connect(('1.1.1.1', 443))\n        return False\n    except OSError as error:\n        return error.errno in (101, 13, 1) and not routes and set(active) <= {'lo'}\n    finally:\n        sock.close()\n", True, "passed",
                ),
                "host_files": (f"def probe():\n    import os\n    return not os.path.exists({str(host_file)!r}) and not os.path.exists('/var/run/docker.sock')\n", True, "passed"),
                "privileges_readonly": (
                    "def probe():\n    import os\n    status = open('/proc/self/status').read()\n    try:\n        open('/qrepair-write-probe', 'w').close()\n        return False\n    except OSError:\n        return os.getuid() == 65532 and 'NoNewPrivs:\\t1' in status and 'CapEff:\\t0000000000000000' in status and 'Seccomp:\\t2' in status\n", True, "passed",
                ),
                "cgroup_limits": (
                    "def probe():\n    from pathlib import Path\n    import os\n    root = Path('/sys/fs/cgroup')\n    if (root / 'cgroup.controllers').exists():\n        memory = int((root / 'memory.max').read_text())\n        quota, period = map(int, (root / 'cpu.max').read_text().split())\n        pids = int((root / 'pids.max').read_text())\n        swap = int((root / 'memory.swap.max').read_text())\n    else:\n        memory = int((root / 'memory/memory.limit_in_bytes').read_text())\n        quota = int((root / 'cpu/cpu.cfs_quota_us').read_text())\n        period = int((root / 'cpu/cpu.cfs_period_us').read_text())\n        pids = int((root / 'pids/pids.max').read_text())\n        swap = int((root / 'memory/memory.memsw.limit_in_bytes').read_text()) - memory\n    scratch = os.statvfs('/tmp')\n"
                    f"    return memory == {self.config.memory_mb * 1024 * 1024} and 0 < quota / period <= {self.config.cpus * 1.001!r} and pids == {self.config.pids_limit} and swap == 0 and scratch.f_blocks * scratch.f_frsize <= {self.config.tmpfs_mb * 1024 * 1024}\n",
                    True, "passed",
                ),
                "process_limit": (
                    f"def probe():\n    import subprocess\n    children = []\n    limited = False\n    try:\n        for _ in range({self.config.pids_limit + 4}):\n            children.append(subprocess.Popen(['python', '-c', 'import time; time.sleep(10)']))\n    except OSError:\n        limited = True\n    finally:\n        for child in children:\n            child.kill()\n        for child in children:\n            child.wait()\n    return limited\n", True, "passed",
                ),
            }
            for name, (candidate, expected, expected_status) in probes.items():
                bundle = TestBundle(bundle_id=f"preflight.{name}", entry_point="probe", visibility="private", cases=[TestCase(expected=expected)])
                result = self._run_bundle(candidate, bundle)
                report["checks"][name] = result.status == expected_status
        report["passed"] = all(report["checks"].values())
        self._preflight_passed = report["passed"]
        return report
