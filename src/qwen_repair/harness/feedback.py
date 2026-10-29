import re

from qwen_repair.schemas import TestResult

from .parser import CandidateError


def truncate_to_budget(text: str, max_tokens: int) -> str:
    return text.encode("utf-8")[:max_tokens].decode("utf-8", errors="ignore")


def sanitize_diagnostic(text: str) -> str:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    lines = []
    in_traceback = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("Traceback", "File ", "During handling", "The above exception")):
            in_traceback = True
            continue
        if in_traceback and raw_line[:1].isspace():
            continue
        in_traceback = False
        line = re.sub(r"(?:[A-Za-z]:[\\/]|/)[^\s\"'<>]*", "[path]", line)
        lines.append(line)
        if len(lines) == 2:
            break
    return "\n".join(lines)


def format_feedback(result: TestResult, max_tokens: int = 256) -> str:
    header = f"Public status: {result.status}; passed {result.tests_passed}/{result.tests_total}."
    detail = sanitize_diagnostic(result.feedback)
    return truncate_to_budget(header + (f"\n{detail}" if detail else ""), max_tokens)


def format_parse_error(error: CandidateError, max_tokens: int = 256) -> str:
    return truncate_to_budget(f"Public status: {error.code}. {error.detail}", max_tokens)
