"""Container-only entry point. Never import this module in the host process."""

import contextlib
import importlib.util
import json
import math
import sys


def is_json_value(value, depth=0) -> bool:
    if depth > 100:
        return False
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(is_json_value(item, depth + 1) for item in value)
    if type(value) is dict:
        return all(type(key) is str and is_json_value(item, depth + 1) for key, item in value.items())
    return False


def main() -> None:
    request = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        spec = importlib.util.spec_from_file_location("solution", "/input/solution.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        answer = getattr(module, request["entry_point"])(
            *request["args"], **request["kwargs"]
        )
    if not is_json_value(answer):
        raise ValueError("The function must return JSON-native values")
    print(json.dumps({"value": answer}, allow_nan=False, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Exception strings may contain candidate-controlled paths or private inputs.
        sys.exit(1)
