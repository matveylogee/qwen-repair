import ast
import re


class CandidateError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _signature_shape(arguments: ast.arguments) -> tuple:
    return (
        tuple(arg.arg for arg in arguments.posonlyargs),
        tuple(arg.arg for arg in arguments.args),
        arguments.vararg.arg if arguments.vararg else None,
        tuple(arg.arg for arg in arguments.kwonlyargs),
        arguments.kwarg.arg if arguments.kwarg else None,
        len(arguments.defaults),
        tuple(value is not None for value in arguments.kw_defaults),
    )


def parse_candidate(response: str, entry_point: str, signature: str | None = None) -> str:
    match = re.fullmatch(r"\s*```python[ \t]*\r?\n(.*?)\r?\n```\s*", response, re.DOTALL)
    if match is None or response.count("```") != 2:
        raise CandidateError("format_error", "Return exactly one fenced python block and no prose.")
    code = match.group(1) + "\n"
    try:
        tree = ast.parse(code, filename="solution.py")
        compile(tree, "solution.py", "exec")
    except (SyntaxError, ValueError) as error:
        line = getattr(error, "lineno", None)
        location = f" at line {line}" if line else ""
        raise CandidateError("syntax_error", f"Invalid Python syntax{location}.") from error

    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
    ]
    if len(functions) != 1 or not isinstance(functions[0], ast.FunctionDef):
        raise CandidateError("entry_point_error", f"Define one top-level function named {entry_point}.")
    if signature:
        declaration = signature.strip()
        if not declaration.startswith("def "):
            declaration = f"def {declaration}"
        declaration = declaration.rstrip(":") + ":\n    pass\n"
        try:
            expected = ast.parse(declaration).body[0]
        except (SyntaxError, ValueError) as error:
            raise ValueError("Task has an invalid expected signature") from error
        if not isinstance(expected, ast.FunctionDef) or expected.name != entry_point:
            raise ValueError("Task signature does not match its entry point")
        if _signature_shape(expected.args) != _signature_shape(functions[0].args):
            raise CandidateError("signature_error", "Keep the expected function parameters.")
    return code
