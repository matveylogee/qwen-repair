from qwen_repair.schemas import Message, PromptView


PROMPT_VERSION = "repair-python-v1"


def build_initial_messages(task: PromptView) -> list[Message]:
    return [
        Message(
            role="system",
            content=(
                "Repair the Python function. Return the complete solution.py in exactly one "
                "fenced python block, with no prose. Keep the requested function signature. "
                "After each attempt you receive public feedback."
            ),
        ),
        Message(
            role="user",
            content=(
                f"Task: {task.task_id}\n{task.statement}\n"
                f"Signature: {task.signature}\n"
                f"Buggy implementation:\n```python\n{task.buggy_code.rstrip()}\n```"
            ),
        ),
    ]
