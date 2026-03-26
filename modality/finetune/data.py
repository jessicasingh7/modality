"""Training data validation and formatting."""

import json
from pathlib import Path


class ValidationError(Exception):
    pass


def validate_jsonl(file_path: str) -> list[dict]:
    """Validate a JSONL training file matches the expected chat format.

    Expected format per line:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    path = Path(file_path)
    if not path.exists():
        raise ValidationError(f"File not found: {file_path}")
    if path.suffix != ".jsonl":
        raise ValidationError(f"Expected .jsonl file, got: {path.suffix}")

    examples = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Line {i}: invalid JSON — {e}")

            if "messages" not in obj:
                raise ValidationError(f"Line {i}: missing 'messages' key")

            messages = obj["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValidationError(f"Line {i}: 'messages' must have at least 2 entries")

            for msg in messages:
                if "role" not in msg or "content" not in msg:
                    raise ValidationError(
                        f"Line {i}: each message needs 'role' and 'content'"
                    )
                if msg["role"] not in ("system", "user", "assistant"):
                    raise ValidationError(
                        f"Line {i}: invalid role '{msg['role']}'"
                    )

            examples.append(obj)

    if len(examples) < 10:
        raise ValidationError(f"Need at least 10 training examples, got {len(examples)}")

    return examples


def split_train_eval(examples: list[dict], eval_fraction: float = 0.1) -> tuple[list[dict], list[dict]]:
    """Split examples into training and evaluation sets."""
    split_idx = max(1, int(len(examples) * (1 - eval_fraction)))
    return examples[:split_idx], examples[split_idx:]


def write_jsonl(examples: list[dict], output_path: str) -> str:
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return output_path
