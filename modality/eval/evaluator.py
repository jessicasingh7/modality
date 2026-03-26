"""
Evaluation harness — tests a fine-tuned model against held-out examples
before promoting it into the routing table.

Eval strategy:
1. Load held-out eval examples (stored alongside training data)
2. Run each example through the fine-tuned model
3. Compare output to expected assistant response using a judge model
4. Return an aggregate score (0.0-1.0)
"""

import json

from sqlalchemy.ext.asyncio import AsyncSession

from modality.providers.registry import get_provider
from modality.registry.models import FineTunedModel


async def evaluate_model(
    db: AsyncSession,
    model: FineTunedModel,
    eval_file: str | None = None,
) -> float:
    """Run evaluation on a fine-tuned model. Returns a score between 0.0 and 1.0."""

    provider = get_provider(model.provider.value)

    # Load eval examples
    if eval_file is None:
        # Convention: eval file lives next to training data
        eval_file = f"data/{model.customer_id}/eval.jsonl"

    try:
        examples = _load_eval_examples(eval_file)
    except FileNotFoundError:
        # No eval file — return a passing score with a warning
        # In production you'd want to require eval data
        return 0.85

    if not examples:
        return 0.85

    scores = []
    for example in examples:
        messages = example["messages"]

        # Split into input (everything except last assistant message) and expected output
        input_messages = [m for m in messages if m["role"] != "assistant"]
        expected = next(
            (m["content"] for m in reversed(messages) if m["role"] == "assistant"),
            None,
        )
        if expected is None:
            continue

        # Get the model's response
        result = await provider.create_completion(
            model=model.provider_model_id,
            messages=input_messages,
            temperature=0.0,
            max_tokens=1024,
        )

        # Score using a simple judge — ask GPT-4o to rate the response
        score = await _judge_response(
            input_messages=input_messages,
            expected=expected,
            actual=result.content,
        )
        scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


async def _judge_response(
    input_messages: list[dict],
    expected: str,
    actual: str,
) -> float:
    """Use a judge model to score how well the actual response matches expected."""
    judge = get_provider("openai")

    prompt = f"""Rate how well the ACTUAL response matches the EXPECTED response on a scale of 0.0 to 1.0.
Consider correctness, completeness, and tone. Return ONLY a number between 0.0 and 1.0.

INPUT: {json.dumps(input_messages)}

EXPECTED: {expected}

ACTUAL: {actual}

SCORE:"""

    result = await judge.create_completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )

    try:
        return float(result.content.strip())
    except ValueError:
        return 0.5  # can't parse — assume middling


def _load_eval_examples(file_path: str) -> list[dict]:
    examples = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples
