import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

DEFAULT_DATASET = Path.home() / ".jarvis" / "training" / "classifier_unified" / "valid.jsonl"
DEFAULT_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            try:
                row = json.loads(line)
                messages = row["messages"]
                expected = json.loads(messages[-1]["content"])["intent_class"]
                cases.append({
                    "messages": messages[:-1],
                    "command": messages[-2]["content"],
                    "expected": expected,
                })
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid classifier case on line {lineno}: {exc}") from exc
    return cases


def parse_classifier_output(content: str) -> dict:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("response did not contain a JSON object")
    output = json.loads(content[start:end + 1])
    if not isinstance(output, dict):
        raise ValueError("classifier output must be a JSON object")
    return output


def evaluate(
    cases: list[dict],
    endpoint: str,
    model: str,
    timeout: float,
) -> dict:
    correct = 0
    malformed = 0
    latencies = []
    expected_counts = Counter()
    correct_counts = Counter()
    confusion = defaultdict(Counter)
    failures = []

    with httpx.Client(timeout=timeout) as client:
        for index, case in enumerate(cases, 1):
            started = time.monotonic()
            try:
                response = client.post(
                    f"{endpoint.rstrip('/')}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": case["messages"],
                        "temperature": 0,
                        "max_tokens": 128,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"] or ""
                output = parse_classifier_output(content)
                predicted = output.get("intent_class")
            except Exception as exc:
                predicted = "__malformed__"
                malformed += 1
                content = str(exc)

            elapsed_ms = (time.monotonic() - started) * 1000
            latencies.append(elapsed_ms)
            expected = case["expected"]
            expected_counts[expected] += 1
            confusion[expected][predicted] += 1
            is_correct = predicted == expected
            if is_correct:
                correct += 1
                correct_counts[expected] += 1
            else:
                failures.append({
                    "command": case["command"],
                    "expected": expected,
                    "predicted": predicted,
                    "response": content[:1000],
                })
            print(
                f"[{index:03d}/{len(cases):03d}] "
                f"{'OK' if is_correct else 'FAIL':<4} "
                f"expected={expected:<17} predicted={predicted:<17} "
                f"{elapsed_ms:.0f}ms"
            )

    per_class_accuracy = {
        label: correct_counts[label] / count * 100
        for label, count in sorted(expected_counts.items())
    }
    return {
        "endpoint": endpoint,
        "model": model,
        "total": len(cases),
        "correct": correct,
        "accuracy_pct": correct / len(cases) * 100 if cases else 0,
        "malformed": malformed,
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
        "p50_latency_ms": statistics.median(latencies) if latencies else 0,
        "p95_latency_ms": sorted(latencies)[min(int(len(latencies) * 0.95), len(latencies) - 1)]
        if latencies else 0,
        "per_class_accuracy_pct": per_class_accuracy,
        "confusion": {expected: dict(predicted) for expected, predicted in confusion.items()},
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a classifier through an OpenAI-compatible endpoint.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8090")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = evaluate(load_cases(args.dataset), args.endpoint, args.model, args.timeout)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
