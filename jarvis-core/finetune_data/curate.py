import argparse
import ast
import json
import random
from collections import Counter
from pathlib import Path

VALID_INTENTS = {"read_only", "prepare", "destructive", "complex_reasoning"}
BAD_TEXT_MARKERS = (
    "ran out of steps",
    "i hit the api rate limit",
    "i'm experiencing an error",
    "please try again or restart jarvis",
)
DEFAULT_TRAINING_LOG = Path.home() / ".jarvis" / "logs" / "training.log"
DEFAULT_OUTPUT_DIR = Path.home() / ".jarvis" / "training" / "executor_dataset"


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"training log not found: {path}")

    records = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[curate] skipping malformed JSON on line {lineno}")
    return records


def _text_blob(record: dict) -> str:
    return " ".join(str(record.get(key) or "") for key in ("display", "speak")).lower()


def is_successful_executor_record(record: dict) -> bool:
    if not record.get("model"):
        return False
    if record.get("intent_class") not in VALID_INTENTS:
        return False
    if not record.get("steps"):
        return False
    if not (record.get("display") or record.get("speak")):
        return False
    text = _text_blob(record)
    if any(marker in text for marker in BAD_TEXT_MARKERS):
        return False
    return True


def parse_step_arguments(step: dict) -> dict:
    raw = step.get("input_summary") or "{}"
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return {"raw_input_summary": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"raw_input_summary": raw}


def build_executor_example(record: dict) -> dict:
    messages = [{"role": "user", "content": record["request"]}]
    for idx, step in enumerate(record.get("steps", []), 1):
        tool_name = step.get("tool", "unknown_tool")
        tool_call_id = f"call_{idx}"
        arguments = parse_step_arguments(step)
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": step.get("result_summary") or "",
        })

    messages.append({
        "role": "assistant",
        "content": record.get("display") or record.get("speak") or "",
    })
    return {
        "messages": messages,
        "meta": {
            "intent_class": record.get("intent_class"),
            "model": record.get("model"),
            "cwd": record.get("cwd"),
            "duration_ms": record.get("duration_ms"),
            "num_steps": len(record.get("steps", [])),
        },
    }


def synthesize_counter_examples() -> list[dict]:
    examples = [
        {
            "messages": [
                {"role": "user", "content": "find every TODO in the src folder"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_content",
                            "arguments": json.dumps(
                                {"pattern": "TODO", "directory": "src"},
                                ensure_ascii=False,
                            ),
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "12 matches in src/"},
                {"role": "assistant", "content": "I found 12 TODO matches in src."},
            ],
            "meta": {"synthetic": True, "failure_mode": "grep_as_text"},
        },
        {
            "messages": [
                {"role": "user", "content": "read the requirements.txt file"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "file_read",
                            "arguments": json.dumps(
                                {"path": "requirements.txt"},
                                ensure_ascii=False,
                            ),
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "fastapi==0.115.0\nhttpx==0.28.1"},
                {"role": "assistant", "content": "The file contains FastAPI and httpx dependencies."},
            ],
            "meta": {"synthetic": True, "failure_mode": "read_file_blindspot"},
        },
        {
            "messages": [
                {"role": "user", "content": "what branch am I on in this repo"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "shell_run",
                            "arguments": json.dumps(
                                {"command": "git branch --show-current"},
                                ensure_ascii=False,
                            ),
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "phase-b-local-first-consolidation"},
                {"role": "assistant", "content": "You are on `phase-b-local-first-consolidation`."},
            ],
            "meta": {"synthetic": True, "failure_mode": "answers_from_memory"},
        },
    ]
    return examples


def split_examples(examples: list[dict], valid_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    items = list(examples)
    random.Random(seed).shuffle(items)
    valid_count = max(1, int(len(items) * valid_ratio)) if items else 0
    if valid_count >= len(items) and len(items) > 1:
        valid_count = len(items) - 1
    valid = items[:valid_count]
    train = items[valid_count:]
    return train, valid


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(records: list[dict], curated: list[dict]) -> dict:
    successful = [r for r in records if is_successful_executor_record(r)]
    return {
        "raw_records": len(records),
        "successful_executor_records": len(successful),
        "models": Counter(r.get("model") for r in successful),
        "intents": Counter(r.get("intent_class") for r in successful),
        "step_counts": Counter(len(r.get("steps", [])) for r in successful),
        "curated_examples": len(curated),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate Jarvis training.log into executor SFT JSONL.")
    parser.add_argument("--training-log", type=Path, default=DEFAULT_TRAINING_LOG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-synthetic", action="store_true")
    args = parser.parse_args()

    records = load_records(args.training_log)
    curated = [build_executor_example(r) for r in records if is_successful_executor_record(r)]
    if args.include_synthetic:
        curated.extend(synthesize_counter_examples())

    train, valid = split_examples(curated, valid_ratio=args.valid_ratio, seed=args.seed)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "valid.jsonl", valid)

    summary = summarize(records, curated)
    summary["train_examples"] = len(train)
    summary["valid_examples"] = len(valid)
    summary["include_synthetic"] = args.include_synthetic
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=lambda x: dict(x)))


if __name__ == "__main__":
    main()
