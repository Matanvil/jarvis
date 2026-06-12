import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier_prompt import CLASSIFY_SYSTEM_PROMPT

VALID_INTENTS = {"read_only", "prepare", "destructive", "complex_reasoning"}
DEFAULT_TRAINING_LOG = Path.home() / ".jarvis" / "logs" / "training.log"
DEFAULT_EXISTING_DIR = Path.home() / ".jarvis" / "training" / "classifier_dataset"
DEFAULT_OUTPUT_DIR = Path.home() / ".jarvis" / "training" / "classifier_unified"
DEFAULT_SEED_FILES = (
    Path.home() / "Desktop" / "seeds.json",
    Path.home() / "Desktop" / "seeds2.json",
    Path.home() / "Desktop" / "seeds3.json",
)

DEFAULT_REASONS = {
    "read_only": "The request can be answered using local read-only tools or local information.",
    "prepare": "The request asks Jarvis to prepare or make a reversible change.",
    "destructive": "The request asks for a destructive, irreversible, or approval-gated action.",
    "complex_reasoning": "The request needs current external information or complex cloud reasoning.",
}


def normalize_command(command: str) -> str:
    return " ".join(command.casefold().split())


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[curate-classifier] skipping malformed JSON: {path}:{lineno}")
    return rows


def load_existing_examples(dataset_dir: Path) -> list[dict]:
    examples = []
    for path in (dataset_dir / "train.jsonl", dataset_dir / "valid.jsonl"):
        for row in load_jsonl(path):
            messages = row.get("messages") or []
            if len(messages) < 2:
                continue
            try:
                output = json.loads(messages[-1]["content"])
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            examples.append({
                "command": messages[-2].get("content", ""),
                "intent_class": output.get("intent_class"),
                "reason": output.get("reason"),
                "source": f"existing:{path.name}",
            })
    return examples


def load_seed_examples(paths: tuple[Path, ...]) -> list[dict]:
    examples = []
    for path in paths:
        if not path.exists():
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[curate-classifier] skipping malformed JSON: {path}")
            continue
        for row in rows:
            examples.append({
                "command": row.get("command", ""),
                "intent_class": row.get("expected_class"),
                "reason": None,
                "source": f"seed:{path.name}",
            })
    return examples


def load_training_log_examples(path: Path) -> list[dict]:
    examples = []
    for row in load_jsonl(path):
        examples.append({
            "command": row.get("request", ""),
            "intent_class": row.get("intent_class"),
            "reason": row.get("reason"),
            "source": f"training_log:{row.get('source') or 'interaction'}",
        })
    return examples


def unify_examples(candidates: list[dict]) -> tuple[list[dict], dict]:
    grouped = defaultdict(list)
    rejected_invalid = 0
    for candidate in candidates:
        command = candidate.get("command", "").strip()
        intent = candidate.get("intent_class")
        if not command or intent not in VALID_INTENTS:
            rejected_invalid += 1
            continue
        grouped[normalize_command(command)].append({**candidate, "command": command})

    unified = []
    conflicts = []
    duplicate_candidates = 0
    for normalized, group in grouped.items():
        labels = {item["intent_class"] for item in group}
        duplicate_candidates += len(group) - 1
        if len(labels) != 1:
            conflicts.append({
                "normalized_command": normalized,
                "labels": sorted(labels),
                "sources": sorted({item["source"] for item in group}),
            })
            continue

        chosen = next((item for item in group if item.get("reason")), group[0])
        unified.append(chosen)

    unified.sort(key=lambda item: normalize_command(item["command"]))
    summary = {
        "input_candidates": len(candidates),
        "rejected_invalid": rejected_invalid,
        "duplicate_candidates": duplicate_candidates,
        "conflicting_commands": len(conflicts),
        "conflicts": conflicts,
    }
    return unified, summary


def build_classifier_example(example: dict) -> dict:
    intent = example["intent_class"]
    output = {
        "can_handle_locally": intent != "complex_reasoning",
        "intent_class": intent,
        "reason": example.get("reason") or DEFAULT_REASONS[intent],
    }
    return {
        "messages": [
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": example["command"]},
            {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)},
        ]
    }


def stratified_split(examples: list[dict], valid_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    by_label = defaultdict(list)
    for example in examples:
        output = json.loads(example["messages"][-1]["content"])
        by_label[output["intent_class"]].append(example)

    rng = random.Random(seed)
    train = []
    valid = []
    for label in sorted(by_label):
        items = by_label[label]
        rng.shuffle(items)
        valid_count = max(1, round(len(items) * valid_ratio)) if len(items) > 1 else 0
        valid.extend(items[:valid_count])
        train.extend(items[valid_count:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_counts(examples: list[dict]) -> dict:
    counts = Counter()
    for example in examples:
        output = json.loads(example["messages"][-1]["content"])
        counts[output["intent_class"]] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified classifier dataset for MLX LoRA.")
    parser.add_argument("--training-log", type=Path, default=DEFAULT_TRAINING_LOG)
    parser.add_argument("--existing-dir", type=Path, default=DEFAULT_EXISTING_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed-files", type=Path, nargs="*", default=list(DEFAULT_SEED_FILES))
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    candidates = [
        *load_existing_examples(args.existing_dir),
        *load_seed_examples(tuple(args.seed_files)),
        *load_training_log_examples(args.training_log),
    ]
    unified, summary = unify_examples(candidates)
    formatted = [build_classifier_example(example) for example in unified]
    train, valid = stratified_split(formatted, valid_ratio=args.valid_ratio, seed=args.seed)

    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "valid.jsonl", valid)
    (args.out_dir / "summary.json").write_text(
        json.dumps({
            **summary,
            "unified_examples": len(formatted),
            "train_examples": len(train),
            "valid_examples": len(valid),
            "label_counts": label_counts(formatted),
            "train_label_counts": label_counts(train),
            "valid_label_counts": label_counts(valid),
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print((args.out_dir / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
