import json

from classifier_prompt import CLASSIFY_SYSTEM_PROMPT
from finetune_data.curate_classifier import (
    build_classifier_example,
    normalize_command,
    stratified_split,
    unify_examples,
)


def test_normalize_command_is_case_and_whitespace_insensitive():
    assert normalize_command("  Show   Git STATUS ") == "show git status"


def test_unify_examples_deduplicates_matching_labels():
    examples, summary = unify_examples([
        {"command": "show git status", "intent_class": "read_only", "reason": None, "source": "seed"},
        {"command": " Show  Git Status ", "intent_class": "read_only", "reason": "Local query.", "source": "log"},
    ])
    assert len(examples) == 1
    assert examples[0]["reason"] == "Local query."
    assert summary["duplicate_candidates"] == 1
    assert summary["conflicting_commands"] == 0


def test_unify_examples_skips_conflicting_labels():
    examples, summary = unify_examples([
        {"command": "delete build", "intent_class": "destructive", "reason": None, "source": "seed"},
        {"command": "delete build", "intent_class": "prepare", "reason": None, "source": "log"},
    ])
    assert examples == []
    assert summary["conflicting_commands"] == 1
    assert summary["conflicts"][0]["labels"] == ["destructive", "prepare"]


def test_build_classifier_example_matches_production_contract():
    example = build_classifier_example({
        "command": "show git status",
        "intent_class": "read_only",
        "reason": None,
        "source": "test",
    })
    assert example["messages"][0] == {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT}
    output = json.loads(example["messages"][-1]["content"])
    assert output["can_handle_locally"] is True
    assert output["intent_class"] == "read_only"
    assert set(output) == {"can_handle_locally", "intent_class", "reason"}


def test_build_classifier_example_marks_complex_reasoning_non_local():
    example = build_classifier_example({
        "command": "find today's stock price",
        "intent_class": "complex_reasoning",
        "reason": None,
        "source": "test",
    })
    output = json.loads(example["messages"][-1]["content"])
    assert output["can_handle_locally"] is False


def test_stratified_split_keeps_every_label_in_validation():
    examples = []
    for label in ("read_only", "prepare", "destructive", "complex_reasoning"):
        for index in range(10):
            examples.append(build_classifier_example({
                "command": f"{label} example {index}",
                "intent_class": label,
                "reason": None,
                "source": "test",
            }))

    train, valid = stratified_split(examples, valid_ratio=0.1, seed=42)
    valid_labels = {
        json.loads(example["messages"][-1]["content"])["intent_class"]
        for example in valid
    }
    assert len(train) == 36
    assert len(valid) == 4
    assert valid_labels == {"read_only", "prepare", "destructive", "complex_reasoning"}
