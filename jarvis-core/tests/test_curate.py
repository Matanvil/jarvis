import json

from finetune_data.curate import (
    build_executor_example,
    is_successful_executor_record,
    parse_step_arguments,
    split_examples,
    synthesize_counter_examples,
)


def test_is_successful_executor_record_requires_model_steps_and_response():
    record = {
        "request": "read README",
        "intent_class": "read_only",
        "model": "qwen3.6:35b-a3b",
        "steps": [{"tool": "file_read", "input_summary": "{'path': 'README.md'}", "result_summary": "docs"}],
        "display": "Here is the README.",
        "speak": "Here is the README.",
    }
    assert is_successful_executor_record(record) is True

    assert is_successful_executor_record({**record, "model": None}) is False
    assert is_successful_executor_record({**record, "steps": []}) is False
    assert is_successful_executor_record({**record, "display": None, "speak": None}) is False


def test_is_successful_executor_record_filters_known_failures():
    record = {
        "request": "do something",
        "intent_class": "prepare",
        "model": "qwen3.6:35b-a3b",
        "steps": [{"tool": "shell_run", "input_summary": "{'command': 'echo hi'}", "result_summary": "hi"}],
        "display": "I ran out of steps while trying to finish this.",
        "speak": "I ran out of steps while trying to finish this.",
    }
    assert is_successful_executor_record(record) is False


def test_parse_step_arguments_uses_literal_eval_for_dict_strings():
    args = parse_step_arguments({"input_summary": "{'query': 'current weather Netanya Israel'}"})
    assert args == {"query": "current weather Netanya Israel"}


def test_build_executor_example_creates_tool_call_messages():
    record = {
        "request": "read README",
        "intent_class": "read_only",
        "model": "claude-haiku-4-5-20251001",
        "cwd": None,
        "duration_ms": 1234,
        "steps": [
            {
                "tool": "file_read",
                "input_summary": "{'path': 'README.md'}",
                "result_summary": "# Project",
            }
        ],
        "display": "The README starts with Project.",
        "speak": "The README starts with Project.",
    }
    example = build_executor_example(record)
    assert example["messages"][0] == {"role": "user", "content": "read README"}
    assert example["messages"][1]["tool_calls"][0]["function"]["name"] == "file_read"
    assert json.loads(example["messages"][1]["tool_calls"][0]["function"]["arguments"]) == {"path": "README.md"}
    assert example["messages"][-1]["content"] == "The README starts with Project."


def test_split_examples_keeps_non_empty_train_when_possible():
    items = [{"messages": [{"role": "user", "content": str(i)}]} for i in range(3)]
    train, valid = split_examples(items, valid_ratio=0.5, seed=42)
    assert len(train) == 2
    assert len(valid) == 1


def test_synthesize_counter_examples_covers_known_failure_modes():
    examples = synthesize_counter_examples()
    failure_modes = {example["meta"]["failure_mode"] for example in examples}
    assert failure_modes == {
        "grep_as_text",
        "read_file_blindspot",
        "answers_from_memory",
    }
