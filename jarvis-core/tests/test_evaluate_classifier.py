import json

from finetune_data.evaluate_classifier import load_cases, parse_classifier_output


def test_parse_classifier_output_extracts_wrapped_json():
    output = parse_classifier_output(
        'thinking first\n{"can_handle_locally": true, "intent_class": "read_only", "reason": "local"}'
    )
    assert output["intent_class"] == "read_only"


def test_parse_classifier_output_rejects_missing_json():
    try:
        parse_classifier_output("read_only")
    except ValueError as exc:
        assert "JSON object" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_cases_uses_assistant_output_as_expected_label(tmp_path):
    path = tmp_path / "valid.jsonl"
    path.write_text(json.dumps({
        "messages": [
            {"role": "system", "content": "classify"},
            {"role": "user", "content": "show git status"},
            {
                "role": "assistant",
                "content": json.dumps({
                    "can_handle_locally": True,
                    "intent_class": "read_only",
                    "reason": "local",
                }),
            },
        ]
    }) + "\n")
    cases = load_cases(path)
    assert cases == [{
        "messages": [
            {"role": "system", "content": "classify"},
            {"role": "user", "content": "show git status"},
        ],
        "command": "show git status",
        "expected": "read_only",
    }]
