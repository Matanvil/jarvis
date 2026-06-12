from pathlib import Path

import pytest

from finetune_data.train_classifier import build_command, validate_inputs


def test_build_command_uses_production_training_defaults():
    command = build_command(
        "/opt/homebrew/bin/mlx_lm.lora",
        "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        Path("/data"),
        Path("/adapters"),
        smoke=False,
    )
    assert command[0] == "/opt/homebrew/bin/mlx_lm.lora"
    assert command[command.index("--iters") + 1] == "600"
    assert command[command.index("--batch-size") + 1] == "4"
    assert command[command.index("--num-layers") + 1] == "8"
    assert "--mask-prompt" in command


def test_build_command_smoke_reduces_work():
    command = build_command("mlx_lm.lora", "model", Path("/data"), Path("/adapters"), smoke=True)
    assert command[command.index("--iters") + 1] == "1"
    assert command[command.index("--batch-size") + 1] == "1"
    assert command[command.index("--num-layers") + 1] == "1"


def test_validate_inputs_requires_train_and_valid(tmp_path):
    (tmp_path / "train.jsonl").write_text("{}\n")
    with pytest.raises(FileNotFoundError, match="valid.jsonl"):
        validate_inputs(tmp_path)

    (tmp_path / "valid.jsonl").write_text("{}\n")
    validate_inputs(tmp_path)
