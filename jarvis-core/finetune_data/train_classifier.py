import argparse
import shutil
import subprocess
from pathlib import Path

DEFAULT_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
DEFAULT_DATA_DIR = Path.home() / ".jarvis" / "training" / "classifier_unified"
DEFAULT_ADAPTER_PATH = Path.home() / ".jarvis" / "training" / "classifier_adapters"


def build_command(
    mlx_lora: str,
    model: str,
    data_dir: Path,
    adapter_path: Path,
    *,
    smoke: bool,
) -> list[str]:
    command = [
        mlx_lora,
        "--model", model,
        "--train",
        "--data", str(data_dir),
        "--fine-tune-type", "lora",
        "--optimizer", "adamw",
        "--mask-prompt",
        "--iters", "1" if smoke else "600",
        "--batch-size", "1" if smoke else "4",
        "--num-layers", "1" if smoke else "8",
        "--learning-rate", "1e-5",
        "--steps-per-report", "1" if smoke else "10",
        "--steps-per-eval", "1" if smoke else "100",
        "--val-batches", "1" if smoke else "25",
        "--save-every", "1" if smoke else "100",
        "--max-seq-length", "1024",
        "--adapter-path", str(adapter_path),
        "--seed", "42",
    ]
    return command


def validate_inputs(data_dir: Path) -> None:
    missing = [
        path.name
        for path in (data_dir / "train.jsonl", data_dir / "valid.jsonl")
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"missing classifier dataset files in {data_dir}: {', '.join(missing)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Jarvis Qwen3-4B classifier LoRA.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--smoke", action="store_true", help="Run one iteration with one LoRA layer.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    args = parser.parse_args()

    validate_inputs(args.data_dir)
    mlx_lora = shutil.which("mlx_lm.lora")
    if not mlx_lora:
        raise FileNotFoundError("mlx_lm.lora is not installed or not on PATH")

    command = build_command(
        mlx_lora,
        args.model,
        args.data_dir,
        args.adapter_path,
        smoke=args.smoke,
    )
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
