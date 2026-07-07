"""Push the validated tasks.jsonl dataset to the HF Hub dataset repo. Human runs this."""

# No automated test: this shells out to the `hf` CLI and performs a real network push to the
# Hugging Face Hub, requiring HF_TOKEN auth. It cannot be exercised in CI or offline tests.
# Run manually: `uv run python scripts/upload_dataset.py` (after scripts/validate_tasks.py).

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DATASET_REPO = "rohanjain2312/toolsmith-tasks"
TASKS_PATH = Path("results/tasks.jsonl")
DATASET_CARD_PATH = Path("results/dataset_card.md")


class DatasetUploadError(RuntimeError):
    """Raised when the `hf` CLI upload command fails."""


def _run_hf_upload(local_path: Path, repo_path: str) -> None:
    """Upload one file to the dataset repo via the `hf` CLI."""
    result = subprocess.run(
        ["hf", "upload", DATASET_REPO, str(local_path), repo_path, "--repo-type", "dataset"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise DatasetUploadError(f"hf upload failed for {local_path}: {result.stderr}")


def upload_dataset() -> None:
    """Push tasks.jsonl and the dataset card (if rendered) to the HF Hub dataset repo."""
    if not TASKS_PATH.exists():
        raise DatasetUploadError(f"{TASKS_PATH} not found — run scripts/validate_tasks.py first")
    _run_hf_upload(TASKS_PATH, "tasks.jsonl")
    if DATASET_CARD_PATH.exists():
        _run_hf_upload(DATASET_CARD_PATH, "README.md")


def main() -> int:
    try:
        upload_dataset()
    except DatasetUploadError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"uploaded dataset to https://huggingface.co/datasets/{DATASET_REPO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
