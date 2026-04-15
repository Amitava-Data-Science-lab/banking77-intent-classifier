"""Upload the Banking77 champion model files to the Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload Banking77 champion artifacts to a Hugging Face model repository."
    )
    parser.add_argument("--repo-id", required=True, help="Destination repo id, e.g. username/model-name.")
    parser.add_argument(
        "--model-dir",
        default="artifacts/champion",
        help="Directory containing champion model artifacts.",
    )
    parser.add_argument(
        "--config-path",
        default="configs/champion.json",
        help="Path to the frozen champion config.",
    )
    parser.add_argument(
        "--model-card",
        default="hub/README.md",
        help="Path to the Hugging Face model card markdown file.",
    )
    parser.add_argument(
        "--requirements-path",
        default="requirements.txt",
        help="Path to requirements.txt to upload with the model.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private if it does not already exist.",
    )
    return parser


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required path not found: {path}. "
            "Train the champion first or point this script at an existing artifact directory."
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise ImportError(
            "huggingface_hub is required. Install it with `pip install huggingface_hub`."
        ) from error

    model_dir = Path(args.model_dir)
    config_path = Path(args.config_path)
    model_card = Path(args.model_card)
    requirements_path = Path(args.requirements_path)

    ensure_exists(model_dir)
    ensure_exists(config_path)
    ensure_exists(model_card)
    ensure_exists(requirements_path)

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(model_dir))
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=str(config_path),
        path_in_repo="champion.json",
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=str(requirements_path),
        path_in_repo="requirements.txt",
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=str(model_card),
        path_in_repo="README.md",
    )


if __name__ == "__main__":
    main()
