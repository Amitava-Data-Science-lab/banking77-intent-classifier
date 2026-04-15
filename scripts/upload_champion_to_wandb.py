"""Upload the champion model and reports to Weights & Biases Artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload Banking77 champion artifacts and reports to W&B Artifacts."
    )
    parser.add_argument("--project", required=True, help="W&B project name.")
    parser.add_argument("--entity", help="Optional W&B entity/team name.")
    parser.add_argument(
        "--model-dir",
        default="artifacts/champion",
        help="Directory containing champion model artifacts.",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports/champion",
        help="Directory containing champion evaluation reports.",
    )
    parser.add_argument(
        "--config-path",
        default="configs/champion.json",
        help="Path to the frozen champion config.",
    )
    parser.add_argument(
        "--run-name",
        default="banking77-champion-upload",
        help="Optional W&B run name.",
    )
    return parser


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


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
        import wandb
    except ImportError as error:
        raise ImportError("wandb is required. Install it with `pip install wandb`.") from error

    model_dir = Path(args.model_dir)
    reports_dir = Path(args.reports_dir)
    config_path = Path(args.config_path)

    ensure_exists(model_dir)
    ensure_exists(reports_dir)
    ensure_exists(config_path)

    metrics_path = reports_dir / "metrics_summary.json"
    model_metadata_path = model_dir / "model_metadata.json"
    ensure_exists(metrics_path)
    ensure_exists(model_metadata_path)

    metrics = read_json(metrics_path)
    metadata = read_json(model_metadata_path)

    with wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        job_type="model-registry",
        config={"config_path": str(config_path)},
    ) as run:
        model_artifact = wandb.Artifact(
            name="banking77-champion-model",
            type="model",
            metadata={**metadata, **metrics},
        )
        model_artifact.add_dir(str(model_dir))
        model_artifact.add_file(str(config_path), name="champion.json")
        run.log_artifact(model_artifact)

        eval_artifact = wandb.Artifact(
            name="banking77-champion-evaluation",
            type="evaluation",
            metadata=metrics,
        )
        eval_artifact.add_dir(str(reports_dir))
        run.log_artifact(eval_artifact)


if __name__ == "__main__":
    main()
