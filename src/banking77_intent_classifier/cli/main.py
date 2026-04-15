"""CLI entrypoint for training Banking77 baseline variants."""

from __future__ import annotations

import argparse
import json

from banking77_intent_classifier.pipeline import run_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Train the Banking77 TF-IDF + linear SVM baseline and export diagnostics."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON experiment configuration file.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    summary = run_training_pipeline(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
