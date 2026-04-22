"""CLI entrypoint for training intent classification experiments."""

from __future__ import annotations

import argparse
import json

from banking77_intent_classifier.pipeline import run_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Train a dataset-configured intent classifier and export diagnostics."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON experiment configuration file.",
    )
    parser.add_argument(
        "--oos-threshold",
        type=float,
        default=None,
        help="Override the config OOS confidence threshold for sentence-transformer runs.",
    )
    parser.add_argument(
        "--oos-margin-threshold",
        type=float,
        default=None,
        help="Override the config OOS margin threshold for sentence-transformer runs.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    summary = run_training_pipeline(
        args.config,
        oos_threshold_override=args.oos_threshold,
        oos_margin_threshold_override=args.oos_margin_threshold,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
