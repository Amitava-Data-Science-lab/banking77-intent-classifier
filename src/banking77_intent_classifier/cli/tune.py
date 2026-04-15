"""CLI entrypoint for randomized hyperparameter search experiments."""

from __future__ import annotations

import argparse
import json

from banking77_intent_classifier.tuning import run_search_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for search runs."""

    parser = argparse.ArgumentParser(
        description="Run Banking77 TF-IDF + LinearSVC randomized hyperparameter search and export diagnostics."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON tuning configuration file.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    summary = run_search_pipeline(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
