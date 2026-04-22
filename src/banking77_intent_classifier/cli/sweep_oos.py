"""CLI entrypoint for sweeping OOS thresholds over a training config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from banking77_intent_classifier.config import load_config
from banking77_intent_classifier.pipeline import run_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Cartesian sweep over OOS max-probability and margin thresholds."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON experiment configuration file.",
    )
    parser.add_argument(
        "--oos-thresholds",
        required=True,
        help="Comma-separated max-probability thresholds, e.g. 0.2,0.3,0.4",
    )
    parser.add_argument(
        "--oos-margin-thresholds",
        required=True,
        help="Comma-separated margin thresholds, e.g. 0.05,0.1,0.15",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    probability_thresholds = _parse_thresholds(args.oos_thresholds)
    margin_thresholds = _parse_thresholds(args.oos_margin_thresholds)

    base_config = load_config(args.config)
    sweep_results: list[dict] = []

    for probability_threshold in probability_thresholds:
        for margin_threshold in margin_thresholds:
            suffix = _build_suffix(probability_threshold, margin_threshold)
            summary = run_training_pipeline(
                args.config,
                oos_threshold_override=probability_threshold,
                oos_margin_threshold_override=margin_threshold,
                output_suffix=suffix,
            )
            sweep_results.append(summary)

    results_df = pd.DataFrame(
        [
            {
                "oos_confidence_threshold": result["oos_confidence_threshold"],
                "oos_margin_threshold": result["oos_margin_threshold"],
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "top_5_accuracy": result["top_5_accuracy"],
                "oos_precision": result["oos_metrics"].get("precision"),
                "oos_recall": result["oos_metrics"].get("recall"),
                "oos_f1": result["oos_metrics"].get("f1"),
                "oos_miss_rate": result["oos_metrics"].get("miss_rate"),
                "in_scope_false_oos_rate": result["oos_metrics"].get("in_scope_false_oos_rate"),
                "reports_dir": result["reports_dir"],
            }
            for result in sweep_results
        ]
    ).sort_values(by=["oos_recall", "macro_f1"], ascending=[False, False])

    sweep_dir = base_config.reports_dir.parent / f"{base_config.reports_dir.name}_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(sweep_dir / "oos_threshold_sweep.csv", index=False)
    with (sweep_dir / "oos_threshold_sweep.json").open("w", encoding="utf-8") as file_handle:
        json.dump(sweep_results, file_handle, indent=2, sort_keys=True)
        file_handle.write("\n")

    print(results_df.to_json(orient="records", indent=2))


def _parse_thresholds(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _build_suffix(probability_threshold: float, margin_threshold: float) -> str:
    probability_part = str(probability_threshold).replace(".", "")
    margin_part = str(margin_threshold).replace(".", "")
    return f"prob_{probability_part}_margin_{margin_part}"


if __name__ == "__main__":
    main()
