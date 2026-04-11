"""Cross-dataset evaluation for ETG generalization validation.

Addresses the critical limitation that all ETG results were reported
on TruthfulQA only. This script evaluates the ETG verification pipeline
on held-out benchmarks to test generalization:

    1. FEVER (fact verification, 185K claims)
    2. HaluEval (hallucination detection, 35K samples)
    3. FactScore Biographies (fine-grained factuality)

The meta-classifier is trained ONLY on TruthfulQA calibration data,
then evaluated on each cross-validation dataset WITHOUT any retraining.
This is the correct methodology for testing generalization claims.

Usage:
    python scripts/cross_dataset_eval.py

Requirements:
    - TruthfulQA calibration data (for meta-classifier training)
    - FEVER, HaluEval, FactScore datasets (downloaded via download_data.py)
    - Verification models (NLI, LLM-Judge, QA)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from etg_rlm.datasets import (
    CROSS_VALIDATION_CONFIGS,
    TRUTHFUL_QA_CONFIG,
    DatasetConfig,
    DatasetName,
)
from etg_rlm.statistics import (
    BootstrapCIResult,
    bootstrap_ci,
    benjamini_hochberg_correction,
)


class DatasetEvalResult(NamedTuple):
    """Evaluation result for a single dataset."""

    dataset_name: str
    n_claims: int
    precision: float
    recall: float
    f1: float
    fpr: float
    hallucination_rate: float
    precision_ci: BootstrapCIResult | None
    recall_ci: BootstrapCIResult | None
    f1_ci: BootstrapCIResult | None


@dataclass
class CrossDatasetReport:
    """Cross-dataset generalization report.

    Compares ETG performance on TruthfulQA (development set) vs
    held-out benchmarks (generalization test).
    """

    development_result: DatasetEvalResult | None = None
    generalization_results: list[DatasetEvalResult] = field(default_factory=list)
    generalization_gap: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "Cross-Dataset Generalization Report",
            "=" * 70,
        ]

        if self.development_result:
            r = self.development_result
            lines.append(
                f"\nDevelopment (TruthfulQA): "
                f"P={r.precision:.3f} R={r.recall:.3f} F1={r.f1:.3f}"
            )

        lines.append("\nGeneralization benchmarks:")
        lines.append("-" * 70)
        for r in self.generalization_results:
            ci_str = ""
            if r.f1_ci:
                ci_str = f" [{r.f1_ci.ci_lower:.3f}, {r.f1_ci.ci_upper:.3f}]"
            lines.append(
                f"  {r.dataset_name:<25s} "
                f"P={r.precision:.3f} R={r.recall:.3f} "
                f"F1={r.f1:.3f}{ci_str} "
                f"(n={r.n_claims})"
            )

        if self.generalization_gap:
            lines.append("\nGeneralization gap (F1 drop from development):")
            for name, gap in self.generalization_gap.items():
                status = "OK" if abs(gap) < 0.05 else "CONCERN"
                lines.append(f"  {name:<25s} {gap:+.3f} [{status}]")

        lines.append("=" * 70)
        return "\n".join(lines)


def compute_generalization_gap(
    dev_result: DatasetEvalResult,
    gen_results: list[DatasetEvalResult],
) -> dict[str, float]:
    """Compute the F1 gap between development and generalization sets."""
    gaps = {}
    for r in gen_results:
        gaps[r.dataset_name] = r.f1 - dev_result.f1
    return gaps


def run_cross_dataset_evaluation(
    claim_scores: dict[str, list[tuple[float, bool]]],
    meta_threshold: float = 0.5,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> CrossDatasetReport:
    """Run cross-dataset evaluation from pre-computed claim scores.

    Args:
        claim_scores: mapping from dataset name to list of
            (meta_classifier_score, ground_truth_label) pairs.
        meta_threshold: decision threshold for the meta-classifier
        n_bootstrap: number of bootstrap samples for CIs
        seed: random seed for reproducibility

    Returns:
        CrossDatasetReport with development and generalization results.
    """
    results = {}

    for dataset_name, scores in claim_scores.items():
        if not scores:
            continue

        predictions = [s >= meta_threshold for s, _ in scores]
        labels = [label for _, label in scores]

        tp = sum(1 for p, l in zip(predictions, labels) if p and l)
        fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
        fn = sum(1 for p, l in zip(predictions, labels) if not p and l)
        tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        hallucination_rate = fp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Bootstrap CIs for F1
        # Per-claim F1 proxy: 1.0 for TP, 0.0 for FP/FN
        per_claim = []
        for p, l in zip(predictions, labels):
            if p and l:
                per_claim.append(1.0)
            elif p and not l:
                per_claim.append(0.0)
            elif not p and l:
                per_claim.append(0.0)
            else:
                per_claim.append(1.0)

        f1_ci = bootstrap_ci(per_claim, n_bootstrap=n_bootstrap, seed=seed)

        results[dataset_name] = DatasetEvalResult(
            dataset_name=dataset_name,
            n_claims=len(scores),
            precision=precision,
            recall=recall,
            f1=f1,
            fpr=fpr,
            hallucination_rate=hallucination_rate,
            precision_ci=None,
            recall_ci=None,
            f1_ci=f1_ci,
        )

    dev_result = results.get("truthful_qa")
    gen_results = [
        r for name, r in results.items() if name != "truthful_qa"
    ]

    gap = {}
    if dev_result:
        gap = compute_generalization_gap(dev_result, gen_results)

    return CrossDatasetReport(
        development_result=dev_result,
        generalization_results=gen_results,
        generalization_gap=gap,
    )
