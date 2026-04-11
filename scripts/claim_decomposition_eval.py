"""Claim decomposition quality evaluation.

Evaluates the quality of atomic claim extraction A(y), which is a
critical and unvalidated assumption in the ETG pipeline. Errors in
claim decomposition propagate through the entire verification pipeline
(garbage in, garbage out).

Metrics:
    1. Completeness: fraction of reference claims covered by extracted claims
    2. Atomicity: fraction of extracted claims that are truly atomic
       (single factual assertion)
    3. Faithfulness: fraction of extracted claims that are faithful to
       the original text (no introduced artifacts)
    4. Over-splitting rate: fraction of atomic facts split into multiple claims
    5. Under-splitting rate: fraction of compound claims not decomposed

This script provides the evaluation framework. Actual annotation data
should be provided by human annotators on a sample of 200+ examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple


class DecompositionMetrics(NamedTuple):
    """Quality metrics for a single claim decomposition."""

    completeness: float  # reference claims covered / total reference claims
    atomicity: float  # atomic claims / total extracted claims
    faithfulness: float  # faithful claims / total extracted claims
    over_split_rate: float  # over-split instances / total
    under_split_rate: float  # compound claims / total extracted claims
    n_extracted: int
    n_reference: int


@dataclass
class DecompositionEvalReport:
    """Aggregated report over multiple decomposition evaluations."""

    n_instances: int = 0
    mean_completeness: float = 0.0
    mean_atomicity: float = 0.0
    mean_faithfulness: float = 0.0
    mean_over_split_rate: float = 0.0
    mean_under_split_rate: float = 0.0
    mean_extracted_per_instance: float = 0.0
    mean_reference_per_instance: float = 0.0
    per_instance: list[DecompositionMetrics] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Claim Decomposition Quality Report (n={self.n_instances})\n"
            f"  Completeness:    {self.mean_completeness:.3f}\n"
            f"  Atomicity:       {self.mean_atomicity:.3f}\n"
            f"  Faithfulness:    {self.mean_faithfulness:.3f}\n"
            f"  Over-split rate: {self.mean_over_split_rate:.3f}\n"
            f"  Under-split rate:{self.mean_under_split_rate:.3f}\n"
            f"  Avg extracted:   {self.mean_extracted_per_instance:.1f}\n"
            f"  Avg reference:   {self.mean_reference_per_instance:.1f}"
        )


def evaluate_decomposition(
    extracted_claims: list[str],
    reference_claims: list[str],
    atomic_flags: list[bool] | None = None,
    faithful_flags: list[bool] | None = None,
    over_split_flags: list[bool] | None = None,
) -> DecompositionMetrics:
    """Evaluate a single claim decomposition against reference.

    Args:
        extracted_claims: claims produced by the decomposition method
        reference_claims: gold-standard atomic claims (human-annotated)
        atomic_flags: per-extracted-claim, True if truly atomic
            (single factual assertion). If None, all assumed atomic.
        faithful_flags: per-extracted-claim, True if faithful to source.
            If None, all assumed faithful.
        over_split_flags: per-reference-claim, True if the reference
            claim was split into multiple extracted claims unnecessarily.

    Returns:
        DecompositionMetrics for this instance.
    """
    n_ext = len(extracted_claims)
    n_ref = len(reference_claims)

    if n_ext == 0:
        return DecompositionMetrics(
            completeness=0.0,
            atomicity=1.0,
            faithfulness=1.0,
            over_split_rate=0.0,
            under_split_rate=0.0,
            n_extracted=0,
            n_reference=n_ref,
        )

    # Completeness: simple text overlap heuristic
    # (In practice, use semantic similarity or human annotation)
    ref_lower = {r.lower().strip() for r in reference_claims}
    ext_lower = {e.lower().strip() for e in extracted_claims}
    covered = sum(1 for r in ref_lower if any(r in e or e in r for e in ext_lower))
    completeness = covered / n_ref if n_ref > 0 else 1.0

    # Atomicity
    if atomic_flags is not None:
        atomicity = sum(atomic_flags) / n_ext
    else:
        atomicity = 1.0

    # Faithfulness
    if faithful_flags is not None:
        faithfulness = sum(faithful_flags) / n_ext
    else:
        faithfulness = 1.0

    # Over-splitting
    if over_split_flags is not None:
        over_split_rate = sum(over_split_flags) / n_ref if n_ref > 0 else 0.0
    else:
        over_split_rate = 0.0

    # Under-splitting: claims that contain multiple facts
    # (inverse of atomicity for extracted claims)
    under_split_rate = 1.0 - atomicity

    return DecompositionMetrics(
        completeness=completeness,
        atomicity=atomicity,
        faithfulness=faithfulness,
        over_split_rate=over_split_rate,
        under_split_rate=under_split_rate,
        n_extracted=n_ext,
        n_reference=n_ref,
    )


def aggregate_decomposition_metrics(
    metrics_list: list[DecompositionMetrics],
) -> DecompositionEvalReport:
    """Aggregate decomposition metrics across multiple instances."""
    n = len(metrics_list)
    if n == 0:
        return DecompositionEvalReport()

    return DecompositionEvalReport(
        n_instances=n,
        mean_completeness=sum(m.completeness for m in metrics_list) / n,
        mean_atomicity=sum(m.atomicity for m in metrics_list) / n,
        mean_faithfulness=sum(m.faithfulness for m in metrics_list) / n,
        mean_over_split_rate=sum(m.over_split_rate for m in metrics_list) / n,
        mean_under_split_rate=sum(m.under_split_rate for m in metrics_list) / n,
        mean_extracted_per_instance=sum(m.n_extracted for m in metrics_list) / n,
        mean_reference_per_instance=sum(m.n_reference for m in metrics_list) / n,
        per_instance=metrics_list,
    )


def propagation_sensitivity_analysis(
    decomposition_error_rate: float,
    n_claims: int,
    verifier_precision: float,
) -> dict[str, float]:
    """Analyze how decomposition errors propagate through the ETG pipeline.

    Estimates the fraction of final output that is corrupted by
    decomposition errors (missed claims, spurious claims, merged claims).

    Args:
        decomposition_error_rate: fraction of claims with decomposition
            errors (1 - faithfulness * atomicity * completeness)
        n_claims: total number of claims in the pipeline
        verifier_precision: precision of the verification ensemble

    Returns:
        Dictionary with propagation analysis results.
    """
    # Claims with decomposition errors that pass verification
    error_pass_rate = decomposition_error_rate * verifier_precision
    n_corrupted = n_claims * error_pass_rate

    # Effective precision after accounting for decomposition errors
    effective_precision = verifier_precision * (1.0 - decomposition_error_rate)

    return {
        "decomposition_error_rate": decomposition_error_rate,
        "error_pass_rate": error_pass_rate,
        "n_corrupted_claims": n_corrupted,
        "effective_precision": effective_precision,
        "precision_loss_pct": (1.0 - effective_precision / verifier_precision) * 100
        if verifier_precision > 0
        else 0.0,
    }
