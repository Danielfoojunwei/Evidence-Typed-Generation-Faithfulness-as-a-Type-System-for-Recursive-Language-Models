"""Statistical analysis for ETG evaluation (Section 4 of experimental design).

Implements the statistical analysis plan:

Section 4.1: Hypothesis Testing
    - Paired t-test for hallucination rate reduction
    - Null H0: ETG does not significantly reduce hallucination vs. RAG
    - Alternative H1: ETG significantly reduces hallucination vs. RAG
    - Significance level alpha = 0.05

Section 4.2: Effect Size
    - Cohen's d for magnitude of improvement
    - d > 0.8 is considered large

Section 4.3: Confidence Intervals
    - 95% bootstrapped confidence intervals (10,000 samples)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Paired t-test (Section 4.1)
# ---------------------------------------------------------------------------


class PairedTTestResult(NamedTuple):
    """Result of a paired t-test.

    Attributes:
        t_statistic: the t-statistic
        p_value: two-tailed p-value
        n_pairs: number of paired observations
        mean_diff: mean of the differences
        std_diff: standard deviation of the differences
        significant: whether p < alpha
    """

    t_statistic: float
    p_value: float
    n_pairs: int
    mean_diff: float
    std_diff: float
    significant: bool


def _students_t_cdf(t: float, df: int) -> float:
    """Approximate the CDF of Student's t-distribution.

    Uses the regularized incomplete beta function approximation.
    For large df (> 100), falls back to normal approximation.
    """
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")

    # For large df, use normal approximation
    if df > 100:
        return _normal_cdf(t)

    # Use the relationship: P(T <= t) = 1 - 0.5 * I_x(df/2, 1/2)
    # where x = df / (df + t^2)
    x = df / (df + t * t)
    a = df / 2.0
    b = 0.5

    # Regularized incomplete beta via continued fraction
    ibeta = _regularized_beta(x, a, b)

    if t >= 0:
        return 1.0 - 0.5 * ibeta
    else:
        return 0.5 * ibeta


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz and Stegun)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _regularized_beta(x: float, a: float, b: float, max_iter: int = 200) -> float:
    """Regularized incomplete beta function I_x(a, b) via Lentz's continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Use the symmetry relation if needed for convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_beta(1.0 - x, b, a, max_iter)

    # Compute the log of the front factor
    log_front = (
        a * math.log(x)
        + b * math.log(1.0 - x)
        - math.log(a)
        - _log_beta(a, b)
    )
    front = math.exp(log_front)

    # Lentz's continued fraction
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, max_iter + 1):
        # Even step
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d

        # Odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d

        if abs(c * d - 1.0) < 1e-10:
            break

    return front * f


def _log_beta(a: float, b: float) -> float:
    """Log of the beta function: log B(a, b) = log Gamma(a) + log Gamma(b) - log Gamma(a+b)."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def paired_t_test(
    x: list[float],
    y: list[float],
    alpha: float = 0.05,
) -> PairedTTestResult:
    """Paired t-test for comparing two systems on the same instances.

    Tests whether the mean difference between paired observations
    is significantly different from zero.

    H0: mean(x - y) = 0
    H1: mean(x - y) != 0

    Args:
        x: per-instance metrics for system 1 (e.g., ETG hallucination rates)
        y: per-instance metrics for system 2 (e.g., RAG hallucination rates)
        alpha: significance level (default 0.05)

    Returns:
        PairedTTestResult with t-statistic, p-value, and significance.

    Raises:
        ValueError: if inputs have different lengths or fewer than 2 pairs.
    """
    if len(x) != len(y):
        raise ValueError(f"Inputs must have same length: {len(x)} != {len(y)}")
    n = len(x)
    if n < 2:
        raise ValueError(f"Need at least 2 paired observations, got {n}")

    diffs = [xi - yi for xi, yi in zip(x, y)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    std_d = math.sqrt(var_d) if var_d > 0 else 0.0

    if std_d == 0:
        # All differences are the same
        if mean_d == 0:
            return PairedTTestResult(
                t_statistic=0.0,
                p_value=1.0,
                n_pairs=n,
                mean_diff=mean_d,
                std_diff=0.0,
                significant=False,
            )
        else:
            return PairedTTestResult(
                t_statistic=float("inf") if mean_d > 0 else float("-inf"),
                p_value=0.0,
                n_pairs=n,
                mean_diff=mean_d,
                std_diff=0.0,
                significant=True,
            )

    se = std_d / math.sqrt(n)
    t_stat = mean_d / se
    df = n - 1

    # Two-tailed p-value
    p_value = 2.0 * (1.0 - _students_t_cdf(abs(t_stat), df))
    p_value = max(0.0, min(1.0, p_value))

    return PairedTTestResult(
        t_statistic=t_stat,
        p_value=p_value,
        n_pairs=n,
        mean_diff=mean_d,
        std_diff=std_d,
        significant=p_value < alpha,
    )


# ---------------------------------------------------------------------------
# Cohen's d (Section 4.2)
# ---------------------------------------------------------------------------


class EffectSizeResult(NamedTuple):
    """Result of effect size computation.

    Attributes:
        cohens_d: Cohen's d effect size
        interpretation: qualitative interpretation
        mean_diff: mean difference between groups
        pooled_std: pooled standard deviation
    """

    cohens_d: float
    interpretation: str
    mean_diff: float
    pooled_std: float


def cohens_d(
    x: list[float],
    y: list[float],
) -> EffectSizeResult:
    """Compute Cohen's d effect size for paired observations.

    Cohen's d quantifies the magnitude of the difference between
    two groups in terms of standard deviation units.

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large

    Args:
        x: per-instance metrics for system 1
        y: per-instance metrics for system 2

    Returns:
        EffectSizeResult with Cohen's d and interpretation.
    """
    if len(x) != len(y):
        raise ValueError(f"Inputs must have same length: {len(x)} != {len(y)}")
    n = len(x)
    if n < 2:
        raise ValueError(f"Need at least 2 observations, got {n}")

    mean_x = sum(x) / n
    mean_y = sum(y) / n
    mean_diff = mean_x - mean_y

    var_x = sum((xi - mean_x) ** 2 for xi in x) / (n - 1)
    var_y = sum((yi - mean_y) ** 2 for yi in y) / (n - 1)

    pooled_var = (var_x + var_y) / 2.0
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 0.0

    if pooled_std == 0:
        d = 0.0 if mean_diff == 0 else float("inf")
    else:
        d = mean_diff / pooled_std

    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return EffectSizeResult(
        cohens_d=d,
        interpretation=interpretation,
        mean_diff=mean_diff,
        pooled_std=pooled_std,
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (Section 4.3)
# ---------------------------------------------------------------------------


class BootstrapCIResult(NamedTuple):
    """Result of bootstrap confidence interval estimation.

    Attributes:
        point_estimate: the statistic computed on the full sample
        ci_lower: lower bound of the confidence interval
        ci_upper: upper bound of the confidence interval
        confidence_level: the confidence level (e.g. 0.95)
        n_bootstrap: number of bootstrap samples used
    """

    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int


def bootstrap_ci(
    data: list[float],
    statistic: str = "mean",
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int | None = None,
) -> BootstrapCIResult:
    """Compute bootstrap confidence interval for a statistic.

    Uses the percentile method with the specified number of
    bootstrap resamples.

    Args:
        data: the observed data
        statistic: which statistic to compute ("mean" or "median")
        confidence: confidence level (default 0.95 for 95% CI)
        n_bootstrap: number of bootstrap samples (default 10,000)
        seed: random seed for reproducibility

    Returns:
        BootstrapCIResult with point estimate and CI bounds.

    Raises:
        ValueError: if data is empty or confidence is out of range.
    """
    if not data:
        raise ValueError("Data must not be empty")
    if not 0 < confidence < 1:
        raise ValueError(f"Confidence must be in (0, 1), got {confidence}")

    rng = random.Random(seed)
    n = len(data)

    def compute_stat(sample: list[float]) -> float:
        if statistic == "mean":
            return sum(sample) / len(sample)
        elif statistic == "median":
            s = sorted(sample)
            m = len(s)
            if m % 2 == 1:
                return float(s[m // 2])
            return (s[m // 2 - 1] + s[m // 2]) / 2.0
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

    point_estimate = compute_stat(data)

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(data) for _ in range(n)]
        bootstrap_stats.append(compute_stat(sample))

    bootstrap_stats.sort()

    # Percentile method
    alpha = 1.0 - confidence
    lower_idx = int(math.floor((alpha / 2.0) * n_bootstrap))
    upper_idx = int(math.ceil((1.0 - alpha / 2.0) * n_bootstrap)) - 1

    lower_idx = max(0, min(lower_idx, n_bootstrap - 1))
    upper_idx = max(0, min(upper_idx, n_bootstrap - 1))

    return BootstrapCIResult(
        point_estimate=point_estimate,
        ci_lower=bootstrap_stats[lower_idx],
        ci_upper=bootstrap_stats[upper_idx],
        confidence_level=confidence,
        n_bootstrap=n_bootstrap,
    )


def bootstrap_paired_ci(
    x: list[float],
    y: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int | None = None,
) -> BootstrapCIResult:
    """Bootstrap CI for the mean difference between paired observations.

    Resamples the paired differences and computes the CI of the
    mean difference.

    Args:
        x: per-instance metrics for system 1
        y: per-instance metrics for system 2
        confidence: confidence level
        n_bootstrap: number of bootstrap samples
        seed: random seed

    Returns:
        BootstrapCIResult for the mean difference.
    """
    if len(x) != len(y):
        raise ValueError(f"Inputs must have same length: {len(x)} != {len(y)}")
    diffs = [xi - yi for xi, yi in zip(x, y)]
    return bootstrap_ci(diffs, statistic="mean", confidence=confidence, n_bootstrap=n_bootstrap, seed=seed)


# ---------------------------------------------------------------------------
# Full statistical analysis report
# ---------------------------------------------------------------------------


@dataclass
class StatisticalAnalysis:
    """Complete statistical analysis of ETG vs. a baseline.

    Attributes:
        metric_name: which metric was compared (e.g. "hallucination_rate")
        t_test: paired t-test result
        effect_size: Cohen's d result
        etg_ci: bootstrap CI for ETG metric
        baseline_ci: bootstrap CI for baseline metric
        diff_ci: bootstrap CI for the paired difference
    """

    metric_name: str
    t_test: PairedTTestResult
    effect_size: EffectSizeResult
    etg_ci: BootstrapCIResult
    baseline_ci: BootstrapCIResult
    diff_ci: BootstrapCIResult


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR correction (addresses multiple comparisons issue)
# ---------------------------------------------------------------------------


class CorrectedPValue(NamedTuple):
    """A p-value with its BH-corrected (adjusted) counterpart.

    Attributes:
        original_index: index of this test in the original list
        raw_p: the raw (uncorrected) p-value
        adjusted_p: the BH-adjusted p-value
        significant: whether the adjusted p-value < alpha
        label: optional label identifying this comparison
    """

    original_index: int
    raw_p: float
    adjusted_p: float
    significant: bool
    label: str


def benjamini_hochberg_correction(
    p_values: list[float],
    alpha: float = 0.05,
    labels: list[str] | None = None,
) -> list[CorrectedPValue]:
    """Apply Benjamini-Hochberg FDR correction for multiple comparisons.

    When testing multiple aggregation methods or threshold configurations
    simultaneously (e.g., 7 aggregation methods x multiple thresholds),
    the probability of finding at least one spuriously significant result
    increases. BH correction controls the False Discovery Rate (FDR).

    The procedure:
        1. Sort p-values in ascending order: p_(1) <= p_(2) <= ... <= p_(m)
        2. Find the largest k such that p_(k) <= k/m * alpha
        3. Reject hypotheses 1, ..., k

    Adjusted p-values are computed as:
        p_adj(i) = min(p_(i) * m / i, 1.0)
    with monotonicity enforcement (adjusted p-values are non-decreasing).

    Args:
        p_values: list of raw p-values from multiple hypothesis tests
        alpha: target FDR level (default 0.05)
        labels: optional labels for each test (e.g., method names)

    Returns:
        List of CorrectedPValue results, in the original order.
    """
    m = len(p_values)
    if m == 0:
        return []

    if labels is None:
        labels = [f"test_{i}" for i in range(m)]
    if len(labels) != m:
        raise ValueError(
            f"labels length ({len(labels)}) must match p_values length ({m})"
        )

    # Create (original_index, p_value) pairs and sort by p-value
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # Compute adjusted p-values with monotonicity enforcement
    adjusted = [0.0] * m
    prev_adj = 0.0
    for rank, (orig_idx, raw_p) in enumerate(indexed, start=1):
        adj_p = min(raw_p * m / rank, 1.0)
        # Enforce monotonicity: adjusted p-values must be non-decreasing
        # when sorted by raw p-value
        adj_p = max(adj_p, prev_adj)
        adjusted[orig_idx] = adj_p
        prev_adj = adj_p

    results = []
    for i in range(m):
        results.append(CorrectedPValue(
            original_index=i,
            raw_p=p_values[i],
            adjusted_p=adjusted[i],
            significant=adjusted[i] < alpha,
            label=labels[i],
        ))

    return results


def full_analysis(
    etg_values: list[float],
    baseline_values: list[float],
    metric_name: str = "hallucination_rate",
    alpha: float = 0.05,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int | None = None,
) -> StatisticalAnalysis:
    """Run the complete statistical analysis pipeline.

    Performs paired t-test, computes Cohen's d, and bootstraps
    confidence intervals for both systems and their difference.

    Args:
        etg_values: per-instance metric values for ETG
        baseline_values: per-instance metric values for the baseline
        metric_name: name of the metric being compared
        alpha: significance level for the t-test
        confidence: confidence level for CIs
        n_bootstrap: number of bootstrap samples
        seed: random seed for reproducibility

    Returns:
        StatisticalAnalysis with all results.
    """
    t_result = paired_t_test(etg_values, baseline_values, alpha=alpha)
    d_result = cohens_d(etg_values, baseline_values)

    etg_ci = bootstrap_ci(etg_values, confidence=confidence, n_bootstrap=n_bootstrap, seed=seed)
    baseline_ci = bootstrap_ci(baseline_values, confidence=confidence, n_bootstrap=n_bootstrap, seed=seed)
    diff_ci = bootstrap_paired_ci(etg_values, baseline_values, confidence=confidence, n_bootstrap=n_bootstrap, seed=seed)

    return StatisticalAnalysis(
        metric_name=metric_name,
        t_test=t_result,
        effect_size=d_result,
        etg_ci=etg_ci,
        baseline_ci=baseline_ci,
        diff_ci=diff_ci,
    )
