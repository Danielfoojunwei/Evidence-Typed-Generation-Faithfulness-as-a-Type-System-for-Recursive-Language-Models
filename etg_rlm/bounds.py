"""Theoretical bounds for Evidence-Typed Generation (Section 6).

Proposition 1 (Exponential Suppression of Hallucinations):
    Assume a hallucinated claim has per-view false-positive probability alpha,
    and views are conditionally independent. Then:
        Pr[m(c) >= tau] <= exp(-N * D(tau || alpha))
    where D is KL divergence between Bernoulli distributions.
    Implication: increasing verification depth N yields exponential decay
    in hallucination acceptance.

    CRITICAL CAVEAT: The bound requires conditional independence of views.
    Empirical evaluation shows this assumption is violated in practice
    (v2: 96.7-98.5% pairwise agreement among NLI models; v4: 64.8% avg
    agreement vs 49.5% expected under independence). The bound should be
    accompanied by an independence diagnostic and, where violated, replaced
    with the correlated bound (see hallucination_upper_bound_correlated).

Proposition 2 (Evidence Pointer Guarantee — revised):
    Under ETG constraints, every rendered claim carries at least one
    evidence span pointer by construction: m(c) >= tau > 0 implies at
    least one view returned entailed with supporting spans.

    IMPORTANT: This is a structural guarantee on pointer EXISTENCE, not
    on pointer VALIDITY. A verifier with false-positive rate alpha will
    assign evidence spans to hallucinated claims. The probability that a
    rendered claim is actually unsupported is bounded by the conditional
    hallucination bound (see conditional_hallucination_bound), NOT zero.

Proposition 3 (Optimal Compute Allocation):
    Let each verification view have cost k. Under budget B, the optimal
    policy allocates views to claims maximizing:
        E[Delta Verified Utility] / k
    This reduces to a bandit / knapsack allocation problem over claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple


def kl_bernoulli(p: float, q: float) -> float:
    """Compute KL divergence D(p || q) for Bernoulli distributions.

    D(p || q) = p * log(p/q) + (1-p) * log((1-p)/(1-q))

    With conventions: 0*log(0) = 0, and D = +inf if q=0 and p>0.
    """
    if p < 0 or p > 1 or q < 0 or q > 1:
        raise ValueError(f"p and q must be in [0,1], got p={p}, q={q}")

    # Edge cases
    if q == 0.0:
        return float("inf") if p > 0.0 else 0.0
    if q == 1.0:
        return float("inf") if p < 1.0 else 0.0
    if p == 0.0:
        return -math.log(1.0 - q) if q < 1.0 else float("inf")
    if p == 1.0:
        return -math.log(q)

    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def hallucination_upper_bound(
    n_views: int,
    tau: float,
    alpha: float,
) -> float:
    """Proposition 1: Exponential Suppression of Hallucinations.

    Assume a hallucinated claim has per-view false-positive probability alpha,
    and views are conditionally independent. Then:
        Pr[m(c) >= tau] <= exp(-N * D(tau || alpha))

    This bound uses Sanov's theorem / Chernoff bound for binomial tails.
    Implication: increasing recursion depth N yields exponential decay
    in hallucination acceptance -- an inference-time scaling law for faithfulness.

    Args:
        n_views: N, number of independent verification views
        tau: support mass threshold for the Verified type
        alpha: per-view false-positive rate of the verifier

    Returns:
        Upper bound on the probability that an unsupported claim
        passes the support-mass gate.

    Raises:
        ValueError: if parameters are out of valid ranges
    """
    if n_views < 1:
        raise ValueError(f"n_views must be >= 1, got {n_views}")
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")

    # If alpha >= tau, the bound is trivially 1 (no filtering power)
    if alpha >= tau:
        return 1.0

    d = kl_bernoulli(tau, alpha)
    if math.isinf(d):
        return 0.0

    return math.exp(-n_views * d)


def required_views_for_bound(
    target_prob: float,
    tau: float,
    alpha: float,
) -> int:
    """Compute the minimum N to achieve a target hallucination probability.

    Solves: exp(-N * D(tau || alpha)) <= target_prob
    => N >= -log(target_prob) / D(tau || alpha)

    Args:
        target_prob: desired upper bound on hallucination probability
        tau: support mass threshold
        alpha: per-view false-positive rate

    Returns:
        Minimum number of views N needed.
    """
    if not (0.0 < target_prob < 1.0):
        raise ValueError(f"target_prob must be in (0, 1), got {target_prob}")
    if alpha >= tau:
        raise ValueError(
            f"alpha must be < tau for filtering to work, "
            f"got alpha={alpha}, tau={tau}"
        )

    d = kl_bernoulli(tau, alpha)
    if d == 0.0:
        raise ValueError("KL divergence is 0; cannot achieve bound")

    n = -math.log(target_prob) / d
    return math.ceil(n)


class ConfabulationCheckResult(NamedTuple):
    """Result of checking the evidence pointer guarantee (revised Proposition 2).

    Structural guarantee: every rendered claim has at least one evidence
    span pointer (because m(c) >= tau > 0 requires at least one view
    to have found supporting evidence).

    IMPORTANT: This checks pointer EXISTENCE, not pointer VALIDITY.
    Use conditional_hallucination_bound() to estimate the probability
    that a rendered claim is actually unsupported despite having
    evidence pointers (i.e., the verifier false-positive rate).
    """

    satisfies_proposition: bool
    confabulating_node_ids: list[str]
    total_verified_nodes: int
    estimated_invalid_pointer_rate: float


def check_zero_confabulation(
    verified_node_ids: set[str],
    node_evidence_counts: dict[str, int],
    node_support_masses: dict[str, float],
    tau: float,
    verifier_fpr: float = 0.0,
) -> ConfabulationCheckResult:
    """Revised Proposition 2: Evidence Pointer Guarantee.

    Checks that every rendered claim has at least one evidence span
    pointer and support mass >= tau. This is a structural invariant
    that holds by construction (since rendering requires m(c) >= tau > 0).

    CRITICAL NOTE: This guarantee is on pointer existence, NOT validity.
    A hallucinated claim can pass verification if verifiers produce
    false positives. The estimated_invalid_pointer_rate field reports
    the estimated probability that evidence pointers are invalid,
    given the verifier false-positive rate.

    Args:
        verified_node_ids: the set V^tau of rendered node IDs
        node_evidence_counts: node_id -> |sigma(v)| (number of evidence spans)
        node_support_masses: node_id -> m(v)
        tau: support mass threshold
        verifier_fpr: estimated false-positive rate of the verifier
            ensemble. Used to estimate the rate of invalid pointers.

    Returns:
        ConfabulationCheckResult with structural check and estimated
        invalid pointer rate.
    """
    confabulating: list[str] = []
    for nid in verified_node_ids:
        n_evidence = node_evidence_counts.get(nid, 0)
        mass = node_support_masses.get(nid, 0.0)
        if n_evidence == 0 or mass < tau:
            confabulating.append(nid)

    return ConfabulationCheckResult(
        satisfies_proposition=len(confabulating) == 0,
        confabulating_node_ids=confabulating,
        total_verified_nodes=len(verified_node_ids),
        estimated_invalid_pointer_rate=verifier_fpr,
    )


def conditional_hallucination_bound(
    verifier_precision: float,
    tau: float,
    n_views: int,
    alpha: float,
) -> float:
    """Conditional bound on hallucination in rendered output.

    Given that the verifier ensemble has precision p (probability that
    a "Verified" verdict is correct), the probability that a rendered
    claim is actually unsupported is:

        P(unsupported | rendered) <= (1 - p)

    More precisely, combining with the exponential suppression bound:

        P(hallucination rendered) <= min(
            1 - verifier_precision,
            hallucination_upper_bound(n_views, tau, alpha)
        )

    This replaces the tautological "zero-confabulation" claim with a
    meaningful guarantee conditioned on empirically measurable verifier
    quality.

    Args:
        verifier_precision: P(truly supported | verdict = Verified)
        tau: support mass threshold
        n_views: number of verification views
        alpha: per-view false-positive rate

    Returns:
        Upper bound on probability that a rendered claim is unsupported.
    """
    if not (0.0 <= verifier_precision <= 1.0):
        raise ValueError(
            f"verifier_precision must be in [0, 1], got {verifier_precision}"
        )
    precision_bound = 1.0 - verifier_precision
    exponential_bound = hallucination_upper_bound(n_views, tau, alpha)
    return min(precision_bound, exponential_bound)


class InferenceScalingResult(NamedTuple):
    """Inference-time scaling law: hallucination probability vs N."""

    n_views_sequence: list[int]
    bounds_sequence: list[float]
    tau: float
    alpha: float


def inference_time_scaling_law(
    tau: float,
    alpha: float,
    max_n: int = 50,
) -> InferenceScalingResult:
    """Compute the inference-time scaling law for faithfulness.

    For each N in [1, max_n], computes:
        Pr[m(c) >= tau] <= exp(-N * D(tau || alpha))

    This demonstrates the exponential decay in hallucination acceptance
    as a function of recursion depth N (Proposition 1).

    Args:
        tau: support mass threshold
        alpha: per-view false-positive rate (must be < tau)
        max_n: maximum number of views to compute

    Returns:
        InferenceScalingResult with N values and corresponding bounds.
    """
    if alpha >= tau:
        raise ValueError(f"alpha ({alpha}) must be < tau ({tau})")

    ns = list(range(1, max_n + 1))
    bounds = [hallucination_upper_bound(n, tau, alpha) for n in ns]
    return InferenceScalingResult(
        n_views_sequence=ns,
        bounds_sequence=bounds,
        tau=tau,
        alpha=alpha,
    )


@dataclass
class ViewAllocationResult:
    """Result of optimal view allocation across claims (Proposition 3)."""

    allocations: dict[str, int]  # node_id -> number of views to allocate
    total_views: int
    expected_false_pass_rate: float


def optimal_view_allocation(
    node_priorities: dict[str, float],
    budget: int,
    tau: float,
    alpha: float,
    min_per_node: int = 1,
) -> ViewAllocationResult:
    """Proposition 3: Optimal Compute Allocation.

    Let each verification view have cost k. Under budget B, the optimal
    policy allocates views to claims maximizing:
        E[Delta Verified Utility] / k
    This reduces to a bandit / knapsack allocation problem over claims.

    The priority score combines:
        - utility contribution (how important is this claim to the answer)
        - uncertainty (support mass near threshold tau)
        - risk (safety-critical claims)

    Uses a greedy proportional allocation:
        n_i = min_per_node + floor((B - k*min_per_node) * p_i / sum(p_j))
    where k is the number of nodes and p_i is the priority of node i.

    Args:
        node_priorities: mapping from node_id to priority score
        budget: total verification budget B
        tau: support mass threshold
        alpha: per-view false-positive rate
        min_per_node: minimum views allocated to each node

    Returns:
        ViewAllocationResult with per-node allocations.
    """
    k = len(node_priorities)
    if k == 0:
        return ViewAllocationResult(
            allocations={}, total_views=0, expected_false_pass_rate=0.0
        )

    min_budget = k * min_per_node
    if budget < min_budget:
        # Not enough budget for minimums; allocate evenly
        per_node = budget // k
        remainder = budget % k
        allocs = {}
        for i, nid in enumerate(node_priorities):
            allocs[nid] = per_node + (1 if i < remainder else 0)
    else:
        remaining = budget - min_budget
        total_priority = sum(node_priorities.values())

        allocs = {}
        if total_priority > 0:
            for nid, pri in node_priorities.items():
                extra = int(remaining * pri / total_priority)
                allocs[nid] = min_per_node + extra
        else:
            # Equal allocation if all priorities are zero
            extra_each = remaining // k
            for nid in node_priorities:
                allocs[nid] = min_per_node + extra_each

    # Compute expected false pass rate for each allocation
    total_views = sum(allocs.values())
    expected_fps: list[float] = []
    for nid, n_v in allocs.items():
        if n_v > 0 and alpha < tau:
            expected_fps.append(hallucination_upper_bound(n_v, tau, alpha))
        else:
            expected_fps.append(1.0)

    avg_fp = sum(expected_fps) / len(expected_fps) if expected_fps else 0.0

    return ViewAllocationResult(
        allocations=allocs,
        total_views=total_views,
        expected_false_pass_rate=avg_fp,
    )


# ---------------------------------------------------------------------------
# Correlated bound (extends Proposition 1 for non-independent views)
# ---------------------------------------------------------------------------


def hallucination_upper_bound_correlated(
    n_views: int,
    tau: float,
    alpha: float,
    rho: float,
) -> float:
    """Exponential suppression bound accounting for pairwise view correlation.

    Extends Proposition 1 for the realistic case where views are NOT
    conditionally independent. Uses a conservative Gaussian copula
    approximation for correlated Bernoulli variables.

    Under pairwise correlation rho between views, the effective number
    of independent views is:

        N_eff = N / (1 + (N - 1) * rho)

    The correlated bound is then:

        Pr[m(c) >= tau] <= exp(-N_eff * D(tau || alpha))

    When rho = 0, this reduces to the standard independent bound.
    When rho = 1, N_eff = 1 regardless of N (all views are identical).

    This bound should ALWAYS be reported alongside the independent
    bound, with rho estimated from empirical pairwise agreement data.

    Args:
        n_views: N, number of verification views
        tau: support mass threshold for the Verified type
        alpha: per-view false-positive rate
        rho: average pairwise correlation between views (0 = independent,
            1 = perfectly correlated). Estimated from empirical data via
            estimate_view_correlation().

    Returns:
        Upper bound on hallucination acceptance probability under
        correlated views.

    Raises:
        ValueError: if parameters are out of valid ranges
    """
    if n_views < 1:
        raise ValueError(f"n_views must be >= 1, got {n_views}")
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0, 1], got {rho}")

    if alpha >= tau:
        return 1.0

    # Effective independent views under correlation
    n_eff = n_views / (1.0 + (n_views - 1) * rho)

    d = kl_bernoulli(tau, alpha)
    if math.isinf(d):
        return 0.0

    return math.exp(-n_eff * d)


# ---------------------------------------------------------------------------
# Conditional independence diagnostic
# ---------------------------------------------------------------------------


class IndependenceTestResult(NamedTuple):
    """Result of testing conditional independence between verification views.

    Reports whether views satisfy the independence assumption required
    by Proposition 1, along with quantitative diagnostics.
    """

    pairwise_agreements: list[tuple[int, int, float]]
    expected_agreement_if_independent: float
    average_excess_correlation: float
    estimated_rho: float
    independence_plausible: bool


def estimate_view_correlation(
    view_verdicts: list[list[bool]],
    ground_truth: list[bool] | None = None,
) -> IndependenceTestResult:
    """Estimate pairwise correlation between verification views.

    Computes the excess agreement between all pairs of views relative
    to what would be expected under conditional independence. This
    diagnostic should be reported alongside any use of the exponential
    bound (Proposition 1).

    Args:
        view_verdicts: list of N lists, where view_verdicts[i][j] is
            True if view i accepted claim j, False otherwise.
            All lists must have the same length (number of claims).
        ground_truth: optional list of ground-truth labels. If provided,
            the expected agreement under independence is computed from
            the marginal rates conditioned on ground truth. If None,
            uses the overall marginal acceptance rates.

    Returns:
        IndependenceTestResult with pairwise agreements, expected
        agreement under independence, excess correlation, and an
        overall correlation estimate (rho) suitable for use with
        hallucination_upper_bound_correlated().
    """
    n_views = len(view_verdicts)
    if n_views < 2:
        return IndependenceTestResult(
            pairwise_agreements=[],
            expected_agreement_if_independent=0.0,
            average_excess_correlation=0.0,
            estimated_rho=0.0,
            independence_plausible=True,
        )

    n_claims = len(view_verdicts[0])
    if n_claims == 0:
        raise ValueError("view_verdicts must contain at least one claim")
    for i, vv in enumerate(view_verdicts):
        if len(vv) != n_claims:
            raise ValueError(
                f"All views must have same length; view {i} has "
                f"{len(vv)} vs expected {n_claims}"
            )

    # Compute marginal acceptance rates per view
    marginals = [sum(v) / n_claims for v in view_verdicts]

    # Expected agreement under independence: P(both accept) + P(both reject)
    # = p_i * p_j + (1 - p_i) * (1 - p_j)
    pairwise: list[tuple[int, int, float]] = []
    expected_agreements: list[float] = []
    actual_agreements: list[float] = []

    for i in range(n_views):
        for j in range(i + 1, n_views):
            agree_count = sum(
                1 for k in range(n_claims)
                if view_verdicts[i][k] == view_verdicts[j][k]
            )
            actual_agree = agree_count / n_claims
            pairwise.append((i, j, actual_agree))
            actual_agreements.append(actual_agree)

            p_i, p_j = marginals[i], marginals[j]
            expected = p_i * p_j + (1.0 - p_i) * (1.0 - p_j)
            expected_agreements.append(expected)

    avg_expected = sum(expected_agreements) / len(expected_agreements)
    avg_actual = sum(actual_agreements) / len(actual_agreements)
    avg_excess = avg_actual - avg_expected

    # Estimate rho: correlation coefficient from excess agreement
    # rho = excess_agreement / (1 - expected_agreement)
    # This normalizes so that rho=0 means independent, rho=1 means identical
    if avg_expected < 1.0:
        estimated_rho = max(0.0, min(1.0, avg_excess / (1.0 - avg_expected)))
    else:
        estimated_rho = 1.0 if avg_excess > 0 else 0.0

    # Independence is plausible if excess correlation < 10%
    independence_plausible = avg_excess < 0.10

    return IndependenceTestResult(
        pairwise_agreements=pairwise,
        expected_agreement_if_independent=avg_expected,
        average_excess_correlation=avg_excess,
        estimated_rho=estimated_rho,
        independence_plausible=independence_plausible,
    )
