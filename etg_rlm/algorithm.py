"""EBRG Algorithm (Section 5) and Constrained Decoding (Section 4.6).

Implements the core iterative verification algorithm:

    def EBRG(q, E, tau, N, budget):
        G = initialize_graph()
        frontier = propose_initial_claims(q)
        while budget_remaining(budget):
            c = select_claim(frontier, G)
            if not verified(c):
                for i in range(N):
                    z_i, S_i = verify_view(E, c, view=i)
                    update_support(c, z_i, S_i)
            if support_mass(c) >= tau:
                mark_verified(c)
                expand_dependencies(c, frontier)
            else:
                mark_unverified(c)
        V_tau = {c for c in G if support_mass(c) >= tau}
        return render_answer(V_tau)

Constrained Decoding (Section 4.6, Definition 5):
    V^tau = {v in V | type(pi(v)) = Verified}
    Y(G_T, tau) = {y | A(y) subset {pi(v) : v in V^tau}}
    y* = argmax_{y in Y(G_T, tau)} log p_theta(y | q, E)

    Unsupported claims are excluded from the output space via
    threshold-based filtering on the ESBG DAG.

Note: The algorithm is iterative (for-loop over topologically ordered
nodes with a budget counter), not recursive in the formal sense.
The "recursive" label in the paper title refers to the conceptual
recursion of expanding claim dependencies, not to algorithmic recursion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from etg_rlm.core import (
    AtomicClaim,
    ClaimStatus,
    ClaimType,
    ESBGNode,
    EvidenceScopedBeliefGraph,
    EvidenceSpan,
)
from etg_rlm.verification import (
    ClaimExtractor,
    MultiViewVerifier,
    VerificationView,
)
from etg_rlm.type_system import (
    EvidenceTypeChecker,
    GraphTypeCheckResult,
    TypeThresholds,
)
from etg_rlm.bounds import (
    check_zero_confabulation,
    hallucination_upper_bound,
)


class ConstrainedDecodingResult(NamedTuple):
    """Result of constrained decoding over an ESBG (Definition 5).

    V^tau = {v in V | type(pi(v)) = Verified}
    Y(G_T, tau) = {y | A(y) subset {pi(v) : v in V^tau}}
    """

    verified_node_ids: set[str]
    verified_claims: list[AtomicClaim]
    rejected_claims: list[AtomicClaim]
    rendered_text: str


def constrained_decode(
    esbg: EvidenceScopedBeliefGraph,
    type_checker: EvidenceTypeChecker,
) -> ConstrainedDecodingResult:
    """Apply constrained decoding to an ESBG (Section 4.6, Definition 5).

    Computes:
        V^tau = {v in V | type(pi(v)) = Verified}
        Y(G_T, tau) = {y | A(y) subset {pi(v) : v in V^tau}}

    Then renders the output from only the verified claims.
    Unsupported claims are excluded from the output space.
    """
    renderable_ids = type_checker.renderable_claims(esbg)

    verified: list[AtomicClaim] = []
    rejected: list[AtomicClaim] = []

    for node in esbg.topological_order():
        if node.node_id in renderable_ids:
            verified.append(node.claim)
        else:
            rejected.append(node.claim)

    rendered = " ".join(c.text for c in verified) if verified else ""

    return ConstrainedDecodingResult(
        verified_node_ids=renderable_ids,
        verified_claims=verified,
        rejected_claims=rejected,
        rendered_text=rendered,
    )


@dataclass
class EBRGStepLog:
    """Log entry for a single step of the EBRG algorithm."""

    step: int
    node_id: str
    view_index: int
    verdict: ClaimStatus
    n_spans_found: int
    support_mass_after: float


@dataclass
class EBRGResult:
    """Result of running the EBRG algorithm (Section 5).

    Contains the final ESBG, constrained decoding result,
    evidence pointer guarantee check (revised Proposition 2),
    and the theoretical hallucination bound (Proposition 1).
    """

    esbg: EvidenceScopedBeliefGraph
    type_check: GraphTypeCheckResult
    decoding: ConstrainedDecodingResult
    hallucination_bound: float
    zero_confabulation_holds: bool
    budget_used: int
    step_log: list[EBRGStepLog]
    graduated_output: str = ""


def ebrg(
    query: str,
    claims: list[AtomicClaim],
    views: list[VerificationView],
    tau: float = 0.7,
    tau_prime: float = 0.3,
    n_views_per_claim: int | None = None,
    budget: int = 100,
    corpus_id: str = "default",
    dependencies: list[tuple[str, str]] | None = None,
) -> EBRGResult:
    """EBRG: Evidence-Based Recursive Generation (Section 5).

    The core algorithm from the paper:

        1. Initialize graph G with claim nodes
        2. For each claim in the frontier:
           a. Run N verification views
           b. Compute support mass
           c. If m(c) >= tau: mark verified, expand dependencies
           d. Else: mark unverified
        3. Compute V^tau = {c in G | m(c) >= tau}
        4. Render answer from V^tau via constrained decoding

    This is the reference implementation of the pseudocode in Section 5,
    with full tracking of the verification process for auditability.

    Args:
        query: the original query q
        claims: pre-extracted atomic claims A(y) = {c_1, ..., c_m}
        views: list of N verification views {V_1, ..., V_N}
        tau: support mass threshold for Verified type (Definition 4)
        tau_prime: lower threshold for Unsupported type (Definition 4)
        n_views_per_claim: views to run per claim (default: all views)
        budget: maximum total verification calls
        corpus_id: identifier for the evidence corpus E
        dependencies: optional pre-detected dependency edges (from_id, to_id)

    Returns:
        EBRGResult with the complete audit trail and verified output.
    """
    if n_views_per_claim is None:
        n_views_per_claim = len(views)

    # Step 1: Initialize graph G
    esbg_graph = EvidenceScopedBeliefGraph()
    for claim in claims:
        node = ESBGNode(node_id=claim.claim_id, claim=claim)
        esbg_graph.add_node(node)

    # Add dependency edges
    if dependencies:
        for from_id, to_id in dependencies:
            try:
                esbg_graph.add_dependency(from_id, to_id)
            except ValueError:
                pass

    # Step 2: Verify each claim with N views
    step_log: list[EBRGStepLog] = []
    budget_used = 0
    step_count = 0

    for node in esbg_graph.topological_order():
        if budget_used >= budget:
            break

        all_verdicts: list[bool] = []
        all_spans: set[EvidenceSpan] = set()
        entailed_count = 0

        n_to_run = min(n_views_per_claim, len(views), budget - budget_used)

        for i in range(n_to_run):
            view = views[i % len(views)]
            result = view.verify(node.claim, corpus_id)
            budget_used += 1
            step_count += 1

            is_entailed = result.verdict == ClaimStatus.ENTAILED
            all_verdicts.append(is_entailed)
            if is_entailed:
                entailed_count += 1
                all_spans |= result.spans

            # Update support mass incrementally
            node.support_mass = entailed_count / len(all_verdicts)
            node.view_verdicts = list(all_verdicts)
            node.evidence_spans = set(all_spans)

            step_log.append(EBRGStepLog(
                step=step_count,
                node_id=node.node_id,
                view_index=i,
                verdict=result.verdict,
                n_spans_found=len(result.spans),
                support_mass_after=node.support_mass,
            ))

        # Determine status
        if node.support_mass >= tau:
            node.status = ClaimStatus.ENTAILED
        elif node.support_mass <= tau_prime:
            node.status = ClaimStatus.UNKNOWN
        else:
            node.status = ClaimStatus.UNKNOWN

    # Step 3: Type-check
    type_checker = EvidenceTypeChecker(TypeThresholds(tau=tau, tau_prime=tau_prime))
    type_result = type_checker.check_graph(esbg_graph)

    # Step 4: Constrained decoding (Definition 5)
    decoding = constrained_decode(esbg_graph, type_checker)

    # Step 4b: Graduated output (preserves coverage with annotations)
    graduated = type_checker.graduated_output(esbg_graph)

    # Verify evidence pointer guarantee (revised Proposition 2)
    node_evidence_counts = {
        nid: len(esbg_graph.get_node(nid).evidence_spans)
        for nid in esbg_graph.all_node_ids()
    }
    node_support_masses = {
        nid: esbg_graph.get_node(nid).support_mass
        for nid in esbg_graph.all_node_ids()
    }
    confab_check = check_zero_confabulation(
        decoding.verified_node_ids, node_evidence_counts, node_support_masses, tau
    )

    # Compute Proposition 1 bound
    if decoding.verified_node_ids:
        min_views = min(
            len(esbg_graph.get_node(nid).view_verdicts)
            for nid in decoding.verified_node_ids
        )
        # Conservative alpha estimate: 0.1
        bound = hallucination_upper_bound(min_views, tau, 0.1) if min_views > 0 else 1.0
    else:
        bound = 1.0

    return EBRGResult(
        esbg=esbg_graph,
        type_check=type_result,
        decoding=decoding,
        hallucination_bound=bound,
        zero_confabulation_holds=confab_check.satisfies_proposition,
        budget_used=budget_used,
        step_log=step_log,
        graduated_output=graduated.annotated_text,
    )
