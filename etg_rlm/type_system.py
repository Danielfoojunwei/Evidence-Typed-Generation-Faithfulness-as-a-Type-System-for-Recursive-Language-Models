"""Evidence-Typed Generation (ETG) — Evidence Confidence Grading (Section 4.4, 4.6).

Implements Definition 4 (Evidence Types) and Definition 5 (Well-Typed Output Space).

Definition 4 (Evidence Types):
    For thresholds tau > tau':
        type(c) =
            Verified      if m(c) >= tau
            Uncertain     if tau' < m(c) < tau
            Unsupported   if m(c) <= tau'
    This enables threshold-based filtering at decoding time.

Definition 5 (Well-Typed Output Space):
    V^tau = {v in V | type(pi(v)) = Verified}
    Y(G_T, tau) = {y | A(y) subset {pi(v) : v in V^tau}}
    y* = argmax_{y in Y(G_T, tau)} log p_theta(y | q, E)

The checker rejects an answer if it contains Unsupported claims.
ETG pipeline:
    1. Decompose generated text into atomic claims
    2. Verify claims via multi-view ensemble
    3. Grade each claim's confidence tier (Verified/Uncertain/Unsupported)
    4. Render only claims that pass the confidence threshold

Analogy to type systems: ETG is *inspired by* type systems in that
unsupported claims are structurally excluded from the output space,
similar to how ill-typed programs cannot compile. However, ETG does
not provide formal type-theoretic guarantees (compositionality,
soundness, decidability) in the programming-language sense. The
"types" are confidence tiers derived from ensemble verification
scores, not formal types derived from inference rules.

The genuine structural contribution is dependency-aware filtering
(Definition 5): claims whose ancestors fail verification are
transitively rejected regardless of their own scores, providing
a compositional propagation guarantee on the DAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from etg_rlm.core import (
    ClaimStatus,
    ClaimType,
    ESBGNode,
    EvidenceScopedBeliefGraph,
)


@dataclass
class TypeThresholds:
    """Threshold parameters for the ETG type system.

    tau: upper threshold — claims with m(c) >= tau are Verified
    tau_prime: lower threshold — claims with m(c) <= tau' are Unsupported
    Claims with tau' < m(c) < tau are Uncertain.
    """

    tau: float = 0.7
    tau_prime: float = 0.3

    def __post_init__(self) -> None:
        if not (0.0 <= self.tau_prime < self.tau <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 <= tau' < tau <= 1, "
                f"got tau'={self.tau_prime}, tau={self.tau}"
            )


class TypeCheckResult(NamedTuple):
    """Result of type-checking a single claim node."""

    node_id: str
    claim_type: ClaimType
    support_mass: float
    well_typed: bool


class GraphTypeCheckResult(NamedTuple):
    """Result of type-checking the entire ESBG."""

    well_typed: bool
    node_results: list[TypeCheckResult]
    verified_count: int
    uncertain_count: int
    unsupported_count: int


class EvidenceTypeChecker:
    """Evidence confidence grader for the ETG framework.

    Assigns confidence tiers to claims based on multi-view verification
    scores and validates that an answer contains only sufficiently
    supported (Verified) claims.

    Note: Despite the "type checker" name (retained for paper
    compatibility), this is a threshold-based confidence grader, not a
    formal type checker in the PL sense. It does not perform type
    inference, unification, or compositional type derivation. The
    structural guarantee it provides is dependency-aware filtering:
    claims whose ancestors are unsupported are transitively rejected.
    """

    def __init__(self, thresholds: TypeThresholds | None = None) -> None:
        self.thresholds = thresholds or TypeThresholds()

    def type_claim(self, node: ESBGNode) -> ClaimType:
        """Assign an evidence type to a claim based on its support mass.

        type(c) =
            Verified      if m(c) >= tau
            Uncertain     if tau' < m(c) < tau
            Unsupported   if m(c) <= tau'
        """
        m = node.support_mass
        if m >= self.thresholds.tau:
            return ClaimType.VERIFIED
        elif m > self.thresholds.tau_prime:
            return ClaimType.UNCERTAIN
        else:
            return ClaimType.UNSUPPORTED

    def check_node(self, node: ESBGNode) -> TypeCheckResult:
        """Type-check a single node and update its claim_type field."""
        claim_type = self.type_claim(node)
        node.claim_type = claim_type
        well_typed = claim_type == ClaimType.VERIFIED
        return TypeCheckResult(
            node_id=node.node_id,
            claim_type=claim_type,
            support_mass=node.support_mass,
            well_typed=well_typed,
        )

    def check_graph(self, esbg: EvidenceScopedBeliefGraph) -> GraphTypeCheckResult:
        """Type-check all nodes in an ESBG.

        The graph is well-typed iff every node is Verified.
        An answer is renderable only from well-typed graphs.
        """
        results: list[TypeCheckResult] = []
        verified = 0
        uncertain = 0
        unsupported = 0

        for node in esbg.topological_order():
            result = self.check_node(node)
            results.append(result)
            if result.claim_type == ClaimType.VERIFIED:
                verified += 1
            elif result.claim_type == ClaimType.UNCERTAIN:
                uncertain += 1
            else:
                unsupported += 1

        all_well_typed = unsupported == 0 and uncertain == 0
        return GraphTypeCheckResult(
            well_typed=all_well_typed,
            node_results=results,
            verified_count=verified,
            uncertain_count=uncertain,
            unsupported_count=unsupported,
        )

    def renderable_claims(
        self, esbg: EvidenceScopedBeliefGraph
    ) -> set[str]:
        """Return the set of node IDs whose claims can be rendered (Definition 5).

        V^tau = {v in V | type(pi(v)) = Verified}

        The allowed output space is:
            Y(G_T, tau) = {y | A(y) subset {pi(v) : v in V^tau}}

        Final decoding objective:
            y* = argmax_{y in Y(G_T, tau)} log p_theta(y | q, E)

        Dependency-aware: a node is only renderable if all its
        predecessors in the DAG are also renderable. This ensures
        compositional claims inherit grounding from their foundations.
        """
        renderable: set[str] = set()
        for node in esbg.topological_order():
            ct = self.type_claim(node)
            if ct == ClaimType.VERIFIED and node.status == ClaimStatus.ENTAILED:
                # Also check that all dependencies are renderable
                deps = esbg.get_dependencies(node.node_id)
                if all(d.node_id in renderable for d in deps):
                    renderable.add(node.node_id)
        return renderable

    def graduated_output(
        self, esbg: EvidenceScopedBeliefGraph
    ) -> GraduatedOutputResult:
        """Produce graduated output with confidence annotations per claim.

        Instead of binary accept/reject, assigns each claim a confidence
        tier and renders all claims with annotations indicating their
        verification status. This preserves recall while still
        communicating verification confidence to the user.

        Returns:
            GraduatedOutputResult with per-tier claim lists and
            annotated rendered text.
        """
        high_confidence: list[str] = []
        moderate_confidence: list[str] = []
        unverified: list[str] = []

        for node in esbg.topological_order():
            ct = self.type_claim(node)
            deps = esbg.get_dependencies(node.node_id)
            ancestors_ok = all(
                self.type_claim(d) == ClaimType.VERIFIED for d in deps
            )

            if ct == ClaimType.VERIFIED and ancestors_ok:
                high_confidence.append(node.node_id)
            elif ct == ClaimType.UNCERTAIN:
                moderate_confidence.append(node.node_id)
            else:
                unverified.append(node.node_id)

        parts: list[str] = []
        for node in esbg.topological_order():
            nid = node.node_id
            text = node.claim.text
            if nid in high_confidence:
                parts.append(text)
            elif nid in moderate_confidence:
                parts.append(f"[moderate confidence] {text}")
            else:
                parts.append(f"[unverified] {text}")

        return GraduatedOutputResult(
            high_confidence_ids=high_confidence,
            moderate_confidence_ids=moderate_confidence,
            unverified_ids=unverified,
            annotated_text=" ".join(parts),
        )


class GraduatedOutputResult(NamedTuple):
    """Result of graduated output rendering.

    Instead of binary filtering, all claims are included with
    confidence annotations, preserving coverage while communicating
    verification status.
    """

    high_confidence_ids: list[str]
    moderate_confidence_ids: list[str]
    unverified_ids: list[str]
    annotated_text: str


def dag_propagation_bound(
    leaf_precision: float,
    depth: int,
) -> float:
    """Compositional guarantee for DAG-based dependency propagation.

    If all leaf claims in a dependency chain are independently verified
    with precision p (probability that a "Verified" claim is truly
    supported), then the probability that the root claim is supported
    given all its ancestors are verified is:

        P(root supported | all ancestors verified) >= p^depth

    This formalizes the genuine compositional contribution of ETG's
    DAG structure: verification errors compound multiplicatively
    along dependency chains, but the bound is tight and meaningful.

    Args:
        leaf_precision: probability that an individual verified claim
            is truly supported by evidence (verifier precision).
        depth: maximum depth of the dependency chain from root to
            deepest leaf.

    Returns:
        Lower bound on probability that root claim is truly supported,
        given that all claims in the chain are individually verified.
    """
    if not (0.0 <= leaf_precision <= 1.0):
        raise ValueError(
            f"leaf_precision must be in [0, 1], got {leaf_precision}"
        )
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    if depth == 0:
        return leaf_precision
    return leaf_precision ** depth
