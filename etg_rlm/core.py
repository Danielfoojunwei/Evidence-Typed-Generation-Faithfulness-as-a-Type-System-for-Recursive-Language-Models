"""Core data structures for Evidence-Typed Generation (Section 4.2).

Implements Definition 1 (ESBG) from the paper:

    An ESBG is a tuple G = (V, ->, pi, sigma, m, z) where:
      - V: set of nodes
      - pi(v): atomic claim associated with node v
      - sigma(v) subset S(E): evidence span pointers
      - m(v) in [0,1]: support mass
      - z(v) in {entailed, contradicted, unknown}
      - u -> v: dependency edge (claim v depends on claim u)

    The graph is a DAG. ESBGs are constructed at inference time,
    not pre-computed or static.

A claim c is entailed iff:
    exists s in S(E) s.t. E[s] |= c

The structural mismatch motivating ETG: the classical LLM objective
    y* = argmax_y log p_theta(y | q, E)
does not penalize hallucinations unless the training distribution
explicitly does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import networkx as nx


class ClaimStatus(Enum):
    """Entailment status z(v) of a claim node."""

    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class ClaimType(Enum):
    """Evidence confidence tier assigned by the ETG grader (Definition 4).

    For thresholds tau > tau':

        type(c) =
            Verified      if m(c) >= tau
            Uncertain     if tau' < m(c) < tau
            Unsupported   if m(c) <= tau'

    Note: These "types" are confidence tiers derived from ensemble
    verification scores, not formal types in the programming-language
    sense. There are no inference rules, no compositionality guarantees,
    and no soundness theorem. The structural guarantee ETG provides is
    dependency-aware filtering on the ESBG DAG.
    """

    VERIFIED = "verified"
    UNCERTAIN = "uncertain"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class EvidenceSpan:
    """A span s in S(E), the set of all addressable evidence spans.

    Represents (doc_id, start, end) -- a contiguous region of text
    in a specific document. Used in the support relation:
        supp(E, c) subset S(E)
    returning spans s in E that entailedly support claim c.
    """

    doc_id: str
    start: int
    end: int
    text: str = ""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(
                f"Invalid span bounds: start={self.start}, end={self.end}"
            )


@dataclass
class AtomicClaim:
    """An atomic factual assertion extracted from a generated answer.

    Produced by the atomic claim decomposition operator (Section 4.1):
        A(y) = {c_1, ..., c_m}

    A claim c is grounded iff exists s in supp(E, c).
    A claim c is hallucinated iff supp(E, c) = empty set.
    """

    claim_id: str
    text: str
    subclaims: list[str] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.claim_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AtomicClaim):
            return NotImplemented
        return self.claim_id == other.claim_id


@dataclass
class ESBGNode:
    """A node v in the Evidence-Scoped Belief Graph (Definition 1).

    Each node in G = (V, ->, pi, sigma, m, z) carries:
      - pi(v) = claim: atomic claim associated with node v
      - sigma(v) = evidence_spans: evidence span pointers, subset S(E)
      - m(v) = support_mass: multi-view support mass in [0, 1]
      - z(v) = status: entailment status {entailed, contradicted, unknown}
      - type(v) = claim_type: evidence type (Definition 4, assigned by type checker)
    """

    node_id: str
    claim: AtomicClaim
    evidence_spans: set[EvidenceSpan] = field(default_factory=set)
    support_mass: float = 0.0
    status: ClaimStatus = ClaimStatus.UNKNOWN
    claim_type: ClaimType = ClaimType.UNSUPPORTED
    view_verdicts: list[bool] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ESBGNode):
            return NotImplemented
        return self.node_id == other.node_id


class EvidenceScopedBeliefGraph:
    """Evidence-Scoped Belief Graph -- Definition 1 (Section 4.2).

    An ESBG is a tuple G = (V, ->, pi, sigma, m, z) where:
      - V: set of nodes
      - pi(v): atomic claim associated with node v
      - sigma(v) subset S(E): evidence span pointers
      - m(v) in [0,1]: support mass (Definition 3)
      - z(v) in {entailed, contradicted, unknown}
      - u -> v: dependency edge (claim v depends on claim u)

    The graph is a DAG. ESBGs are constructed at inference time via the
    recursive graph construction policy rho (Section 4.5), not pre-computed
    or static. This is not "chain-of-thought" -- it is an externalized
    belief structure with explicit provenance.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: dict[str, ESBGNode] = {}

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def nodes(self) -> dict[str, ESBGNode]:
        return dict(self._nodes)

    def add_node(self, node: ESBGNode) -> None:
        """Add a claim node to the ESBG."""
        if node.node_id in self._nodes:
            raise ValueError(f"Node {node.node_id!r} already exists")
        self._nodes[node.node_id] = node
        self._graph.add_node(node.node_id)

    def add_dependency(self, from_id: str, to_id: str) -> None:
        """Add a dependency edge: claim to_id depends on claim from_id.

        Raises ValueError if adding the edge would create a cycle,
        since the ESBG must remain a DAG.
        """
        for nid in (from_id, to_id):
            if nid not in self._nodes:
                raise ValueError(f"Node {nid!r} not in graph")

        self._graph.add_edge(from_id, to_id)
        if not nx.is_directed_acyclic_graph(self._graph):
            self._graph.remove_edge(from_id, to_id)
            raise ValueError(
                f"Adding edge {from_id!r} -> {to_id!r} would create a cycle"
            )

    def get_node(self, node_id: str) -> ESBGNode:
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id!r} not found")
        return self._nodes[node_id]

    def get_dependencies(self, node_id: str) -> list[ESBGNode]:
        """Return the nodes that node_id depends on (its predecessors)."""
        return [self._nodes[pid] for pid in self._graph.predecessors(node_id)]

    def get_dependents(self, node_id: str) -> list[ESBGNode]:
        """Return the nodes that depend on node_id (its successors)."""
        return [self._nodes[sid] for sid in self._graph.successors(node_id)]

    def topological_order(self) -> list[ESBGNode]:
        """Return nodes in topological order (dependencies before dependents)."""
        return [self._nodes[nid] for nid in nx.topological_sort(self._graph)]

    def verified_subgraph(self, tau: float) -> set[str]:
        """Return V^tau: nodes with support_mass >= tau and status ENTAILED.

        V^tau = {v in V : m(pi(v)) >= tau AND z(v) = entailed}
        """
        return {
            nid
            for nid, node in self._nodes.items()
            if node.support_mass >= tau and node.status == ClaimStatus.ENTAILED
        }

    def all_node_ids(self) -> set[str]:
        return set(self._nodes.keys())

    def num_nodes(self) -> int:
        return len(self._nodes)

    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    def summary(self) -> dict:
        """Return a summary of the graph state."""
        status_counts = {s: 0 for s in ClaimStatus}
        type_counts = {t: 0 for t in ClaimType}
        for node in self._nodes.values():
            status_counts[node.status] += 1
            type_counts[node.claim_type] += 1
        return {
            "num_nodes": self.num_nodes(),
            "num_edges": self.num_edges(),
            "status_counts": {s.value: c for s, c in status_counts.items()},
            "type_counts": {t.value: c for t, c in type_counts.items()},
            "mean_support_mass": (
                sum(n.support_mass for n in self._nodes.values()) / len(self._nodes)
                if self._nodes
                else 0.0
            ),
        }
