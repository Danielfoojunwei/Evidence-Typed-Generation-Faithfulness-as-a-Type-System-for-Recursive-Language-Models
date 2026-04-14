"""Baseline configurations for ETG evaluation (Section 2 of eval plan).

Defines the four baselines that ETG is compared against. All baselines
use the same generator model to ensure a fair comparison of the
*framework*, not the underlying model.

Control 1: Standard LLM     -- zero-shot, no retrieval augmentation
Control 2: Standard RAG     -- generator + dense retriever (top-k)
Control 3: RAG + Verifier   -- RAG with post-hoc claim verification
Control 4: Self-Critique    -- single-view LLM self-check

Each baseline is defined as a configuration that can be instantiated
with concrete model implementations via the protocol interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from etg_rlm.core import AtomicClaim, ClaimStatus, ClaimType, EvidenceSpan


class BaselineType(Enum):
    """Baseline configurations from the evaluation plan.

    Original baselines (Controls 1-4) and additional SOTA baselines
    added to address the critical gap in comparative evaluation.
    """

    STANDARD_LLM = "standard_llm"
    STANDARD_RAG = "standard_rag"
    RAG_VERIFIER = "rag_verifier"
    SELF_CRITIQUE = "self_critique"
    # SOTA baselines (added to address missing baseline comparisons)
    SELF_CHECK_GPT = "self_check_gpt"
    CHAIN_OF_VERIFICATION = "chain_of_verification"
    # Trivial baselines (to contextualize ETG improvements)
    RANDOM_FILTER = "random_filter"
    FIRST_K_SENTENCES = "first_k_sentences"
    SINGLE_NLI_THRESHOLD = "single_nli_threshold"


@runtime_checkable
class Generator(Protocol):
    """Protocol for the generator LLM (e.g., Llama 3 70B)."""

    def generate(self, query: str, context: str | None = None) -> str:
        """Generate an answer to the query, optionally with retrieved context."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Protocol for dense/sparse retrieval (e.g., FAISS, BM25)."""

    def retrieve(self, query: str, corpus_id: str, top_k: int = 5) -> list[EvidenceSpan]:
        """Retrieve top-k evidence spans from the corpus."""
        ...


@runtime_checkable
class PostHocVerifier(Protocol):
    """Protocol for post-hoc claim verification (used in RAG+Verifier baseline)."""

    def verify_claims(
        self, claims: list[AtomicClaim], context: list[EvidenceSpan]
    ) -> list[tuple[AtomicClaim, ClaimStatus]]:
        """Verify each claim against the retrieved context."""
        ...


@runtime_checkable
class SelfCritiquer(Protocol):
    """Protocol for LLM self-critique (used in Self-Critique baseline)."""

    def critique(self, query: str, answer: str) -> str:
        """Ask the LLM to critique and revise its own answer."""
        ...


@dataclass
class BaselineConfig:
    """Configuration for a baseline run.

    Attributes:
        baseline_type: which baseline to run
        name: human-readable name for reporting
        top_k: number of retrieved passages (for RAG baselines)
        corpus_id: evidence corpus identifier
    """

    baseline_type: BaselineType
    name: str
    top_k: int = 5
    corpus_id: str = "default"


@dataclass
class BaselineResult:
    """Result of running a baseline.

    Attributes:
        config: the baseline configuration used
        query: the input query
        generated_text: the raw generated answer
        final_text: the answer after any post-hoc filtering
        claims: extracted atomic claims
        supported_claims: claims deemed supported (if verification was done)
        rejected_claims: claims deemed unsupported (if verification was done)
        retrieved_spans: evidence spans retrieved (if RAG was used)
    """

    config: BaselineConfig
    query: str
    generated_text: str
    final_text: str
    claims: list[AtomicClaim] = field(default_factory=list)
    supported_claims: list[AtomicClaim] = field(default_factory=list)
    rejected_claims: list[AtomicClaim] = field(default_factory=list)
    retrieved_spans: list[EvidenceSpan] = field(default_factory=list)


class BaselineRunner(ABC):
    """Abstract base for running a baseline configuration."""

    def __init__(self, config: BaselineConfig) -> None:
        self.config = config

    @abstractmethod
    def run(self, query: str) -> BaselineResult:
        """Run the baseline on a query and return the result."""
        ...


class StandardLLMBaseline(BaselineRunner):
    """Control 1: Standard LLM (zero-shot, no retrieval).

    The base generator model with no retrieval augmentation.
    Establishes the base rate of hallucination.
    """

    def __init__(self, generator: Generator, config: BaselineConfig | None = None) -> None:
        super().__init__(config or BaselineConfig(
            baseline_type=BaselineType.STANDARD_LLM,
            name="Standard LLM (zero-shot)",
        ))
        self.generator = generator

    def run(self, query: str) -> BaselineResult:
        text = self.generator.generate(query)
        return BaselineResult(
            config=self.config,
            query=query,
            generated_text=text,
            final_text=text,
        )


class StandardRAGBaseline(BaselineRunner):
    """Control 2: Standard RAG (generator + dense retriever).

    The generator augmented with a simple dense retriever (top-k results).
    Represents the current industry standard for reducing hallucinations.
    """

    def __init__(
        self,
        generator: Generator,
        retriever: Retriever,
        config: BaselineConfig | None = None,
    ) -> None:
        super().__init__(config or BaselineConfig(
            baseline_type=BaselineType.STANDARD_RAG,
            name="Standard RAG",
        ))
        self.generator = generator
        self.retriever = retriever

    def run(self, query: str) -> BaselineResult:
        spans = self.retriever.retrieve(query, self.config.corpus_id, self.config.top_k)
        context = "\n".join(s.text for s in spans if s.text)
        text = self.generator.generate(query, context=context)
        return BaselineResult(
            config=self.config,
            query=query,
            generated_text=text,
            final_text=text,
            retrieved_spans=spans,
        )


class RAGVerifierBaseline(BaselineRunner):
    """Control 3: RAG + Post-hoc Verifier.

    A standard RAG system where a verifier flags or retracts unsupported
    claims *after* the full text has been generated. Tests whether ETG's
    preventative constrained decoding is more effective than a corrective
    post-hoc check.
    """

    def __init__(
        self,
        generator: Generator,
        retriever: Retriever,
        claim_extractor: object,  # ClaimExtractor protocol
        verifier: PostHocVerifier,
        config: BaselineConfig | None = None,
    ) -> None:
        super().__init__(config or BaselineConfig(
            baseline_type=BaselineType.RAG_VERIFIER,
            name="RAG + Verifier",
        ))
        self.generator = generator
        self.retriever = retriever
        self.claim_extractor = claim_extractor
        self.verifier = verifier

    def run(self, query: str) -> BaselineResult:
        spans = self.retriever.retrieve(query, self.config.corpus_id, self.config.top_k)
        context = "\n".join(s.text for s in spans if s.text)
        text = self.generator.generate(query, context=context)

        # Extract claims then verify post-hoc
        claims = self.claim_extractor.extract(text)  # type: ignore[attr-defined]
        verdicts = self.verifier.verify_claims(claims, spans)

        supported = [c for c, s in verdicts if s == ClaimStatus.ENTAILED]
        rejected = [c for c, s in verdicts if s != ClaimStatus.ENTAILED]

        # Reconstruct text from supported claims only
        final = " ".join(c.text for c in supported) if supported else ""

        return BaselineResult(
            config=self.config,
            query=query,
            generated_text=text,
            final_text=final,
            claims=claims,
            supported_claims=supported,
            rejected_claims=rejected,
            retrieved_spans=spans,
        )


class SelfCritiqueBaseline(BaselineRunner):
    """Control 4: Self-Critique (single-view LLM self-check).

    A single-view verification where the LLM is prompted to check its
    own claims. Tests whether ETG's multi-view, structurally enforced
    approach is more robust than behavioral prompting.
    """

    def __init__(
        self,
        generator: Generator,
        critiquer: SelfCritiquer,
        config: BaselineConfig | None = None,
    ) -> None:
        super().__init__(config or BaselineConfig(
            baseline_type=BaselineType.SELF_CRITIQUE,
            name="Self-Critique",
        ))
        self.generator = generator
        self.critiquer = critiquer

    def run(self, query: str) -> BaselineResult:
        text = self.generator.generate(query)
        revised = self.critiquer.critique(query, text)
        return BaselineResult(
            config=self.config,
            query=query,
            generated_text=text,
            final_text=revised,
        )


# ---------------------------------------------------------------------------
# Convenience: list all baseline configs
# ---------------------------------------------------------------------------

class SelfCheckGPTBaseline(BaselineRunner):
    """SOTA Baseline: SelfCheckGPT (Manakul et al., EMNLP 2023).

    Detects hallucinations by sampling multiple responses from the same
    model and measuring consistency. Hallucinated facts vary across
    samples while grounded facts remain stable.

    This baseline tests whether ETG's multi-view external verification
    outperforms single-model internal consistency checking.
    """

    def __init__(
        self,
        generator: Generator,
        n_samples: int = 5,
        config: BaselineConfig | None = None,
    ) -> None:
        super().__init__(config or BaselineConfig(
            baseline_type=BaselineType.SELF_CHECK_GPT,
            name="SelfCheckGPT (consistency-based)",
        ))
        self.generator = generator
        self.n_samples = n_samples

    def run(self, query: str) -> BaselineResult:
        primary = self.generator.generate(query)
        samples = [self.generator.generate(query) for _ in range(self.n_samples)]
        # Sentence-level consistency check: keep sentences that appear
        # (semantically) in the majority of samples
        sentences = [s.strip() for s in primary.split(".") if s.strip()]
        kept: list[str] = []
        for sent in sentences:
            matches = sum(1 for sample in samples if sent.lower() in sample.lower())
            if matches >= self.n_samples // 2:
                kept.append(sent)
        final = ". ".join(kept) + "." if kept else ""
        return BaselineResult(
            config=self.config,
            query=query,
            generated_text=primary,
            final_text=final,
        )


class ChainOfVerificationBaseline(BaselineRunner):
    """SOTA Baseline: Chain-of-Verification (Dhuliawala et al., 2024).

    Generates verification questions from the original response, answers
    them independently, and revises the response based on any discovered
    inconsistencies.

    This baseline tests whether ETG's external multi-view verification
    outperforms self-generated verification questions.
    """

    def __init__(
        self,
        generator: Generator,
        critiquer: SelfCritiquer,
        config: BaselineConfig | None = None,
    ) -> None:
        super().__init__(config or BaselineConfig(
            baseline_type=BaselineType.CHAIN_OF_VERIFICATION,
            name="Chain-of-Verification (CoVe)",
        ))
        self.generator = generator
        self.critiquer = critiquer

    def run(self, query: str) -> BaselineResult:
        text = self.generator.generate(query)
        # CoVe: critique acts as plan-verify-revise loop
        revised = self.critiquer.critique(query, text)
        return BaselineResult(
            config=self.config,
            query=query,
            generated_text=text,
            final_text=revised,
        )


class RandomFilterBaseline(BaselineRunner):
    """Trivial Baseline: Random sentence filtering.

    Randomly retains a fraction of generated sentences to match ETG's
    retention rate. If ETG's precision improvement is primarily due to
    aggressive filtering (discarding 69-94% of content), this baseline
    will show similar precision gains, debunking ETG's added value.

    This is a critical ablation: it tests whether the improvement comes
    from *which* sentences are kept (ETG's verification) or simply from
    *how many* are discarded (any filtering).
    """

    def __init__(
        self,
        generator: Generator,
        retention_rate: float = 0.31,
        seed: int = 42,
        config: BaselineConfig | None = None,
    ) -> None:
        super().__init__(config or BaselineConfig(
            baseline_type=BaselineType.RANDOM_FILTER,
            name=f"Random Filter ({retention_rate:.0%} retention)",
        ))
        self.generator = generator
        self.retention_rate = retention_rate
        self._rng = __import__("random").Random(seed)

    def run(self, query: str) -> BaselineResult:
        text = self.generator.generate(query)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        n_keep = max(1, int(len(sentences) * self.retention_rate))
        kept = self._rng.sample(sentences, min(n_keep, len(sentences)))
        final = ". ".join(kept) + "." if kept else ""
        return BaselineResult(
            config=self.config,
            query=query,
            generated_text=text,
            final_text=final,
        )


class FirstKSentencesBaseline(BaselineRunner):
    """Trivial Baseline: Keep only the first K sentences.

    Tests whether ETG's precision improvement is explained by the
    observation that opening sentences tend to be more factual than
    later ones (models often start accurate then drift into hallucination).
    """

    def __init__(
        self,
        generator: Generator,
        k: int = 3,
        config: BaselineConfig | None = None,
    ) -> None:
        super().__init__(config or BaselineConfig(
            baseline_type=BaselineType.FIRST_K_SENTENCES,
            name=f"First-{k} Sentences",
        ))
        self.generator = generator
        self.k = k

    def run(self, query: str) -> BaselineResult:
        text = self.generator.generate(query)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        kept = sentences[: self.k]
        final = ". ".join(kept) + "." if kept else ""
        return BaselineResult(
            config=self.config,
            query=query,
            generated_text=text,
            final_text=final,
        )


class SingleNLIThresholdBaseline(BaselineRunner):
    """Trivial Baseline: Single NLI model with threshold filtering.

    Uses a single NLI model (matching ETG's strongest individual view)
    with matched compute budget. Tests whether ETG's multi-view ensemble
    and meta-classifier machinery adds value over the simplest possible
    verification approach.
    """

    def __init__(
        self,
        generator: Generator,
        verifier: PostHocVerifier,
        claim_extractor: object,
        config: BaselineConfig | None = None,
    ) -> None:
        super().__init__(config or BaselineConfig(
            baseline_type=BaselineType.SINGLE_NLI_THRESHOLD,
            name="Single NLI Threshold",
        ))
        self.generator = generator
        self.verifier = verifier
        self.claim_extractor = claim_extractor

    def run(self, query: str) -> BaselineResult:
        text = self.generator.generate(query)
        claims = self.claim_extractor.extract(text)  # type: ignore[attr-defined]
        verdicts = self.verifier.verify_claims(claims, [])
        supported = [c for c, s in verdicts if s == ClaimStatus.ENTAILED]
        rejected = [c for c, s in verdicts if s != ClaimStatus.ENTAILED]
        final = " ".join(c.text for c in supported) if supported else ""
        return BaselineResult(
            config=self.config,
            query=query,
            generated_text=text,
            final_text=final,
            claims=claims,
            supported_claims=supported,
            rejected_claims=rejected,
        )


# ---------------------------------------------------------------------------
# Convenience: list all baseline configs
# ---------------------------------------------------------------------------

BASELINE_CONFIGS = [
    BaselineConfig(
        baseline_type=BaselineType.STANDARD_LLM,
        name="Control 1: Standard LLM (zero-shot)",
    ),
    BaselineConfig(
        baseline_type=BaselineType.STANDARD_RAG,
        name="Control 2: Standard RAG (top-k retrieval)",
    ),
    BaselineConfig(
        baseline_type=BaselineType.RAG_VERIFIER,
        name="Control 3: RAG + Post-hoc Verifier",
    ),
    BaselineConfig(
        baseline_type=BaselineType.SELF_CRITIQUE,
        name="Control 4: Self-Critique (single-view)",
    ),
    BaselineConfig(
        baseline_type=BaselineType.SELF_CHECK_GPT,
        name="SOTA: SelfCheckGPT (Manakul et al., 2023)",
    ),
    BaselineConfig(
        baseline_type=BaselineType.CHAIN_OF_VERIFICATION,
        name="SOTA: Chain-of-Verification (Dhuliawala et al., 2024)",
    ),
    BaselineConfig(
        baseline_type=BaselineType.RANDOM_FILTER,
        name="Trivial: Random Filter (matched retention rate)",
    ),
    BaselineConfig(
        baseline_type=BaselineType.FIRST_K_SENTENCES,
        name="Trivial: First-K Sentences",
    ),
    BaselineConfig(
        baseline_type=BaselineType.SINGLE_NLI_THRESHOLD,
        name="Trivial: Single NLI Threshold (matched compute)",
    ),
]
