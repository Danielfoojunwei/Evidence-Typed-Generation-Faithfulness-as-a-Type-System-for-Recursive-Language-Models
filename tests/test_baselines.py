"""Tests for baseline configurations (Section 2 of eval plan)."""

import pytest

from etg_rlm.core import AtomicClaim, ClaimStatus, EvidenceSpan
from etg_rlm.baselines import (
    BASELINE_CONFIGS,
    BaselineConfig,
    BaselineResult,
    BaselineType,
    StandardLLMBaseline,
    StandardRAGBaseline,
    SelfCritiqueBaseline,
)


class StubGenerator:
    def generate(self, query: str, context: str | None = None) -> str:
        if context:
            return f"Answer to '{query}' with context."
        return f"Answer to '{query}' zero-shot."


class StubRetriever:
    def retrieve(self, query: str, corpus_id: str, top_k: int = 5) -> list[EvidenceSpan]:
        return [
            EvidenceSpan(doc_id="d1", start=0, end=50, text=f"Evidence for {query}"),
        ]


class StubCritiquer:
    def critique(self, query: str, answer: str) -> str:
        return f"Revised: {answer}"


class TestStandardLLMBaseline:
    def test_generates_zero_shot(self):
        baseline = StandardLLMBaseline(generator=StubGenerator())
        result = baseline.run("What is X?")
        assert "zero-shot" in result.generated_text
        assert result.config.baseline_type == BaselineType.STANDARD_LLM
        assert result.final_text == result.generated_text

    def test_no_retrieved_spans(self):
        baseline = StandardLLMBaseline(generator=StubGenerator())
        result = baseline.run("q")
        assert result.retrieved_spans == []


class TestStandardRAGBaseline:
    def test_generates_with_context(self):
        baseline = StandardRAGBaseline(
            generator=StubGenerator(),
            retriever=StubRetriever(),
        )
        result = baseline.run("What is X?")
        assert "with context" in result.generated_text
        assert len(result.retrieved_spans) == 1
        assert result.config.baseline_type == BaselineType.STANDARD_RAG


class TestSelfCritiqueBaseline:
    def test_produces_revised_output(self):
        baseline = SelfCritiqueBaseline(
            generator=StubGenerator(),
            critiquer=StubCritiquer(),
        )
        result = baseline.run("What is X?")
        assert result.generated_text != result.final_text
        assert "Revised" in result.final_text
        assert result.config.baseline_type == BaselineType.SELF_CRITIQUE


class TestBaselineConfigs:
    def test_four_configs(self):
        assert len(BASELINE_CONFIGS) == 9

    def test_all_types_covered(self):
        types = {c.baseline_type for c in BASELINE_CONFIGS}
        assert types == {
            BaselineType.STANDARD_LLM,
            BaselineType.STANDARD_RAG,
            BaselineType.RAG_VERIFIER,
            BaselineType.SELF_CRITIQUE,
            BaselineType.SELF_CHECK_GPT,
            BaselineType.CHAIN_OF_VERIFICATION,
            BaselineType.RANDOM_FILTER,
            BaselineType.FIRST_K_SENTENCES,
            BaselineType.SINGLE_NLI_THRESHOLD,
        }
