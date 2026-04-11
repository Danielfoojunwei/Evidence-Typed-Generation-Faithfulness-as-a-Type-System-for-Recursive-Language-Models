"""Dataset specifications for ETG evaluation (Section 1 of experimental design).

Defines the evaluation datasets and their configurations:
    1. Natural Questions (NQ) -- factual extraction from long documents
    2. HotpotQA -- multi-hop reasoning across multiple documents
    3. TruthfulQA -- resistance to plausible-sounding misconceptions
    4. HaluEval -- hallucination detection with adversarial examples
    5. XSum -- faithful abstractive summarization
    6. FEVER -- large-scale fact verification (cross-dataset validation)
    7. FactScore Biographies -- fine-grained factuality on long-form text

Datasets 6-7 were added to address the critical limitation that all original
results were reported on TruthfulQA only. Cross-dataset evaluation is essential
to validate that ETG generalizes beyond its development benchmark.

Each dataset is defined as a configuration that includes the dataset name,
evaluation subset size, task type, and ground truth specification. A DatasetLoader
protocol enables plugging in concrete data sources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from etg_rlm.evaluation import EvalInstance


class DatasetName(Enum):
    """Evaluation datasets from the experimental design."""

    NATURAL_QUESTIONS = "natural_questions"
    HOTPOT_QA = "hotpot_qa"
    TRUTHFUL_QA = "truthful_qa"
    HALU_EVAL = "halu_eval"
    XSUM = "xsum"
    # Cross-dataset generalization benchmarks (added to address
    # single-dataset limitation identified in critical review)
    FEVER = "fever"
    FACTSCORE_BIO = "factscore_bio"


class TaskType(Enum):
    """The type of task for each dataset."""

    QUESTION_ANSWERING = "question_answering"
    MULTI_HOP_QA = "multi_hop_qa"
    TRUTHFULNESS = "truthfulness"
    HALLUCINATION_DETECTION = "hallucination_detection"
    SUMMARIZATION = "summarization"
    FACT_VERIFICATION = "fact_verification"
    BIOGRAPHY_FACTUALITY = "biography_factuality"


class GroundTruthType(Enum):
    """The kind of ground truth available in each dataset."""

    SHORT_ANSWER = "short_answer"
    LONG_ANSWER = "long_answer"
    SUPPORTING_FACTS = "supporting_facts"
    TRUE_FALSE_REFERENCES = "true_false_references"
    BINARY_LABELS = "binary_labels"
    REFERENCE_SUMMARY = "reference_summary"


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a single evaluation dataset.

    Attributes:
        name: which dataset
        task_type: the task type
        eval_subset_size: number of instances to evaluate
        ground_truth_types: kinds of ground truth available
        description: human-readable description
        rationale: why this dataset is included in the evaluation
        n_hops: for multi-hop QA, the number of hops (None otherwise)
        n_categories: for categorized datasets (e.g. TruthfulQA)
        tracks: for multi-track datasets (e.g. HaluEval)
    """

    name: DatasetName
    task_type: TaskType
    eval_subset_size: int
    ground_truth_types: tuple[GroundTruthType, ...]
    description: str
    rationale: str
    n_hops: int | None = None
    n_categories: int | None = None
    tracks: tuple[str, ...] = ()


@runtime_checkable
class DatasetLoader(Protocol):
    """Protocol for loading evaluation instances from a dataset."""

    def load(
        self, config: DatasetConfig, max_instances: int | None = None
    ) -> list[EvalInstance]:
        """Load evaluation instances from the dataset.

        Args:
            config: the dataset configuration
            max_instances: optional cap on number of instances

        Returns:
            List of EvalInstance objects ready for evaluation.
        """
        ...


# ---------------------------------------------------------------------------
# Pre-defined dataset configurations (Section 1 of experimental design)
# ---------------------------------------------------------------------------

NQ_CONFIG = DatasetConfig(
    name=DatasetName.NATURAL_QUESTIONS,
    task_type=TaskType.QUESTION_ANSWERING,
    eval_subset_size=1000,
    ground_truth_types=(GroundTruthType.SHORT_ANSWER, GroundTruthType.LONG_ANSWER),
    description=(
        "Large-scale QA dataset of real user queries from Google Search "
        "paired with Wikipedia articles containing the answer."
    ),
    rationale=(
        "Tests the system's ability to extract specific factual information "
        "from long documents and avoid hallucinating details not present "
        "in the source."
    ),
)

HOTPOT_QA_CONFIG = DatasetConfig(
    name=DatasetName.HOTPOT_QA,
    task_type=TaskType.MULTI_HOP_QA,
    eval_subset_size=500,
    ground_truth_types=(GroundTruthType.SUPPORTING_FACTS, GroundTruthType.SHORT_ANSWER),
    description=(
        "Multi-hop QA dataset requiring reasoning across multiple documents."
    ),
    rationale=(
        "Tests the ESBG's ability to construct dependency relationships "
        "between claims (edges in the graph) when the answer requires "
        "synthesizing information from multiple sources."
    ),
    n_hops=2,
)

TRUTHFUL_QA_CONFIG = DatasetConfig(
    name=DatasetName.TRUTHFUL_QA,
    task_type=TaskType.TRUTHFULNESS,
    eval_subset_size=817,
    ground_truth_types=(GroundTruthType.TRUE_FALSE_REFERENCES,),
    description=(
        "Dataset designed to test whether models generate truthful answers "
        "to questions where humans might have false beliefs or misconceptions."
    ),
    rationale=(
        "Tests the system's resistance to generating plausible-sounding "
        "but factually incorrect information, a core strength of ETG."
    ),
    n_categories=38,
)

HALU_EVAL_CONFIG = DatasetConfig(
    name=DatasetName.HALU_EVAL,
    task_type=TaskType.HALLUCINATION_DETECTION,
    eval_subset_size=1000,
    ground_truth_types=(GroundTruthType.BINARY_LABELS,),
    description=(
        "Benchmark for hallucination detection in LLM-generated text, "
        "including knowledge-grounded dialogue, QA, and summarization."
    ),
    rationale=(
        "Provides a direct measure of hallucination rates with carefully "
        "constructed adversarial examples."
    ),
    tracks=("qa", "summarization"),
)

XSUM_CONFIG = DatasetConfig(
    name=DatasetName.XSUM,
    task_type=TaskType.SUMMARIZATION,
    eval_subset_size=500,
    ground_truth_types=(GroundTruthType.REFERENCE_SUMMARY,),
    description=(
        "Extreme summarization dataset: generate a one-sentence summary "
        "of a news article."
    ),
    rationale=(
        "Tests the system's ability to compress information faithfully "
        "without introducing unsupported claims, a common failure mode "
        "in abstractive summarization."
    ),
)

FEVER_CONFIG = DatasetConfig(
    name=DatasetName.FEVER,
    task_type=TaskType.FACT_VERIFICATION,
    eval_subset_size=5000,
    ground_truth_types=(GroundTruthType.BINARY_LABELS,),
    description=(
        "Large-scale fact verification dataset with 185K claims extracted "
        "from Wikipedia. Each claim is labeled as SUPPORTS, REFUTES, or "
        "NOT ENOUGH INFO, with annotated evidence sentences."
    ),
    rationale=(
        "Critical for cross-dataset generalization: tests whether ETG's "
        "verification pipeline transfers to structured fact-checking claims "
        "outside TruthfulQA. FEVER's scale (185K claims) also enables "
        "statistically meaningful evaluation with tight confidence intervals."
    ),
)

FACTSCORE_BIO_CONFIG = DatasetConfig(
    name=DatasetName.FACTSCORE_BIO,
    task_type=TaskType.BIOGRAPHY_FACTUALITY,
    eval_subset_size=500,
    ground_truth_types=(GroundTruthType.BINARY_LABELS,),
    description=(
        "FactScore biography benchmark: long-form biographies generated "
        "by LLMs, decomposed into atomic facts, and scored against "
        "Wikipedia for factual precision."
    ),
    rationale=(
        "Tests ETG on long-form generation where claim decomposition "
        "quality matters most. The original FactScore benchmark uses "
        "fine-grained atomic fact evaluation, directly testing ETG's "
        "claim-level filtering on realistic LLM outputs."
    ),
)

ALL_DATASET_CONFIGS = [
    NQ_CONFIG,
    HOTPOT_QA_CONFIG,
    TRUTHFUL_QA_CONFIG,
    HALU_EVAL_CONFIG,
    XSUM_CONFIG,
    FEVER_CONFIG,
    FACTSCORE_BIO_CONFIG,
]

# Datasets used for cross-dataset generalization validation
CROSS_VALIDATION_CONFIGS = [
    FEVER_CONFIG,
    HALU_EVAL_CONFIG,
    FACTSCORE_BIO_CONFIG,
]


def get_dataset_config(name: DatasetName) -> DatasetConfig:
    """Look up a dataset configuration by name."""
    for config in ALL_DATASET_CONFIGS:
        if config.name == name:
            return config
    raise ValueError(f"Unknown dataset: {name}")


def total_eval_instances() -> int:
    """Total number of evaluation instances across all datasets."""
    return sum(c.eval_subset_size for c in ALL_DATASET_CONFIGS)
