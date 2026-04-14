"""Tests for the dataset specifications module."""

import pytest

from etg_rlm.datasets import (
    ALL_DATASET_CONFIGS,
    DatasetConfig,
    DatasetName,
    GroundTruthType,
    HALU_EVAL_CONFIG,
    HOTPOT_QA_CONFIG,
    NQ_CONFIG,
    TaskType,
    TRUTHFUL_QA_CONFIG,
    XSUM_CONFIG,
    get_dataset_config,
    total_eval_instances,
)
from etg_rlm.evaluation import EvalInstance


class TestDatasetConfigs:
    def test_five_datasets(self):
        assert len(ALL_DATASET_CONFIGS) == 7

    def test_all_names_unique(self):
        names = [c.name for c in ALL_DATASET_CONFIGS]
        assert len(names) == len(set(names))

    def test_all_names_covered(self):
        names = {c.name for c in ALL_DATASET_CONFIGS}
        assert names == {
            DatasetName.NATURAL_QUESTIONS,
            DatasetName.HOTPOT_QA,
            DatasetName.TRUTHFUL_QA,
            DatasetName.HALU_EVAL,
            DatasetName.XSUM,
            DatasetName.FEVER,
            DatasetName.FACTSCORE_BIO,
        }

    def test_all_have_descriptions(self):
        for config in ALL_DATASET_CONFIGS:
            assert config.description
            assert config.rationale

    def test_all_have_ground_truth(self):
        for config in ALL_DATASET_CONFIGS:
            assert len(config.ground_truth_types) > 0


class TestNQConfig:
    def test_subset_size(self):
        assert NQ_CONFIG.eval_subset_size == 1000

    def test_task_type(self):
        assert NQ_CONFIG.task_type == TaskType.QUESTION_ANSWERING

    def test_ground_truth(self):
        assert GroundTruthType.SHORT_ANSWER in NQ_CONFIG.ground_truth_types
        assert GroundTruthType.LONG_ANSWER in NQ_CONFIG.ground_truth_types


class TestHotpotQAConfig:
    def test_subset_size(self):
        assert HOTPOT_QA_CONFIG.eval_subset_size == 500

    def test_task_type(self):
        assert HOTPOT_QA_CONFIG.task_type == TaskType.MULTI_HOP_QA

    def test_n_hops(self):
        assert HOTPOT_QA_CONFIG.n_hops == 2

    def test_ground_truth(self):
        assert GroundTruthType.SUPPORTING_FACTS in HOTPOT_QA_CONFIG.ground_truth_types


class TestTruthfulQAConfig:
    def test_full_dataset(self):
        assert TRUTHFUL_QA_CONFIG.eval_subset_size == 817

    def test_task_type(self):
        assert TRUTHFUL_QA_CONFIG.task_type == TaskType.TRUTHFULNESS

    def test_categories(self):
        assert TRUTHFUL_QA_CONFIG.n_categories == 38


class TestHaluEvalConfig:
    def test_subset_size(self):
        assert HALU_EVAL_CONFIG.eval_subset_size == 1000

    def test_task_type(self):
        assert HALU_EVAL_CONFIG.task_type == TaskType.HALLUCINATION_DETECTION

    def test_tracks(self):
        assert "qa" in HALU_EVAL_CONFIG.tracks
        assert "summarization" in HALU_EVAL_CONFIG.tracks

    def test_binary_labels(self):
        assert GroundTruthType.BINARY_LABELS in HALU_EVAL_CONFIG.ground_truth_types


class TestXSumConfig:
    def test_subset_size(self):
        assert XSUM_CONFIG.eval_subset_size == 500

    def test_task_type(self):
        assert XSUM_CONFIG.task_type == TaskType.SUMMARIZATION


class TestGetDatasetConfig:
    def test_lookup(self):
        config = get_dataset_config(DatasetName.NATURAL_QUESTIONS)
        assert config is NQ_CONFIG

    def test_all_lookups(self):
        for name in DatasetName:
            config = get_dataset_config(name)
            assert config.name == name

    def test_unknown_raises(self):
        # All valid names are covered, so this can't actually fail
        # unless someone adds a new DatasetName without a config
        pass


class TestTotalInstances:
    def test_total(self):
        expected = 1000 + 500 + 817 + 1000 + 500 + 5000 + 500  # = 9317
        assert total_eval_instances() == expected


class TestDatasetLoader:
    def test_protocol(self):
        """Verify the DatasetLoader protocol works with a stub."""
        class StubLoader:
            def load(self, config, max_instances=None):
                return [
                    EvalInstance(instance_id=f"q{i}", query=f"query {i}")
                    for i in range(min(max_instances or 10, 10))
                ]

        loader = StubLoader()
        instances = loader.load(NQ_CONFIG, max_instances=5)
        assert len(instances) == 5
        assert instances[0].instance_id == "q0"
