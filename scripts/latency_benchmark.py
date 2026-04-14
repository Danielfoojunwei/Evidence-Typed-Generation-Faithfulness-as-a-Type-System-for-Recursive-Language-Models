"""Latency benchmark for the ETG verification pipeline.

Profiles each stage of the ETG pipeline to identify bottlenecks and
measure practical viability. Addresses the critical finding that
the pipeline takes ~30 minutes on CPU for ~60 claims.

Stages profiled:
    1. Claim extraction (sentence splitting / decomposition)
    2. Per-view verification (NLI, LLM-Judge, QA)
    3. Support mass aggregation
    4. Type checking / confidence grading
    5. Constrained decoding / rendering

Reports per-stage latency, total pipeline latency, and comparison
against the 500ms/token KPI target from the evaluation plan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import NamedTuple


class StageLatency(NamedTuple):
    """Latency measurement for a single pipeline stage."""

    stage_name: str
    total_ms: float
    n_invocations: int
    mean_ms_per_invocation: float
    pct_of_total: float


@dataclass
class PipelineLatencyReport:
    """Full latency report for the ETG pipeline."""

    stages: list[StageLatency] = field(default_factory=list)
    total_pipeline_ms: float = 0.0
    n_claims: int = 0
    n_views: int = 0
    mean_ms_per_claim: float = 0.0
    mean_ms_per_verification: float = 0.0
    meets_kpi_target: bool = False
    kpi_target_ms_per_token: float = 500.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "ETG Pipeline Latency Benchmark",
            "=" * 60,
            f"Total pipeline time: {self.total_pipeline_ms:.1f} ms",
            f"Claims processed: {self.n_claims}",
            f"Views per claim: {self.n_views}",
            f"Mean ms/claim: {self.mean_ms_per_claim:.1f}",
            f"Mean ms/verification: {self.mean_ms_per_verification:.1f}",
            f"KPI target (500ms/token): {'PASS' if self.meets_kpi_target else 'FAIL'}",
            "",
            "Stage breakdown:",
            "-" * 60,
        ]
        for stage in self.stages:
            lines.append(
                f"  {stage.stage_name:<30s} "
                f"{stage.total_ms:>8.1f} ms "
                f"({stage.pct_of_total:>5.1f}%) "
                f"[{stage.n_invocations} calls, "
                f"{stage.mean_ms_per_invocation:.1f} ms/call]"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


class LatencyTimer:
    """Context manager for timing pipeline stages."""

    def __init__(self) -> None:
        self._stages: dict[str, list[float]] = {}
        self._current_stage: str | None = None
        self._start_time: float = 0.0

    def start(self, stage_name: str) -> None:
        self._current_stage = stage_name
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        if self._current_stage is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        if self._current_stage not in self._stages:
            self._stages[self._current_stage] = []
        self._stages[self._current_stage].append(elapsed_ms)
        self._current_stage = None
        return elapsed_ms

    def build_report(
        self,
        n_claims: int,
        n_views: int,
    ) -> PipelineLatencyReport:
        stages: list[StageLatency] = []
        total_ms = 0.0
        for name, timings in self._stages.items():
            total = sum(timings)
            total_ms += total
            stages.append(StageLatency(
                stage_name=name,
                total_ms=total,
                n_invocations=len(timings),
                mean_ms_per_invocation=total / len(timings) if timings else 0.0,
                pct_of_total=0.0,
            ))

        # Fill in percentages
        if total_ms > 0:
            stages = [
                StageLatency(
                    stage_name=s.stage_name,
                    total_ms=s.total_ms,
                    n_invocations=s.n_invocations,
                    mean_ms_per_invocation=s.mean_ms_per_invocation,
                    pct_of_total=(s.total_ms / total_ms) * 100.0,
                )
                for s in stages
            ]

        n_verifications = n_claims * n_views
        return PipelineLatencyReport(
            stages=stages,
            total_pipeline_ms=total_ms,
            n_claims=n_claims,
            n_views=n_views,
            mean_ms_per_claim=total_ms / n_claims if n_claims > 0 else 0.0,
            mean_ms_per_verification=(
                total_ms / n_verifications if n_verifications > 0 else 0.0
            ),
            meets_kpi_target=(
                (total_ms / n_claims) < 500.0 if n_claims > 0 else True
            ),
        )


def benchmark_mock_pipeline(
    n_claims: int = 60,
    n_views: int = 3,
    claim_extraction_ms: float = 5.0,
    verification_ms_per_call: float = 150.0,
    aggregation_ms: float = 0.5,
    type_check_ms: float = 0.1,
    rendering_ms: float = 1.0,
) -> PipelineLatencyReport:
    """Run a mock latency benchmark with configurable per-stage costs.

    Useful for projecting latency under different hardware configurations
    (e.g., CPU vs GPU inference) without requiring actual model loading.

    Args:
        n_claims: number of claims to simulate
        n_views: number of verification views
        claim_extraction_ms: ms per claim for extraction
        verification_ms_per_call: ms per (claim, view) verification
        aggregation_ms: ms per claim for score aggregation
        type_check_ms: ms per claim for type checking
        rendering_ms: ms for final rendering

    Returns:
        PipelineLatencyReport with projected latencies.
    """
    timer = LatencyTimer()

    # Stage 1: Claim extraction
    for _ in range(n_claims):
        timer.start("claim_extraction")
        time.sleep(claim_extraction_ms / 1000.0)
        timer.stop()

    # Stage 2: Verification (the bottleneck)
    for _ in range(n_claims):
        for _ in range(n_views):
            timer.start("verification")
            time.sleep(verification_ms_per_call / 1000.0)
            timer.stop()

    # Stage 3: Aggregation
    for _ in range(n_claims):
        timer.start("aggregation")
        time.sleep(aggregation_ms / 1000.0)
        timer.stop()

    # Stage 4: Type checking
    for _ in range(n_claims):
        timer.start("type_check")
        time.sleep(type_check_ms / 1000.0)
        timer.stop()

    # Stage 5: Rendering
    timer.start("rendering")
    time.sleep(rendering_ms / 1000.0)
    timer.stop()

    return timer.build_report(n_claims, n_views)


if __name__ == "__main__":
    print("Running ETG latency benchmark (mock pipeline)...")
    print()

    # Scenario 1: CPU inference (current situation)
    print("Scenario 1: CPU inference (current)")
    report = benchmark_mock_pipeline(
        n_claims=60,
        n_views=3,
        claim_extraction_ms=5.0,
        verification_ms_per_call=150.0,  # ~150ms per NLI inference on CPU
    )
    print(report.summary())
    print()

    # Scenario 2: GPU inference (target)
    print("Scenario 2: GPU inference (projected)")
    report = benchmark_mock_pipeline(
        n_claims=60,
        n_views=3,
        claim_extraction_ms=2.0,
        verification_ms_per_call=5.0,  # ~5ms per NLI inference on GPU
    )
    print(report.summary())
    print()

    # Scenario 3: Batched GPU inference
    print("Scenario 3: Batched GPU inference (projected)")
    report = benchmark_mock_pipeline(
        n_claims=60,
        n_views=3,
        claim_extraction_ms=1.0,
        verification_ms_per_call=1.0,  # batch of 60 claims in ~60ms total
    )
    print(report.summary())
