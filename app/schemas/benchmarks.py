"""
Pydantic models for the benchmark API.
"""

from pydantic import BaseModel, Field
from typing import Any


class BenchmarkRequest(BaseModel):
    """Request body for POST /benchmarks/tiny"""
    model: str | None = None
    scenario_ids: list[str] | None = None  # None = run all scenarios
    temperature: float = 0.0
    max_tokens: int = 1024
    runs_per_scenario: int = 1


class ScenarioScore(BaseModel):
    """Per-scenario scoring breakdown."""
    tool_selection: float = 0.0
    argument_accuracy: float = 0.0
    no_false_positive: float = 0.0
    overall: float = 0.0


class ScenarioResult(BaseModel):
    """Result for a single scenario run."""
    scenario_id: str
    category: str = ""
    description: str = ""
    scores: ScenarioScore
    latency_ms: float
    called_tools: list[str] = Field(default_factory=list)
    called_args: dict[str, Any] = Field(default_factory=dict)
    expected_tools: list[str] = Field(default_factory=list)
    content: str | None = None


class BenchmarkResponse(BaseModel):
    """Full benchmark response with aggregate metrics."""
    run_id: str
    timestamp: str
    model: str
    total_scenarios: int
    passed: int  # scenarios with overall >= 0.5
    failed: int
    aggregate_scores: ScenarioScore
    results: list[ScenarioResult]


class BenchmarkRunSummary(BaseModel):
    """Summary of a past benchmark run (for listing)."""
    run_id: str
    timestamp: str
    model: str
    total_scenarios: int
    passed: int
    failed: int
    overall_score: float