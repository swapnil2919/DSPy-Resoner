"""
Benchmark API endpoints.

POST /benchmarks/tiny  — Run the internal micro-benchmark
GET  /benchmarks/scenarios — List available scenarios
GET  /benchmarks/results — List past benchmark runs
GET  /benchmarks/results/{run_id} — Get a specific past run
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.adapters.tiny_benchmark import TinyBenchmarkAdapter
from app.core.config import DEFAULT_MODEL
from app.evaluators.tiny_eval import TinyBenchmarkEvaluator
from app.logger_config import logger
from app.schemas.benchmarks import (
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkRunSummary,
    ScenarioResult,
    ScenarioScore,
)
from app.services.scenario_loader import list_scenarios, list_scenario_ids, load_scenario

router = APIRouter()

adapter = TinyBenchmarkAdapter()
evaluator = TinyBenchmarkEvaluator()

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/tiny", response_model=BenchmarkResponse)
async def run_tiny_benchmark(req: BenchmarkRequest):
    """
    Run the internal tiny benchmark against the proxy's LLM pipeline.

    If scenario_ids is not provided, runs all available scenarios.
    """
    model = req.model or DEFAULT_MODEL
    scenario_ids = req.scenario_ids or list_scenario_ids()
    run_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info(
        f"Benchmark started | run_id={run_id} model={model} "
        f"scenarios={len(scenario_ids)} runs_per_scenario={req.runs_per_scenario}"
    )

    config = {
        "model": model,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }

    results: list[ScenarioResult] = []

    for scenario_id in scenario_ids:
        try:
            scenario = load_scenario(scenario_id)
        except KeyError as e:
            logger.warning(str(e))
            continue

        for run_num in range(req.runs_per_scenario):
            start = time.time()

            try:
                output = await adapter.run(config, scenario)
            except Exception as e:
                logger.error(f"Scenario {scenario_id} run {run_num + 1} failed: {e}", exc_info=True)
                results.append(ScenarioResult(
                    scenario_id=scenario_id,
                    category=scenario.get("category", ""),
                    description=scenario.get("description", ""),
                    scores=ScenarioScore(),
                    latency_ms=round((time.time() - start) * 1000, 2),
                    expected_tools=scenario.get("expected_tools", []),
                    content=f"Error: {e}",
                ))
                continue

            latency_ms = round((time.time() - start) * 1000, 2)
            scores = evaluator.evaluate(output, scenario)

            results.append(ScenarioResult(
                scenario_id=scenario_id,
                category=scenario.get("category", ""),
                description=scenario.get("description", ""),
                scores=ScenarioScore(**scores),
                latency_ms=latency_ms,
                called_tools=output.get("called_tools", []),
                called_args=output.get("called_args", {}),
                expected_tools=scenario.get("expected_tools", []),
                content=output.get("content"),
            ))

            logger.info(
                f"Scenario {scenario_id} run {run_num + 1}: "
                f"overall={scores['overall']:.2f} latency={latency_ms}ms"
            )

    # Aggregate scores
    if results:
        avg_tool = sum(r.scores.tool_selection for r in results) / len(results)
        avg_arg = sum(r.scores.argument_accuracy for r in results) / len(results)
        avg_fp = sum(r.scores.no_false_positive for r in results) / len(results)
        avg_overall = sum(r.scores.overall for r in results) / len(results)
    else:
        avg_tool = avg_arg = avg_fp = avg_overall = 0.0

    passed = sum(1 for r in results if r.scores.overall >= 0.5)
    failed = len(results) - passed

    logger.info(
        f"Benchmark complete | run_id={run_id} passed={passed}/{len(results)} "
        f"overall={avg_overall:.2f}"
    )

    response = BenchmarkResponse(
        run_id=run_id,
        timestamp=timestamp,
        model=model,
        total_scenarios=len(results),
        passed=passed,
        failed=failed,
        aggregate_scores=ScenarioScore(
            tool_selection=round(avg_tool, 4),
            argument_accuracy=round(avg_arg, 4),
            no_false_positive=round(avg_fp, 4),
            overall=round(avg_overall, 4),
        ),
        results=results,
    )

    # Persist to disk
    result_file = RESULTS_DIR / f"{run_id}.json"
    result_file.write_text(response.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"Results saved to {result_file}")

    return response


@router.get("/scenarios")
async def get_scenarios():
    """List all available benchmark scenarios."""
    return {"scenarios": list_scenarios()}


@router.get("/results", response_model=list[BenchmarkRunSummary])
async def list_results():
    """List all past benchmark runs, newest first."""
    summaries = []
    for f in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            summaries.append(BenchmarkRunSummary(
                run_id=data["run_id"],
                timestamp=data["timestamp"],
                model=data["model"],
                total_scenarios=data["total_scenarios"],
                passed=data["passed"],
                failed=data["failed"],
                overall_score=data["aggregate_scores"]["overall"],
            ))
        except (json.JSONDecodeError, KeyError):
            continue
    return summaries


@router.get("/results/{run_id}", response_model=BenchmarkResponse)
async def get_result(run_id: str):
    """Retrieve a specific past benchmark run by run_id."""
    result_file = RESULTS_DIR / f"{run_id}.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return BenchmarkResponse.model_validate_json(result_file.read_text())