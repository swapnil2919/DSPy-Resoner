"""
Load benchmark scenarios from the JSON file.
"""

import json
from pathlib import Path

SCENARIOS_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "scenarios.json"

_cache: dict | None = None


def _load_all() -> dict:
    """Load and cache all scenarios from the JSON file."""
    global _cache
    if _cache is None:
        with open(SCENARIOS_PATH) as f:
            data = json.load(f)
        _cache = {s["id"]: s for s in data["scenarios"]}
    return _cache


def load_scenario(scenario_id: str) -> dict:
    """Load a single scenario by ID."""
    scenarios = _load_all()
    if scenario_id not in scenarios:
        raise KeyError(f"Scenario '{scenario_id}' not found. Available: {list(scenarios.keys())}")
    return scenarios[scenario_id]


def list_scenarios() -> list[dict]:
    """Return all available scenarios as a list."""
    return list(_load_all().values())


def list_scenario_ids() -> list[str]:
    """Return all available scenario IDs."""
    return list(_load_all().keys())