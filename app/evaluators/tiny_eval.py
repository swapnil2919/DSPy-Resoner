"""
Evaluator for the internal tiny benchmark.

Scores results based on:
- Tool selection accuracy (did it call the right tools?)
- Argument accuracy (did it extract the right arguments?)
- False positive avoidance (did it avoid calling tools when it shouldn't?)
"""


class TinyBenchmarkEvaluator:
    """
    Evaluates benchmark results against expected outcomes.

    Returns a dict with per-dimension scores and an overall score.
    """

    def evaluate(self, result: dict, scenario: dict) -> dict:
        expected_tools = scenario.get("expected_tools", [])
        expected_args = scenario.get("expected_args", {})
        called_tools = result.get("called_tools", [])
        called_args = result.get("called_args", {})

        scores = {}

        # --- Tool Selection Score ---
        if not expected_tools:
            # Negative test: should NOT have called any tools
            scores["tool_selection"] = 1.0 if not called_tools else 0.0
            scores["no_false_positive"] = 1.0 if not called_tools else 0.0
        else:
            # Positive test: should have called the expected tools
            expected_set = sorted(expected_tools)
            called_set = sorted(called_tools)

            if expected_set == called_set:
                scores["tool_selection"] = 1.0
            elif set(expected_tools).issubset(set(called_tools)):
                # Called all expected tools + some extras
                scores["tool_selection"] = 0.5
            elif set(expected_tools).intersection(set(called_tools)):
                # Called at least one expected tool
                scores["tool_selection"] = 0.25
            else:
                scores["tool_selection"] = 0.0

            scores["no_false_positive"] = 1.0  # Not applicable for positive tests

        # --- Argument Accuracy Score ---
        if expected_args and called_args:
            arg_scores = []
            for fn_name, expected_fn_args in expected_args.items():
                actual_fn_args = called_args.get(fn_name, {})
                if not expected_fn_args:
                    arg_scores.append(1.0)
                    continue

                match_count = 0
                total = len(expected_fn_args)
                for key, expected_val in expected_fn_args.items():
                    actual_val = actual_fn_args.get(key, "")
                    if isinstance(expected_val, str) and isinstance(actual_val, str):
                        # Fuzzy string match: check if expected is contained in actual
                        if expected_val.lower() in actual_val.lower() or actual_val.lower() in expected_val.lower():
                            match_count += 1
                    elif expected_val == actual_val:
                        match_count += 1

                arg_scores.append(match_count / total if total > 0 else 1.0)

            scores["argument_accuracy"] = sum(arg_scores) / len(arg_scores) if arg_scores else 1.0
        elif not expected_args:
            scores["argument_accuracy"] = 1.0  # No args to check
        else:
            scores["argument_accuracy"] = 0.0  # Expected args but none provided

        # --- Overall Score ---
        weights = {
            "tool_selection": 0.5,
            "argument_accuracy": 0.3,
            "no_false_positive": 0.2,
        }
        scores["overall"] = sum(
            scores.get(k, 0) * w for k, w in weights.items()
        )

        return scores
