"""
Unified adapter for the internal tiny benchmark.

Runs scenarios against the proxy's LLM service and captures tool call results
for evaluation. Uses the full tool-calling loop so we can measure the proxy's
end-to-end orchestration quality.
"""

import json
import time

from app.adapters.base import BenchmarkAdapter
from app.core.openrouter_client import call_openrouter
from app.services.tool_executor import execute_tool_call
from app.tools.registry import TOOL_MAP, TOOL_SCHEMAS
from app.logger_config import logger

MAX_TOOL_ROUNDS = 5


class TinyBenchmarkAdapter(BenchmarkAdapter):
    """
    Adapter that runs a single scenario through the LLM pipeline,
    recording every tool call made during the multi-turn loop.
    """

    async def run(self, config: dict, scenario: dict) -> dict:
        messages = [{"role": "user", "content": scenario["prompt"]}]

        payload = {
            "model": config["model"],
            "temperature": config.get("temperature", 0.0),
            "max_tokens": config.get("max_tokens", 1024),
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
        }

        # Track ALL tool calls across rounds
        all_called_tools: list[str] = []
        all_called_args: dict[str, dict] = {}
        final_content = None
        final_finish_reason = ""

        for round_num in range(MAX_TOOL_ROUNDS):
            payload["messages"] = messages
            response = await call_openrouter(payload)

            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "")

            tool_calls = message.get("tool_calls")

            if not tool_calls or finish_reason != "tool_calls":
                # No more tool calls — this is the final response
                final_content = message.get("content")
                final_finish_reason = finish_reason
                break

            # Record the tool calls
            messages.append(message)

            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                fn_args_str = fn.get("arguments", "{}")

                all_called_tools.append(fn_name)
                try:
                    all_called_args[fn_name] = json.loads(fn_args_str)
                except (json.JSONDecodeError, TypeError):
                    all_called_args[fn_name] = {}

                # Execute the tool
                tc_id = tc.get("id", "")
                if fn_name in TOOL_MAP:
                    tool_msg = await execute_tool_call(tc_id, fn_name, fn_args_str)
                    messages.append(tool_msg)
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps({"error": f"Tool '{fn_name}' not available"}),
                    })

            logger.info(
                f"[TinyBench] Round {round_num + 1}: "
                f"called {[tc.get('function', {}).get('name') for tc in tool_calls]}"
            )

        return {
            "called_tools": all_called_tools,
            "called_args": all_called_args,
            "content": final_content,
            "finish_reason": final_finish_reason,
            "raw_messages": messages,
        }
