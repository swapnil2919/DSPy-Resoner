"""
Execute tool calls received in OpenAI tool_call format.
"""

import json
import time

from app.tools.registry import TOOL_MAP
from app.logger_config import logger


async def execute_tool_call(tool_call_id: str, function_name: str, arguments_json: str) -> dict:
    """
    Execute a single tool call and return the result as a tool-role message dict.

    Args:
        tool_call_id: The unique ID from the LLM's tool_call
        function_name: Name of the function to execute
        arguments_json: JSON-encoded string of function arguments

    Returns:
        A dict in OpenAI tool-message format:
        {"role": "tool", "tool_call_id": "...", "content": "..."}
    """
    logger.info(f"Tool execution requested: {function_name} (id={tool_call_id})")

    tool_fn = TOOL_MAP.get(function_name)
    if not tool_fn:
        logger.error(f"Tool not found: {function_name}")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps({"error": f"Tool '{function_name}' not found"}),
        }

    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid arguments JSON for {function_name}: {e}")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps({"error": f"Invalid arguments: {e}"}),
        }

    try:
        start = time.time()
        result = await tool_fn(**args)
        duration = round(time.time() - start, 3)
        logger.info(f"Tool executed successfully: {function_name} ({duration}s)")

        # Ensure result is a string
        content = result if isinstance(result, str) else json.dumps(result)

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    except Exception as e:
        logger.error(f"Error executing tool: {function_name}", exc_info=True)
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps({"error": str(e)}),
        }