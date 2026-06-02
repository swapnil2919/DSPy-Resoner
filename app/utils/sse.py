"""
OpenAI-compatible SSE formatting helpers.
"""

import json


def format_chunk(chunk: dict) -> str:
    """Format a chat.completion.chunk dict as an SSE data line."""
    return f"data: {json.dumps(chunk)}\n\n"


def format_done() -> str:
    """Emit the SSE termination signal."""
    return "data: [DONE]\n\n"