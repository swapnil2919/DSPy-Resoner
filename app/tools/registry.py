from app.tools.weather_tool import get_weather
from app.tools.email_tool import generate_email
from app.tools.search_tool import web_search
from app.tools.code_tool import run_python


# Dispatch map: function_name -> callable
TOOL_MAP = {
    "get_weather": get_weather,
    "generate_email": generate_email,
    "web_search": web_search,
    "run_python": run_python,
}

# OpenAI-format function schemas (used in tool definitions)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name to get weather for",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute a Python code snippet on the server and return its output. "
                "Use this for file operations (move, copy, delete, create directories), "
                "math calculations, data processing, or any task that Python can handle. "
                "Always use print() to output results. Example: "
                "'import shutil; shutil.move(\"a.txt\", \"b/a.txt\"); print(\"done\")'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Valid Python source code to execute. Use print() to produce output.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for real-time information, facts, news, or anything not in the model's training data",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 3)",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_email",
            "description": "Generate a formatted email with the given recipient, subject, and optional tone",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Email recipient address",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line",
                    },
                    "tone": {
                        "type": "string",
                        "description": "Tone of the email (e.g. formal, casual)",
                        "default": "formal",
                    },
                },
                "required": ["to", "subject"],
            },
        },
    },
]

# Legacy metadata (kept for /tools endpoint)
TOOL_META = {
    schema["function"]["name"]: {
        "name": schema["function"]["name"],
        "description": schema["function"]["description"],
        "parameters": schema["function"]["parameters"],
    }
    for schema in TOOL_SCHEMAS
}