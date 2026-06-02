"""
Python code execution tool.

Lets the ReAct agent write and run arbitrary Python code to solve tasks that
don't map to a specific pre-built tool (file ops, math, data processing, etc.).
Runs in a subprocess with a hard timeout so a bad script cannot hang the server.

WARNING: Do not expose this to untrusted users — it executes real Python code
on the server. Safe for local dev and controlled environments.
"""

import asyncio
import subprocess
import sys

from ..logger_config import logger

_TIMEOUT = 15  # seconds before the subprocess is killed


async def run_python(code: str) -> dict:
    """
    Execute a Python code snippet and return its output.

    Parameters
    ----------
    code : str
        Valid Python source code. Use print() to produce output.
        Standard library is available (os, shutil, pathlib, json, etc.).
        Third-party packages installed in the current venv are also available.

    Returns
    -------
    dict with keys:
        stdout     — anything printed to stdout
        stderr     — any error / traceback text
        returncode — 0 means success, non-zero means the script crashed
    """
    logger.info(f"run_python called | code preview: {code[:120]!r}")

    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                sys.executable, "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=5,  # time to spawn
        )
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(),
            timeout=_TIMEOUT,
        )

        result = {
            "stdout": stdout_b.decode(errors="replace").strip(),
            "stderr": stderr_b.decode(errors="replace").strip(),
            "returncode": proc.returncode,
        }
        logger.info(f"run_python finished | rc={proc.returncode}")
        return result

    except asyncio.TimeoutError:
        logger.warning("run_python timed out")
        return {"error": f"Code execution timed out after {_TIMEOUT}s", "returncode": -1}
    except Exception as exc:
        logger.error("run_python unexpected error", exc_info=True)
        return {"error": str(exc), "returncode": -1}
