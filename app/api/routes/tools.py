from fastapi import APIRouter
from app.tools.registry import TOOL_META

router = APIRouter()

@router.get("")
async def list_tools():
    return {
        "count": len(TOOL_META),
        "tools": list(TOOL_META.values())
    }