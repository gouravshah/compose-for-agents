import json
import os
from typing import Any, Optional, Type

from crewai.tools import BaseTool
from crewai_tools import MCPServerAdapter, ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel


class StringifyingToolWrapper(BaseTool):
    """Wrapper that ensures tool results are always strings to fix Gemma3 template issues."""

    name: str = ""
    description: str = ""
    wrapped_tool: Any = None
    args_schema: Optional[Type[BaseModel]] = None

    def __init__(self, tool: BaseTool):
        # Initialize with proper values
        super().__init__(
            name=tool.name,
            description=tool.description,
            wrapped_tool=tool,
            args_schema=getattr(tool, "args_schema", None),
        )

    def _run(self, **kwargs: Any) -> str:
        """Run the wrapped tool and ensure the result is a string."""
        result = self.wrapped_tool._run(**kwargs)
        if isinstance(result, str):
            return result
        # Convert non-string results to JSON string
        return json.dumps(result, indent=2, default=str)


def get_tools() -> list[BaseTool]:
    """
    Returns a list of tools available for the marketing posts crew.
    """
    if os.getenv("MCP_SERVER_URL"):
        return _get_tools_mcp()
    return _get_tools_crewai()


def _get_tools_crewai() -> list[BaseTool]:
    return [SerperDevTool(), ScrapeWebsiteTool()]


_server: Optional[MCPServerAdapter] = None


def _get_tools_mcp() -> list[BaseTool]:
    global _server
    if _server is None:
        _server = MCPServerAdapter(dict(url=os.getenv("MCP_SERVER_URL")))
        print(f"Available MCP tools {[tool.name for tool in _server.tools]}")
    # Wrap all tools to ensure string output
    return [StringifyingToolWrapper(tool) for tool in _server.tools]
