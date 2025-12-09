import json
import os
from typing import Any, Optional, Type

from crewai.tools import BaseTool
from crewai_tools import MCPServerAdapter, ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, Field


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


class DuckDuckGoSearchInput(BaseModel):
    """Input schema for DuckDuckGo search tool."""

    query: str = Field(..., description="The search query to look up")


class DuckDuckGoSearchTool(BaseTool):
    """
    A search tool using the duckduckgo-search Python library directly.
    This avoids the MCP server's rate limiting issues by using the library
    with proper request handling.
    """

    name: str = "search"
    description: str = (
        "Search the web using DuckDuckGo. Useful for finding current information "
        "about companies, products, news, and general knowledge."
    )
    args_schema: Type[BaseModel] = DuckDuckGoSearchInput

    def _run(self, query: str, **kwargs: Any) -> str:
        """Execute a DuckDuckGo search and return results."""
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                # Use region="wt-wt" for worldwide English results
                # or "us-en" for US English specifically
                results = list(ddgs.text(query, region="wt-wt", max_results=10))

            if not results:
                return (
                    "No results found for the search query. "
                    "Try rephrasing your search with different keywords."
                )

            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. **{result.get('title', 'No title')}**\n"
                    f"   URL: {result.get('href', 'No URL')}\n"
                    f"   {result.get('body', 'No description')}\n"
                )

            return "\n".join(formatted_results)

        except Exception as e:
            error_msg = str(e).lower()
            if "ratelimit" in error_msg or "rate" in error_msg:
                return (
                    "Search rate limit reached. Please wait a moment and try again "
                    "with a more specific query."
                )
            return f"Search failed: {str(e)}. Try rephrasing your query."


class WebFetchInput(BaseModel):
    """Input schema for web fetch tool."""

    url: str = Field(..., description="The URL to fetch content from")


class WebFetchTool(BaseTool):
    """
    A tool to fetch and extract content from web pages.
    """

    name: str = "fetch"
    description: str = (
        "Fetch and extract the main content from a web page URL. "
        "Useful for reading articles, documentation, or any web page content."
    )
    args_schema: Type[BaseModel] = WebFetchInput

    def _run(self, url: str, **kwargs: Any) -> str:
        """Fetch content from a URL and extract main text."""
        try:
            import urllib.request
            import urllib.error
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text_parts = []
                    self.skip_tags = {"script", "style", "head", "meta", "link"}
                    self.current_tag = None

                def handle_starttag(self, tag, attrs):
                    self.current_tag = tag

                def handle_endtag(self, tag):
                    self.current_tag = None

                def handle_data(self, data):
                    if self.current_tag not in self.skip_tags:
                        text = data.strip()
                        if text:
                            self.text_parts.append(text)

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/91.0.4472.124 Safari/537.36"
                )
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                html = response.read().decode("utf-8", errors="ignore")

            parser = TextExtractor()
            parser.feed(html)
            content = " ".join(parser.text_parts)

            # Truncate if too long
            if len(content) > 8000:
                content = content[:8000] + "... [truncated]"

            return content if content else "Could not extract content from the page."

        except urllib.error.HTTPError as e:
            return f"HTTP error fetching URL: {e.code} {e.reason}"
        except urllib.error.URLError as e:
            return f"URL error: {str(e.reason)}"
        except Exception as e:
            return f"Failed to fetch content: {str(e)}"


def get_tools() -> list[BaseTool]:
    """
    Returns a list of tools available for the marketing posts crew.

    Tool selection priority:
    1. MCP_SERVER_URL - Uses MCP server (Tavily or other)
    2. SERPER_API_KEY - Uses Serper for Google search
    3. Default - Uses DuckDuckGo Python library directly (no API key needed)
    """
    if os.getenv("MCP_SERVER_URL"):
        return _get_tools_mcp()
    if os.getenv("SERPER_API_KEY"):
        return _get_tools_crewai()
    # Default: Use DuckDuckGo Python library directly (more reliable than MCP)
    return _get_tools_duckduckgo()


def _get_tools_crewai() -> list[BaseTool]:
    return [SerperDevTool(), ScrapeWebsiteTool()]


def _get_tools_duckduckgo() -> list[BaseTool]:
    """Return tools using the DuckDuckGo Python library directly."""
    return [DuckDuckGoSearchTool(), WebFetchTool()]


_server: Optional[MCPServerAdapter] = None


def _get_tools_mcp() -> list[BaseTool]:
    global _server
    if _server is None:
        _server = MCPServerAdapter(dict(url=os.getenv("MCP_SERVER_URL")))
        print(f"Available MCP tools {[tool.name for tool in _server.tools]}")
    # Wrap all tools to ensure string output
    return [StringifyingToolWrapper(tool) for tool in _server.tools]
