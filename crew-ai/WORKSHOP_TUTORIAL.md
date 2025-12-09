# Advanced Docker for Agentic AI Workshop

## Building AI Agent Applications with Docker Compose and Local LLMs

This tutorial demonstrates how to build and run a multi-agent AI application using Docker Compose with Docker Model Runner and CrewAI.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Understanding the Components](#understanding-the-components)
6. [Step-by-Step: Running the Application](#step-by-step-running-the-application)
7. [Deep Dive: Docker Compose Extensions](#deep-dive-docker-compose-extensions)
8. [Deep Dive: Search Tools](#deep-dive-search-tools)
9. [Deep Dive: CrewAI Agents](#deep-dive-crewai-agents)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This project showcases a **marketing content generation system** powered by:

- **CrewAI**: A multi-agent AI framework that orchestrates specialized AI agents
- **Docker Model Runner**: Local LLM inference using Docker Desktop
- **DuckDuckGo Search**: Web search capabilities via the `duckduckgo-search` Python library

The system creates marketing posts by coordinating three AI agents:
1. **Lead Market Analyst** - Researches market trends
2. **Chief Marketing Strategist** - Develops marketing strategy
3. **Creative Content Creator** - Produces the final content

---

## Prerequisites

Before starting, ensure you have:

- [ ] **Docker Desktop 4.40+** with Docker Model Runner enabled
- [ ] **Docker Compose v2.35+** (for advanced Docker Desktop extensions)
- [ ] At least **8GB RAM** available for Docker
- [ ] Internet connection (for web search tools)

### Verify Docker Desktop Setup

```bash
# Check Docker version
docker --version
# Should show: Docker version 28.x.x or higher

# Check Docker Compose version
docker compose version
# Should show: Docker Compose version v2.35.x or higher

# Verify Docker Model Runner is available
docker model list
# Should list available models
```

### Enable Docker Model Runner

1. Open Docker Desktop
2. Go to **Settings** > **Features in development**
3. Enable **Docker Model Runner**
4. Enable **Enable inference with TCP** (for container access)
5. Apply & Restart

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Docker Compose Stack                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │  agents (CrewAI)                                         │   │
│  │  ┌─────────────────┐  ┌─────────────────────────────┐   │   │
│  │  │ DuckDuckGo      │  │ Web Fetch Tool              │   │   │
│  │  │ Search Tool     │  │ (URL content extraction)    │   │   │
│  │  └─────────────────┘  └─────────────────────────────┘   │   │
│  │                                                          │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│                           │ OpenAI-compatible API                │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Docker Model Runner (gemma3:4B)                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. CrewAI agents receive tasks and need to search the web
2. Agents use the built-in DuckDuckGo search tool (Python library)
3. Agents can fetch and parse content from URLs using the fetch tool
4. Agents use Docker Model Runner for LLM inference (gemma3)
5. Agents coordinate to produce marketing content

---

## Project Structure

```
crew-ai/
├── compose.yaml              # Docker Compose configuration
├── Dockerfile                # Container build instructions
├── pyproject.toml            # Python dependencies (Poetry)
├── poetry.lock               # Locked dependencies
├── entrypoint.sh             # Container startup script
└── src/
    └── marketing_posts/
        ├── __init__.py
        ├── main.py           # Application entry point
        ├── crew.py           # CrewAI crew definition
        ├── tools.py          # Tool provider (MCP/CrewAI)
        └── config/
            ├── agents.yaml   # Agent role definitions
            ├── tasks.yaml    # Task definitions
            └── input.yaml    # Input configuration
```

---

## Understanding the Components

### 1. Docker Compose Configuration (`compose.yaml`)

```yaml
services:
  agents:
    build:
      context: .
    # Search tools are built-in using duckduckgo-search Python library
    # No external MCP server needed - simpler and more reliable
    restart: "no"
    models:
      gemma3:
        endpoint_var: MODEL_RUNNER_URL
        model_var: MODEL_RUNNER_MODEL

models:
  gemma3:
    model: ai/gemma3:4B-Q4_0
    context_size: 8192
```

**Key Docker Desktop Extensions:**

| Extension | Purpose |
|-----------|---------|
| `models:` (service-level) | Injects LLM endpoint into container environment |
| `models:` (top-level) | Declares models to pre-pull and configure |
| `endpoint_var` | Environment variable name for the LLM API URL |
| `model_var` | Environment variable name for the model name |
| `context_size` | Maximum context window for the model |

### 2. Built-in Search Tools

The agents use built-in search tools implemented in Python:

- **DuckDuckGo Search Tool**: Uses the `duckduckgo-search` library for web searches
- **Web Fetch Tool**: Extracts text content from URLs

This approach is simpler and more reliable than using external MCP servers, which can experience rate limiting or bot detection issues.

### 3. CrewAI Agent Architecture

CrewAI orchestrates multiple specialized AI agents:

**Agents** (`config/agents.yaml`):
- **Lead Market Analyst**: Conducts market research using web search
- **Chief Marketing Strategist**: Synthesizes research into strategy
- **Creative Content Creator**: Produces final marketing content

**Tasks** (`config/tasks.yaml`):
1. Research task → Market Analyst
2. Project understanding → Strategist
3. Marketing strategy → Strategist
4. Campaign ideation → Content Creator
5. Content creation → Content Creator

### 4. Tool Provider (`tools.py`)

The application provides flexible tool selection:

```python
def get_tools() -> list[BaseTool]:
    """
    Tool selection priority:
    1. MCP_SERVER_URL - Uses MCP server (Tavily or other)
    2. SERPER_API_KEY - Uses Serper for Google search
    3. Default - Uses DuckDuckGo Python library directly (no API key needed)
    """
    if os.getenv("MCP_SERVER_URL"):
        return _get_tools_mcp()
    if os.getenv("SERPER_API_KEY"):
        return _get_tools_crewai()
    return _get_tools_duckduckgo()  # Default: no API key needed
```

By default (no environment variables set), agents use the built-in DuckDuckGo search tool, which requires no API keys or external services.

---

## Step-by-Step: Running the Application

### Step 1: Clone and Navigate

```bash
cd compose-for-agents/crew-ai
```

### Step 2: Review the Configuration

```bash
# View the compose file
cat compose.yaml

# View the agent definitions
cat src/marketing_posts/config/agents.yaml

# View the task definitions
cat src/marketing_posts/config/tasks.yaml
```

### Step 3: Build and Start

```bash
docker compose up --build
```

**What happens:**

1. Docker Compose reads the `compose.yaml`
2. The `models:` section triggers Docker Model Runner to:
   - Pull `ai/gemma3:4B-Q4_0` if not already present
   - Start the model with specified context size
3. The `agents` service starts:
   - Receives `MODEL_RUNNER_URL` and `MODEL_RUNNER_MODEL` environment variables
   - Loads built-in DuckDuckGo search and web fetch tools
   - CrewAI initializes and runs the marketing workflow

### Step 4: Watch the Output

You'll see the agents working through their tasks:

```
agents-1       | # Agent: Lead Market Analyst
agents-1       | ## Task: Conduct market research...
agents-1       | ## Using tool: search
agents-1       | ## Tool Input: {"query": "crewai market trends 2024"}
agents-1       | ## Tool Output: 1. **CrewAI vs AutoGen? - Reddit**
agents-1       |                    URL: https://www.reddit.com/r/AI_Agents/...
...
```

### Step 5: View the Results

The final output includes:
- Market research findings
- Marketing strategy document
- Generated marketing content

### Step 6: Cleanup

```bash
docker compose down
```

---

## Deep Dive: Docker Compose Extensions

Docker Desktop extends Docker Compose with AI-specific features:

### The `models:` Block

**Top-level declaration:**
```yaml
models:
  gemma3:
    model: ai/gemma3:4B-Q4_0
    context_size: 8192
    runtime_flags:
      - --no-prefill-assistant
```

This tells Docker Desktop to:
- Pre-pull the specified model before starting services
- Configure the model with the given context size
- Apply runtime flags to the inference engine

**Service-level binding:**
```yaml
services:
  agents:
    models:
      gemma3:
        endpoint_var: MODEL_RUNNER_URL
        model_var: MODEL_RUNNER_MODEL
```

This injects environment variables into the container:
- `MODEL_RUNNER_URL`: The API endpoint (e.g., `http://model-runner.docker.internal/v1`)
- `MODEL_RUNNER_MODEL`: The model name (e.g., `ai/gemma3:4B-Q4_0`)

### Benefits

1. **No external API keys needed** - LLM runs locally
2. **Automatic model management** - Docker handles pulling and caching
3. **Consistent configuration** - Model settings in one place
4. **Network isolation** - Uses Docker's internal networking

---

## Deep Dive: Search Tools

### Built-in Tools

The project includes two built-in tools that require no external services or API keys:

#### DuckDuckGo Search Tool

```python
class DuckDuckGoSearchTool(BaseTool):
    name: str = "search"
    description: str = "Search the web using DuckDuckGo..."

    def _run(self, query: str) -> str:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=10))
        # Format and return results
```

#### Web Fetch Tool

```python
class WebFetchTool(BaseTool):
    name: str = "fetch"
    description: str = "Fetch and extract content from a web page URL..."

    def _run(self, url: str) -> str:
        # Fetch URL and extract text content
        # Returns cleaned text from the page
```

### Alternative Search Providers

You can use alternative search providers by setting environment variables:

| Provider | Environment Variable | Notes |
|----------|---------------------|-------|
| DuckDuckGo (default) | None needed | Free, no signup required |
| Tavily | `MCP_SERVER_URL=https://mcp.tavily.com/mcp/?tavilyApiKey=YOUR_KEY` | AI-optimized search |
| Serper (Google) | `SERPER_API_KEY=your_key` | Google search results |

### Why Not MCP Gateway?

The original implementation used an MCP server for DuckDuckGo search, but this approach had issues:
- **Bot detection**: DuckDuckGo's bot detection would block requests from containerized environments
- **Rate limiting**: The MCP server was subject to aggressive rate limits
- **Complexity**: Required an additional container (supergateway) to run

The direct Python library approach is simpler and more reliable.

---

## Deep Dive: CrewAI Agents

### Agent Configuration

Each agent is defined in `agents.yaml` with:

```yaml
lead_market_analyst:
  role: Lead Market Analyst
  goal: Conduct research about the customer and competitors
  backstory: >
    As a Lead Market Analyst, you're responsible for
    providing strategic insights...
```

### Task Orchestration

Tasks flow through agents in sequence:

```yaml
research_task:
  description: Research the customer's domain...
  expected_output: A detailed report...
  agent: lead_market_analyst
```

### Tool Integration

Agents automatically receive tools from `tools.py`:

```python
@agent
def lead_market_analyst(self) -> Agent:
    return Agent(
        config=self.agents_config["lead_market_analyst"],
        tools=get_tools(),  # MCP or CrewAI tools
        llm=get_llm(),      # Docker Model Runner or OpenAI
    )
```

---

## Troubleshooting

### Issue: Model Runner Not Available

**Symptom:**
```
Error: Could not connect to MODEL_RUNNER_URL
```

**Solution:**
1. Open Docker Desktop Settings
2. Enable "Docker Model Runner" in Features
3. Enable "Enable inference with TCP"
4. Restart Docker Desktop

### Issue: Search Rate Limiting

**Symptom:**
```
Search rate limit reached. Please wait a moment and try again.
```

**Solution:**
- Wait a few seconds between searches
- Use more specific search queries
- Consider using Tavily or Serper for production workloads (requires API key)

### Issue: Out of Memory

**Symptom:**
```
OOM killed
```

**Solution:**
- Reduce `context_size` in models configuration
- Increase Docker Desktop memory allocation
- Use a smaller model variant

### Issue: "Cannot have 2 or more assistant messages at the end of the list"

**Symptom:**
```
Error code: 400 - {'error': {'code': 400, 'message': 'Cannot have 2 or more assistant messages at the end of the list.'}}
```

**Explanation:**
Docker Model Runner validates message sequences and rejects requests with consecutive assistant messages. This can occur during complex CrewAI agent interactions with tool calls.

**Solution:**
The project includes a message sanitizer (`src/marketing_posts/custom_llm.py`) that automatically merges consecutive assistant messages before sending to the model. This is installed at startup in `main.py`.

### Issue: "Invalid content type" Error with Gemma3

**Symptom:**
```
Error code: 500 - Invalid content type at row 39, column 51
```

**Explanation:**
Some models (like Gemma3) have chat templates that expect message content to be a string, but LiteLLM may send content as arrays (multi-part content format).

**Solution:**
1. Use the `hosted_vllm/` model prefix instead of `openai/` in the Dockerfile
2. Set `function_calling_llm=False` on agents to force ReAct pattern
3. The project's `StringifyingToolWrapper` in `tools.py` ensures tool outputs are always strings

### Issue: Pydantic Validation Errors on Task Output

**Symptom:**
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for TaskOutput
```

**Explanation:**
Smaller local models like Gemma3 may struggle to produce strictly formatted JSON output that matches Pydantic schemas.

**Solution:**
The project has removed `output_json` constraints from tasks to allow free-form text output, which works better with smaller models. If you need structured output, consider using a larger model or post-processing the results

---

## Summary

In this tutorial, you learned:

1. **Docker Compose extensions** for AI applications (`models:` block)
2. **Docker Model Runner** for local LLM inference
3. **MCP Gateway** for providing AI agents access to external tools
4. **CrewAI** for orchestrating multi-agent AI workflows
5. How to **combine these technologies** into a production-ready application

### Key Takeaways

- Docker Desktop provides first-class support for AI workloads
- MCP standardizes how AI applications access tools
- Multi-agent systems can coordinate complex tasks
- Everything runs locally - no external API keys required

---

## Next Steps

1. **Modify the agents** - Edit `agents.yaml` to create your own specialists
2. **Add new tools** - Integrate additional MCP servers
3. **Try different models** - Experiment with other models from Docker Model Runner
4. **Scale up** - Increase context size for more complex tasks

Happy building!
