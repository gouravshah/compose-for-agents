# Advanced Docker for Agentic AI Workshop

## Building AI Agent Applications with Docker Compose, MCP, and Local LLMs

This tutorial demonstrates how to build and run a multi-agent AI application using Docker Compose with advanced features like Model Context Protocol (MCP), Docker Model Runner, and CrewAI.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Understanding the Components](#understanding-the-components)
6. [Step-by-Step: Running the Application](#step-by-step-running-the-application)
7. [Deep Dive: Docker Compose Extensions](#deep-dive-docker-compose-extensions)
8. [Deep Dive: MCP Gateway](#deep-dive-mcp-gateway)
9. [Deep Dive: CrewAI Agents](#deep-dive-crewai-agents)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This project showcases a **marketing content generation system** powered by:

- **CrewAI**: A multi-agent AI framework that orchestrates specialized AI agents
- **Docker Model Runner**: Local LLM inference using Docker Desktop
- **MCP (Model Context Protocol)**: A standardized protocol for AI tool access
- **MCP Gateway**: A proxy that provides AI agents access to external tools (like web search)

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
│  ┌──────────────────┐         ┌──────────────────────────────┐  │
│  │                  │   SSE   │                              │  │
│  │  agents          │◄───────►│  mcp-gateway                 │  │
│  │  (CrewAI)        │         │  (supergateway)              │  │
│  │                  │         │                              │  │
│  └────────┬─────────┘         └──────────────────────────────┘  │
│           │                                                      │
│           │ OpenAI-compatible API                                │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  Docker Model    │                                           │
│  │  Runner          │                                           │
│  │  (gemma3:4B)     │                                           │
│  └──────────────────┘                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. CrewAI agents receive tasks and need to search the web
2. Agents call MCP tools via the MCP Gateway (SSE endpoint)
3. MCP Gateway routes requests to duckduckgo-mcp-server
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
    environment:
      - MCP_SERVER_URL=http://mcp-gateway:8000/sse
    restart: "no"
    depends_on:
      mcp-gateway:
        condition: service_healthy
    models:
      gemma3:
        endpoint_var: MODEL_RUNNER_URL
        model_var: MODEL_RUNNER_MODEL

  mcp-gateway:
    image: supercorp/supergateway:uvx
    command:
      - --stdio
      - uvx duckduckgo-mcp-server
      - --port
      - "8000"
      - --host
      - "0.0.0.0"
      - --cors
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:8000/sse"]
      interval: 5s
      timeout: 10s
      retries: 10
      start_period: 30s

models:
  gemma3:
    model: ai/gemma3:4B-Q4_0
    context_size: 8192
    runtime_flags:
      - --no-prefill-assistant
```

**Key Docker Desktop Extensions:**

| Extension | Purpose |
|-----------|---------|
| `models:` (service-level) | Injects LLM endpoint into container environment |
| `models:` (top-level) | Declares models to pre-pull and configure |
| `endpoint_var` | Environment variable name for the LLM API URL |
| `model_var` | Environment variable name for the model name |
| `context_size` | Maximum context window for the model |
| `runtime_flags` | Additional flags for model runner |

### 2. The MCP Gateway

The MCP Gateway provides AI agents access to external tools via a standardized protocol.

**Why Supergateway?**

The `docker/mcp-gateway` image is designed to work with Docker Desktop's internal APIs, which aren't available when running as a standalone container. We use `supercorp/supergateway` as an alternative that:

- Wraps stdio-based MCP servers with SSE transport
- Works as a standard container without special privileges
- Supports any MCP server that uses stdio transport

**How it works:**
```
Agent → SSE Request → Supergateway → stdio → duckduckgo-mcp-server
                                    ←      ←
```

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

The application dynamically selects tools based on configuration:

```python
def get_tools() -> list[BaseTool]:
    if os.getenv("MCP_SERVER_URL"):
        return _get_tools_mcp()    # Use MCP tools
    return _get_tools_crewai()      # Use CrewAI native tools
```

When `MCP_SERVER_URL` is set, agents use tools from the MCP Gateway (duckduckgo search). Otherwise, they fall back to CrewAI's built-in tools (requires API keys).

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
3. The `mcp-gateway` service starts:
   - Runs supergateway with duckduckgo-mcp-server
   - Exposes SSE endpoint on port 8000
   - Healthcheck ensures it's ready before agents start
4. The `agents` service starts:
   - Receives `MODEL_RUNNER_URL` and `MODEL_RUNNER_MODEL` environment variables
   - Receives `MCP_SERVER_URL` pointing to the gateway
   - CrewAI initializes and runs the marketing workflow

### Step 4: Watch the Output

You'll see the agents working through their tasks:

```
agents-1       | Available MCP tools ['search']
agents-1       | # Agent: Lead Market Analyst
agents-1       | ## Task: Conduct market research...
agents-1       | ## Using tool: search
agents-1       | ## Tool Input: {"query": "crewai market trends 2024"}
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

## Deep Dive: MCP Gateway

### What is MCP?

**Model Context Protocol (MCP)** is a standardized protocol that allows AI applications to access external tools and data sources. It defines:

- **Tools**: Functions the AI can call (search, database queries, etc.)
- **Resources**: Data sources the AI can read
- **Transport**: How messages are sent (stdio, SSE, etc.)

### MCP in This Project

```
┌─────────────┐     SSE      ┌──────────────────┐    stdio    ┌─────────────────┐
│   CrewAI    │◄────────────►│   Supergateway   │◄───────────►│  duckduckgo-    │
│   Agents    │              │   (SSE proxy)    │             │  mcp-server     │
└─────────────┘              └──────────────────┘             └─────────────────┘
```

### Why SSE Transport?

- **Server-Sent Events (SSE)** allows long-lived HTTP connections
- AI agents can receive streaming responses from tools
- Works through standard HTTP infrastructure (proxies, load balancers)

### Adding More MCP Servers

To add additional MCP tools, you can run multiple supergateway instances or use MCP servers that support SSE natively:

```yaml
services:
  another-mcp-server:
    image: supercorp/supergateway:uvx
    command:
      - --stdio
      - uvx another-mcp-server
      - --port
      - "8001"
```

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

### Issue: MCP Gateway Health Check Fails

**Symptom:**
```
mcp-gateway-1  | unhealthy
```

**Solution:**
- Wait longer - the first startup may take time to download dependencies
- Check if port 8000 is available
- View logs: `docker compose logs mcp-gateway`

### Issue: Agents Can't Find Tools

**Symptom:**
```
Available MCP tools []
```

**Solution:**
1. Verify MCP Gateway is healthy: `docker compose ps`
2. Check the SSE endpoint: `curl http://localhost:8000/sse`
3. Ensure `MCP_SERVER_URL` environment variable is set correctly

### Issue: Out of Memory

**Symptom:**
```
OOM killed
```

**Solution:**
- Reduce `context_size` in models configuration
- Increase Docker Desktop memory allocation
- Use a smaller model variant

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
