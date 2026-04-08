# 🔬 Multi-Agent Research Assistant

A multi-agent AI system that automates web research, fact-checking, and report generation — powered entirely by **local LLMs** via [Ollama](https://ollama.com/) and orchestrated with [LangGraph](https://github.com/langchain-ai/langgraph).

> **Zero LLM API costs** — all language models run locally on your hardware.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│   SUPERVISOR     │ ← qwen2.5:7b-instruct (routes tasks)
│   (Orchestrator) │
└────────┬────────┘
         │ (conditional edges)
    ┌────┼──────────────────┐
    │    │                  │
    ▼    ▼                  ▼
┌────────────┐ ┌──────────────┐ ┌──────────────┐
│ RESEARCHER │ │   VERIFIER   │ │ SYNTHESIZER  │
│ mistral:7b │ │  mistral:7b  │ │  mistral:7b  │
│            │ │              │ │              │
│ Web Search │ │ Fact-check   │ │ Markdown     │
│ → Notes    │ │ Claims       │ │ Report       │
└────────────┘ └──────────────┘ └──────────────┘
    │                │                  │
    └────────────────┴──────────────────┘
                     │
                     ▼
              ┌─────────────┐
              │ FINAL REPORT│
              │  (Markdown) │
              └─────────────┘
```

The LangGraph workflow is **cyclic** — the Supervisor can loop agents back for additional research or verification (up to a configurable maximum).

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Orchestration | LangGraph ≥ 0.2.0 |
| LLM Server | Ollama (local) |
| Supervisor Model | `qwen2.5:7b-instruct` |
| Sub-agent Model | `mistral:7b-instruct` |
| Web Search | Tavily API (free tier — 1,000 req/month) |
| Python | 3.11+ |
| LLM Integration | `langchain-ollama` (`ChatOllama`) |
| Configuration | `pydantic-settings` |
| CLI Output | `rich` |

---

## 📋 Prerequisites

1. **Python 3.11+**
2. **Ollama** — installed and running (`ollama serve`)
3. **Tavily API key** — [get a free key](https://tavily.com/) (1,000 requests/month)
4. **NVIDIA GPU** (recommended) — 8 GB+ VRAM for 7B models

---

## 🚀 Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd multi-agent-research-assistant

# 2. Pull the required Ollama models
bash scripts/setup_ollama.sh

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Edit .env and set TAVILY_API_KEY=tvly-...

# 5. Run a research query
python main.py --query "What are the latest advances in quantum computing?"
```

---

## 💡 Usage

```bash
# Basic usage
python main.py --query "What is quantum computing?"

# Verbose mode with custom iterations
python main.py --query "Climate change effects on agriculture" --verbose --max-iterations 5

# Save report to a specific file
python main.py --query "AI in healthcare" --output report.md
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--query, -q` | Research question (required) | — |
| `--verbose, -v` | Detailed agent logs | `false` |
| `--max-iterations, -m` | Max supervisor loops | `3` |
| `--output, -o` | Output report file path | auto-generated |

---

## 📁 Project Structure

```
multi-agent-research-assistant/
├── src/
│   ├── agents/          # Agent node functions
│   │   ├── researcher.py    # Web search & claim extraction
│   │   ├── verifier.py      # Cross-reference fact-checking
│   │   ├── synthesizer.py   # Markdown report generation
│   │   └── supervisor.py    # Routing & orchestration
│   ├── graph/           # LangGraph definitions
│   │   ├── state.py         # ResearchState TypedDict
│   │   └── workflow.py      # StateGraph construction
│   ├── tools/           # External tool wrappers
│   │   └── search.py        # Tavily search wrapper
│   ├── prompts/         # System prompt templates
│   │   └── templates.py
│   └── config/          # Settings & configuration
│       └── settings.py      # pydantic-settings
├── tests/               # Unit & smoke tests
├── scripts/             # Setup scripts (bash + PowerShell)
├── notebooks/           # Exploration notebooks
├── main.py              # CLI entry point
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## 🔄 How It Works

1. **User** submits a research query via CLI
2. **Supervisor** (qwen2.5) analyses the state and routes to the appropriate agent
3. **Researcher** (mistral:7b) searches the web via Tavily and extracts key claims
4. **Supervisor** routes claims to the Verifier
5. **Verifier** (mistral:7b) cross-references claims against multiple sources
6. **Supervisor** routes verified data to the Synthesizer
7. **Synthesizer** (mistral:7b) generates a polished Markdown report with confidence indicators (✅ ⚠️ ❌)
8. **Supervisor** may loop back for more research or declare FINISH
9. Final report is displayed in terminal and saved to disk

---

## ⚙️ Configuration

All settings are loaded from `.env` via `pydantic-settings`:

| Variable | Description | Default |
|----------|-------------|---------|
| `TAVILY_API_KEY` | Tavily search API key | (required) |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `SUPERVISOR_MODEL` | Supervisor LLM model | `qwen2.5:7b-instruct` |
| `AGENT_MODEL` | Sub-agent LLM model | `mistral:7b-instruct` |

---

## 🧪 Testing

```bash
# Run all tests (graph & state tests work without Ollama)
python -m pytest tests/ -v

# Run only graph compilation tests
python -m pytest tests/test_graph.py -v

# Run agent smoke tests (requires Ollama)
SKIP_OLLAMA_TESTS=false python -m pytest tests/test_agents.py -v
```

---

## 🛠️ Built With

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green?logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?logo=ollama&logoColor=white)
![Tavily](https://img.shields.io/badge/Tavily-Search_API-orange)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-red?logo=pydantic&logoColor=white)

---

## 📄 License

MIT
