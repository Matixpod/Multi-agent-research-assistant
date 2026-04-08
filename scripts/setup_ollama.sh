#!/usr/bin/env bash
# scripts/setup_ollama.sh
# Setup script for Ollama models required by Multi-Agent Research Assistant.

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SUPERVISOR_MODEL="qwen2.5:7b-instruct"
AGENT_MODEL="mistral:7b-instruct"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Multi-Agent Research Assistant Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# -------------------------------------------------------------------
# 1. Check if Ollama is installed
# -------------------------------------------------------------------
echo -e "${YELLOW}[1/4] Checking Ollama installation...${NC}"
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}ERROR: Ollama is not installed.${NC}"
    echo ""
    echo "Install Ollama:"
    echo "  Linux:   curl -fsSL https://ollama.com/install.sh | sh"
    echo "  macOS:   brew install ollama"
    echo "  Windows: Download from https://ollama.com/download"
    echo ""
    echo "After installing, start the Ollama server:"
    echo "  ollama serve"
    exit 1
fi

OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
echo -e "${GREEN}✓ Ollama found: ${OLLAMA_VERSION}${NC}"
echo ""

# -------------------------------------------------------------------
# 2. Pull required models
# -------------------------------------------------------------------
echo -e "${YELLOW}[2/4] Pulling required models...${NC}"
echo ""

echo "Pulling ${SUPERVISOR_MODEL} (Supervisor agent)..."
ollama pull "${SUPERVISOR_MODEL}"
echo -e "${GREEN}✓ ${SUPERVISOR_MODEL} ready${NC}"
echo ""

echo "Pulling ${AGENT_MODEL} (Sub-agents)..."
ollama pull "${AGENT_MODEL}"
echo -e "${GREEN}✓ ${AGENT_MODEL} ready${NC}"
echo ""

# -------------------------------------------------------------------
# 3. Test models
# -------------------------------------------------------------------
echo -e "${YELLOW}[3/4] Testing models...${NC}"
echo ""

echo "Testing ${SUPERVISOR_MODEL}..."
RESPONSE=$(ollama run "${SUPERVISOR_MODEL}" "Say 'Hello, I am ready.' and nothing else." 2>/dev/null || echo "FAILED")
if [[ "${RESPONSE}" == "FAILED" ]]; then
    echo -e "${RED}✗ ${SUPERVISOR_MODEL} test failed${NC}"
else
    echo -e "${GREEN}✓ ${SUPERVISOR_MODEL} responded: ${RESPONSE}${NC}"
fi
echo ""

echo "Testing ${AGENT_MODEL}..."
RESPONSE=$(ollama run "${AGENT_MODEL}" "Say 'Hello, I am ready.' and nothing else." 2>/dev/null || echo "FAILED")
if [[ "${RESPONSE}" == "FAILED" ]]; then
    echo -e "${RED}✗ ${AGENT_MODEL} test failed${NC}"
else
    echo -e "${GREEN}✓ ${AGENT_MODEL} responded: ${RESPONSE}${NC}"
fi
echo ""

# -------------------------------------------------------------------
# 4. GPU status
# -------------------------------------------------------------------
echo -e "${YELLOW}[4/4] GPU Status...${NC}"
echo ""
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found — running on CPU or non-NVIDIA GPU."
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Setup complete! ✓${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and set your TAVILY_API_KEY"
echo "  2. Install Python dependencies: pip install -r requirements.txt"
echo "  3. Run: python main.py --query 'Your research question here'"
