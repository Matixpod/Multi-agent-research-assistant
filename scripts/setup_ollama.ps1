# scripts/setup_ollama.ps1
# PowerShell setup script for Ollama models required by Multi-Agent Research Assistant.

$ErrorActionPreference = "Stop"

$SUPERVISOR_MODEL = "qwen2.5:7b-instruct"
$AGENT_MODEL = "mistral:7b-instruct"

Write-Host "========================================" -ForegroundColor Green
Write-Host " Multi-Agent Research Assistant Setup"   -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# -------------------------------------------------------------------
# 1. Check if Ollama is installed
# -------------------------------------------------------------------
Write-Host "[1/4] Checking Ollama installation..." -ForegroundColor Yellow

$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaCmd) {
    Write-Host "ERROR: Ollama is not installed." -ForegroundColor Red
    Write-Host ""
    Write-Host "Install Ollama:"
    Write-Host "  Download from https://ollama.com/download"
    Write-Host ""
    Write-Host "After installing, start the Ollama server:"
    Write-Host "  ollama serve"
    exit 1
}

$ollamaVersion = & ollama --version 2>$null
Write-Host "OK Ollama found: $ollamaVersion" -ForegroundColor Green
Write-Host ""

# -------------------------------------------------------------------
# 2. Pull required models
# -------------------------------------------------------------------
Write-Host "[2/4] Pulling required models..." -ForegroundColor Yellow
Write-Host ""

Write-Host "Pulling $SUPERVISOR_MODEL (Supervisor agent)..."
& ollama pull $SUPERVISOR_MODEL
Write-Host "OK $SUPERVISOR_MODEL ready" -ForegroundColor Green
Write-Host ""

Write-Host "Pulling $AGENT_MODEL (Sub-agents)..."
& ollama pull $AGENT_MODEL
Write-Host "OK $AGENT_MODEL ready" -ForegroundColor Green
Write-Host ""

# -------------------------------------------------------------------
# 3. Test models
# -------------------------------------------------------------------
Write-Host "[3/4] Testing models..." -ForegroundColor Yellow
Write-Host ""

Write-Host "Testing $SUPERVISOR_MODEL..."
try {
    $response = & ollama run $SUPERVISOR_MODEL "Say 'Hello, I am ready.' and nothing else." 2>$null
    Write-Host "OK $SUPERVISOR_MODEL responded: $response" -ForegroundColor Green
} catch {
    Write-Host "FAIL $SUPERVISOR_MODEL test failed" -ForegroundColor Red
}
Write-Host ""

Write-Host "Testing $AGENT_MODEL..."
try {
    $response = & ollama run $AGENT_MODEL "Say 'Hello, I am ready.' and nothing else." 2>$null
    Write-Host "OK $AGENT_MODEL responded: $response" -ForegroundColor Green
} catch {
    Write-Host "FAIL $AGENT_MODEL test failed" -ForegroundColor Red
}
Write-Host ""

# -------------------------------------------------------------------
# 4. GPU status
# -------------------------------------------------------------------
Write-Host "[4/4] GPU Status..." -ForegroundColor Yellow
Write-Host ""

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    & nvidia-smi
} else {
    Write-Host "nvidia-smi not found — running on CPU or non-NVIDIA GPU."
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Copy .env.example to .env and set your TAVILY_API_KEY"
Write-Host "  2. Install Python dependencies: pip install -r requirements.txt"
Write-Host "  3. Run: python main.py --query 'Your research question here'"
