#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Meeting AI - Startup Check"
echo "=========================================="

# Check NVIDIA GPU availability
echo ""
echo "[1/5] Checking NVIDIA GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA drivers not installed?"
    exit 1
fi

nvidia-smi
echo ""

if ! nvidia-smi | grep -q "RTX 4090"; then
    echo "WARNING: RTX 4090 not detected. Check GPU compatibility."
fi

# Check .env file
echo "[2/5] Checking .env file..."
ENV_FILE="$PROJECT_ROOT/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env file not found at $ENV_FILE"
    echo "Please copy .env.example to .env and fill in your HF_TOKEN:"
    echo "  cp $PROJECT_ROOT/.env.example $PROJECT_ROOT/.env"
    echo "  # Then edit .env and add your Hugging Face token"
    exit 1
fi

# Check HF_TOKEN is set
source "$ENV_FILE"
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN is not set in .env"
    echo "Gated models (pyannote) will fail to download."
fi

# Check Docker
echo "[3/5] Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Please install Docker."
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "ERROR: Docker Compose not available."
    exit 1
fi

# Check NVIDIA Container Toolkit
echo "[4/5] Checking NVIDIA Container Toolkit..."
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo "WARNING: NVIDIA Container Toolkit may not be configured."
    echo "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Start services
echo "[5/5] Starting Docker Compose services..."
cd "$PROJECT_ROOT"
docker compose pull
docker compose up -d

echo ""
echo "=========================================="
echo "Startup complete!"
echo "=========================================="
echo ""
echo "Services:"
echo "  Gemma (vLLM):  http://127.0.0.1:${VLLM_PORT:-8000}/v1"
echo "  Audio API:     http://127.0.0.1:${AUDIO_PORT:-8001}"
echo ""
echo "Useful commands:"
echo "  View logs:     $SCRIPT_DIR/logs.sh"
echo "  GPU status:    $SCRIPT_DIR/gpu-status.sh"
echo "  Health check:  $SCRIPT_DIR/healthcheck.sh"
echo "  Stop services: $SCRIPT_DIR/stop.sh"
echo ""
echo "Monitor GPU memory:"
echo "  watch -n 1 nvidia-smi"