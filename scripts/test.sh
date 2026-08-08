#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

VLLM_PORT="${VLLM_PORT:-8000}"
AUDIO_PORT="${AUDIO_PORT:-8001}"

VLLM_URL="http://127.0.0.1:${VLLM_PORT}/v1/models"
AUDIO_HEALTH_URL="http://127.0.0.1:${AUDIO_PORT}/health"
AUDIO_GPU_URL="http://127.0.0.1:${AUDIO_PORT}/gpu"
AUDIO_PROCESS_URL="http://127.0.0.1:${AUDIO_PORT}/process"

TEST_FILE="$PROJECT_ROOT/audio-files/test.wav"

echo "=========================================="
echo "Meeting AI - Test Suite"
echo "=========================================="
echo ""

# Test 1: Check GPU availability
echo "[1/5] Checking GPU availability..."
if nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader
    echo "GPU check: OK"
else
    echo "GPU check: FAILED - nvidia-smi not available"
    exit 1
fi
echo ""

# Test 2: Check vLLM
echo "[2/5] Checking vLLM /v1/models..."
if response=$(curl -s -f --max-time 10 "$VLLM_URL" 2>/dev/null); then
    echo "$response" | python3 -m json.tool
    echo "vLLM check: OK"
else
    echo "vLLM check: FAILED - Is the gemma service running?"
    exit 1
fi
echo ""

# Test 3: Check Audio API health
echo "[3/5] Checking Audio API /health..."
if response=$(curl -s -f --max-time 10 "$AUDIO_HEALTH_URL" 2>/dev/null); then
    echo "$response" | python3 -m json.tool
    echo "Audio health check: OK"
else
    echo "Audio health check: FAILED - Is the audio service running?"
    exit 1
fi
echo ""

# Test 4: Check Audio API GPU info
echo "[4/5] Checking Audio API /gpu..."
if response=$(curl -s -f --max-time 10 "$AUDIO_GPU_URL" 2>/dev/null); then
    echo "$response" | python3 -m json.tool
    echo "Audio GPU check: OK"
else
    echo "Audio GPU check: FAILED"
    exit 1
fi
echo ""

# Test 5: Test /process endpoint if test file exists
echo "[5/5] Testing /process endpoint..."
if [[ -f "$TEST_FILE" ]]; then
    echo "Using test file: $TEST_FILE"
    if response=$(curl -s -f --max-time 300 -X POST -F "file=@$TEST_FILE" "$AUDIO_PROCESS_URL" 2>/dev/null); then
        echo "$response" | python3 -m json.tool
        echo "Process endpoint: OK"
    else
        echo "Process endpoint: FAILED"
        exit 1
    fi
else
    echo "Test file not found: $TEST_FILE"
    echo "Skipping /process test. Add a test.wav file to audio-files/ to enable this test."
fi
echo ""

echo "=========================================="
echo "All tests passed!"
echo "=========================================="