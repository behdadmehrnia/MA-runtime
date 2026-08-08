#!/usr/bin/env bash
set -euo pipefail

VLLM_PORT="${VLLM_PORT:-8000}"
AUDIO_PORT="${AUDIO_PORT:-8001}"

VLLM_URL="http://127.0.0.1:${VLLM_PORT}/v1/models"
AUDIO_HEALTH_URL="http://127.0.0.1:${AUDIO_PORT}/health"
AUDIO_GPU_URL="http://127.0.0.1:${AUDIO_PORT}/gpu"

echo "=========================================="
echo "Meeting AI - Health Check"
echo "=========================================="
echo ""

check_service() {
    local name="$1"
    local url="$2"
    local expected_field="${3:-}"
    
    echo -n "Checking $name... "
    if response=$(curl -s -f --max-time 10 "$url" 2>/dev/null); then
        echo "OK"
        if [[ -n "$expected_field" ]]; then
            echo "$response" | python3 -m json.tool | grep -A2 -B2 "$expected_field" || true
        fi
        return 0
    else
        echo "FAILED"
        return 1
    fi
}

all_ok=true

check_service "Gemma (vLLM)" "$VLLM_URL" "id" || all_ok=false
echo ""
check_service "Audio API Health" "$AUDIO_HEALTH_URL" "status" || all_ok=false
echo ""
check_service "Audio API GPU Info" "$AUDIO_GPU_URL" "cuda" || all_ok=false
echo ""

if $all_ok; then
    echo "=========================================="
    echo "All services healthy!"
    echo "=========================================="
    exit 0
else
    echo "=========================================="
    echo "Some services are unhealthy"
    echo "=========================================="
    exit 1
fi