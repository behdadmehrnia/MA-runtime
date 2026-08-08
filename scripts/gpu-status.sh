#!/usr/bin/env bash
set -euo pipefail

echo "GPU Status Monitor (press Ctrl+C to exit)"
echo "=========================================="
watch -n 1 nvidia-smi