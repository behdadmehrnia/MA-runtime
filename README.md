# Meeting Assistant Local Inference

Production-ready local inference stack for meeting transcription with speaker diarization, running on a single NVIDIA RTX 4090 (24 GB VRAM).

## Architecture

```
                    RTX 4090 24GB
                         |
            +------------+------------+
            |                         |
         audio                       gemma
            |                         |
      +-----+------+              vLLM
      |            |                |
  Whisper      pyannote         Gemma 4
  Large-v3     3.1             12B (4-bit)
      |            |
      +------+-----+
             |
        FastAPI API
```

**Services:**
- **gemma**: vLLM OpenAI-compatible server with Gemma 4 12B (W4A16 quantized)
- **audio**: FastAPI service with Whisper Large-v3 + pyannote Speaker Diarization 3.1

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA RTX 4090 (24 GB VRAM) | RTX 4090 (24 GB) |
| RAM | 32 GB | 64 GB |
| Storage | 50 GB free | 100 GB free (NVMe) |
| CPU | 8 cores | 16+ cores |

## Software Requirements

### Ubuntu
- Ubuntu 22.04 LTS or 24.04 LTS

### NVIDIA Driver
- **550+** (for CUDA 12.4 support)
- Verify: `nvidia-smi`

### NVIDIA Container Toolkit
Required for GPU access in Docker containers.

```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify: `docker info | grep -i nvidia`

### Docker & Docker Compose
- Docker 24+
- Docker Compose v2 (built into Docker Desktop or `docker compose` plugin)

## Hugging Face Authentication

**Required for gated models (pyannote).**

1. **Create token**: https://huggingface.co/settings/tokens (read access)
2. **Accept pyannote conditions**: https://huggingface.co/pyannote/speaker-diarization-3.1
3. **Configure**:

```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

**Never commit `.env`** - it's gitignored.

## First Startup

```bash
# 1. Clone/navigate to project
cd meeting-ai

# 2. Configure environment
cp .env.example .env
# Edit .env with your HF_TOKEN

# 3. Run startup script (validates GPU, Docker, starts services)
./scripts/start.sh
```

The startup script will:
- Verify NVIDIA GPU access
- Check `.env` exists
- Pull/build Docker images
- Start both services

## Subsequent Startups

```bash
# Start services
docker compose up -d

# Or use the script
./scripts/start.sh
```

## Health Checks

```bash
# Run all health checks
./scripts/healthcheck.sh

# Individual checks
curl http://127.0.0.1:8000/v1/models    # Gemma/vLLM
curl http://127.0.0.1:8001/health       # Audio service
curl http://127.0.0.1:8001/gpu          # GPU info
```

## Testing

### Test Whisper
```bash
curl -X POST -F "file=@audio-files/test.wav" http://127.0.0.1:8001/transcribe
```

### Test pyannote (Speaker Diarization)
```bash
curl -X POST -F "file=@audio-files/test.wav" http://127.0.0.1:8001/diarize
```

### Test Gemma (LLM)
```bash
# List models
curl http://127.0.0.1:8000/v1/models

# Chat completion
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-12b-it-qat-w4a16-ct",
    "messages": [{"role": "user", "content": "سلام، چطور می‌توانم کمکتان کنم؟"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Test Combined Pipeline (`/process`)
```bash
curl -X POST -F "file=@audio-files/test.wav" http://127.0.0.1:8001/process
```

**Response format:**
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "speaker": "SPEAKER_00",
      "text": "سلام به همه دوستان، امروز در مورد API و REST صحبت می‌کنیم"
    },
    {
      "start": 3.5,
      "end": 7.2,
      "speaker": "SPEAKER_01",
      "text": "بله، FastAPI و Kubernetes هم بررسی خواهیم کرد"
    }
  ]
}
```

### Run Full Test Suite
```bash
./scripts/test.sh
```

Place a test file at `audio-files/test.wav` to enable the `/process` test.

## GPU Monitoring

```bash
# Real-time GPU monitoring
./scripts/gpu-status.sh
# Or directly:
watch -n 1 nvidia-smi
```

## VRAM Troubleshooting

### Expected VRAM Usage (Approximate)

| Component | VRAM Usage |
|-----------|------------|
| Gemma 4 12B (W4A16, gpu_mem_util=0.50) | ~12 GB |
| Whisper Large-v3 (FP16) | ~3 GB |
| pyannote Diarization 3.1 | ~2 GB |
| **Total** | **~17 GB** |
| **Headroom** | **~7 GB** |

### If CUDA OOM Occurs

1. **Reduce vLLM memory utilization** (in `.env`):
   ```bash
   VLLM_GPU_MEMORY_UTILIZATION=0.45  # Try lower values
   ```

2. **Reduce max sequence length**:
   ```bash
   VLLM_MAX_MODEL_LEN=8192
   ```

3. **Restart services**:
   ```bash
   docker compose down && docker compose up -d
   ```

4. **Monitor**: `watch -n 1 nvidia-smi`

### How to Increase vLLM Memory Utilization

Start conservative, increase gradually while monitoring:

```bash
# Step 1: Current (safe)
VLLM_GPU_MEMORY_UTILIZATION=0.50

# Step 2: If headroom > 4 GB consistently
VLLM_GPU_MEMORY_UTILIZATION=0.55

# Step 3: If headroom > 3 GB consistently  
VLLM_GPU_MEMORY_UTILIZATION=0.60
```

**After each change:**
```bash
docker compose down && docker compose up -d
./scripts/healthcheck.sh
watch -n 1 nvidia-smi  # Monitor for 10+ minutes
```

**Do NOT exceed 0.70** - Whisper + pyannote need ~5-6 GB minimum.

## Configuration Reference

### `.env` Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(required)* | Hugging Face token for gated models |
| `GEMMA_MODEL` | `google/gemma-4-12b-it-qat-w4a16-ct` | Gemma model ID |
| `VLLM_PORT` | `8000` | vLLM API port |
| `AUDIO_PORT` | `8001` | Audio API port |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.50` | GPU memory fraction for vLLM |
| `VLLM_MAX_MODEL_LEN` | `16384` | Max context length |
| `VLLM_MAX_NUM_SEQS` | `1` | Max concurrent sequences |
| `WHISPER_MODEL` | `openai/whisper-large-v3` | Whisper model |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | Diarization model |
| `MAX_AUDIO_FILE_SIZE_MB` | `100` | Max upload size |
| `MAX_CONCURRENT_JOBS` | `1` | Concurrent GPU jobs |
| `LOG_LEVEL` | `INFO` | Log level |

## Model Storage

Models are cached in Docker volume `hf_cache` (mapped to `/root/.cache/huggingface` in containers).

**Host location**: `/var/lib/docker/volumes/meeting-ai_hf_cache/_data`

To backup: `docker run --rm -v meeting-ai_hf_cache:/data -v $(pwd):/backup alpine tar czf /backup/hf_cache_backup.tar.gz -C /data .`

## Stopping Services

```bash
./scripts/stop.sh
# Or:
docker compose down
```

## Restarting Services

```bash
docker compose restart
# Or full restart:
docker compose down && docker compose up -d
```

## Logs

```bash
# All services
./scripts/logs.sh

# Specific service
./scripts/logs.sh gemma
./scripts/logs.sh audio
```

## Common Issues

### "CUDA out of memory"
- Reduce `VLLM_GPU_MEMORY_UTILIZATION`
- Reduce `VLLM_MAX_MODEL_LEN`
- Ensure no other GPU processes running

### "pyannote model not found" / 401 Unauthorized
- Verify `HF_TOKEN` in `.env`
- Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
- Token needs `read` permission

### "Whisper: CUDA error" on startup
- Check driver version: `nvidia-smi`
- Ensure CUDA 12.4 compatible driver (550+)

### Services unhealthy after startup
- Check logs: `./scripts/logs.sh audio`
- Model download may take 5-10 minutes on first run
- Increase healthcheck `start_period` in docker-compose.yml if needed

### Audio format not supported
- Ensure ffmpeg is installed in container (it is)
- Supported: wav, mp3, m4a, flac, ogg, webm, mp4

## Version Compatibility Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| CUDA | 12.4 | Base image: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` |
| PyTorch | 2.4.1 | CUDA 12.4 build |
| vLLM | 0.6.3 | Supports Gemma 4 W4A16 compressed-tensors |
| Transformers | 4.44.2 | |
| Whisper | 20240930 | openai-whisper |
| pyannote.audio | 3.3.2 | Speaker diarization 3.1 |


## Production Considerations

- Ports bound to `127.0.0.1` only (not `0.0.0.0`)
- `HF_TOKEN` never exposed via API or logs
- Temporary files cleaned up after processing
- Upload size limited (default 100 MB)
- Structured logging
- Health checks on both services
- Automatic restart on failure

## License

MIT License - see LICENSE file.