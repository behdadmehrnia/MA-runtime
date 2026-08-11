# Meeting Assistant Local Inference

Production-ready local inference stack for meeting transcription with speaker diarization.
Runs on **NVIDIA GPU (CUDA)**, **Apple Silicon (MPS)**, or **CPU**.

## Architecture

```
                    GPU / MPS / CPU
                         |
            +------------+------------+
            |                         |
         audio                       gemma
            |                         |
       +----+------+              vLLM
       |           |                |
   Whisper      pyannote         Gemma 4
   Large-v3     3.1             12B (4-bit)
       |           |
       +------+----+
              |
         FastAPI API
```

**Services:**
- **gemma**: vLLM OpenAI-compatible server with Gemma 4 12B (W4A16 quantized) — *NVIDIA GPU only*
- **audio**: FastAPI service with **configurable models**:
  - Whisper (transcription) + pyannote (speaker diarization) — **both**
  - Whisper only — **whisper**
  - pyannote only — **diarization**

## Hardware Compatibility

| Platform | Backend | Whisper | pyannote | Gemma/vLLM | Notes |
|----------|---------|---------|----------|------------|-------|
| **NVIDIA GPU** | CUDA | ✅ | ✅ | ✅ | RTX 30/40 series, A100, H100, etc. |
| **Apple Silicon** | MPS | ✅ | ✅ | ❌ | M1/M2/M3 Macs (Metal Performance Shaders) |
| **CPU only** | CPU | ✅ | ✅ | ❌ | Any x86_64/ARM64, slower |

### VRAM Requirements (NVIDIA GPU)

| Model Mode | Whisper | pyannote | Gemma (W4A16) | Total (est.) |
|------------|---------|----------|---------------|--------------|
| `both` | ~3 GB | ~2 GB | ~12 GB | **~17 GB** |
| `whisper` | ~3 GB | — | ~12 GB | **~15 GB** |
| `diarization` | — | ~2 GB | ~12 GB | **~14 GB** |
| **audio only** (no Gemma) | ~3 GB | ~2 GB | — | **~5 GB** |

> **Minimum GPU:** 8 GB VRAM (audio service only, `diarization` or `whisper` mode)
> **Recommended:** 12+ GB VRAM for full stack, 24 GB for comfortable headroom

## Software Requirements

### NVIDIA GPU (Linux)
- Ubuntu 22.04/24.04 LTS
- NVIDIA Driver **550+** (CUDA 12.4)
- NVIDIA Container Toolkit
- Docker 24+ / Docker Compose v2

### Apple Silicon (macOS)
- macOS 13+ (Ventura) / 14+ (Sonoma)
- Docker Desktop 4.25+ (has MPS support)
- No NVIDIA toolkit needed

### CPU Only (Linux/macOS/Windows)
- Docker 24+ / Docker Compose v2
- No GPU drivers needed

## Quick Start

### 1. Clone & Configure

```bash
cd /Users/bmdarklight/Projects/MA-runtime

# Copy and edit environment
cp .env.example .env
# Required: Add your HF_TOKEN for pyannote
# Optional: Set MODEL_MODE (both|whisper|diarization), defaults to "both"
```

### 2. Start Services

```bash
# Full stack (Gemma + Audio)
docker compose up -d

# Audio only (no Gemma/vLLM) - works on any platform
docker compose up -d audio

# With custom model mode
MODEL_MODE=diarization docker compose up -d audio
```

### 3. Verify

```bash
# Health check
curl http://127.0.0.1:8001/health

# GPU/Backend info
curl http://127.0.0.1:8001/gpu
```

## Configuration

### `.env` Variables (Audio Service)

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(required)* | Hugging Face token for gated models (pyannote) |
| `MODEL_MODE` | `both` | **Model selection: `both` \| `whisper` \| `diarization`** |
| `WHISPER_MODEL` | `openai/whisper-large-v3` | Whisper model ID (openai/whisper-*) |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | pyannote model ID |
| `WHISPER_DEVICE` | `cuda` | `cuda` \| `mps` \| `cpu` (auto-detected if unset) |
| `WHISPER_COMPUTE_TYPE` | `float16` | `float16` \| `float32` \| `int8` |
| `DIARIZATION_DEVICE` | `cuda` | `cuda` \| `mps` \| `cpu` (auto-detected if unset) |
| `MAX_AUDIO_FILE_SIZE_MB` | `100` | Max upload size |
| `MAX_CONCURRENT_JOBS` | `1` | Concurrent GPU jobs |
| `LOG_LEVEL` | `INFO` | Log level |

### Model Mode Examples

```bash
# Full pipeline (Whisper + Diarization) — default
MODEL_MODE=both docker compose up -d audio

# Transcription only (no speaker labels) — saves ~2 GB VRAM
MODEL_MODE=whisper docker compose up -d audio

# Speaker diarization only (no transcription) — saves ~3 GB VRAM
MODEL_MODE=diarization docker compose up -d audio

# CPU-only (no GPU needed)
WHISPER_DEVICE=cpu DIARIZATION_DEVICE=cpu MODEL_MODE=both docker compose up -d audio
```

## API Endpoints

| Endpoint | Description | Requires |
|----------|-------------|----------|
| `GET /health` | Service health + loaded models | — |
| `GET /gpu` | GPU/MPS/CPU info + VRAM | — |
| `POST /transcribe` | Speech-to-text only | `MODEL_MODE=both` or `whisper` |
| `POST /diarize` | Speaker diarization only | `MODEL_MODE=both` or `diarization` |
| `POST /process` | Combined (transcribe + diarize + align) | `MODEL_MODE=both` |

### Example Requests

```bash
# Transcribe
curl -X POST -F "file=@audio-files/test.wav" http://127.0.0.1:8001/transcribe

# Diarize
curl -X POST -F "file=@audio-files/test.wav" http://127.0.0.1:8001/diarize

# Full pipeline (transcribe + diarize + align)
curl -X POST -F "file=@audio-files/test.wav" http://127.0.0.1:8001/process
```

**Response (`/process`):**
```json
{
  "segments": [
    {"start": 0.0, "end": 3.5, "speaker": "SPEAKER_00", "text": "سلام به همه دوستان"},
    {"start": 3.5, "end": 7.2, "speaker": "SPEAKER_01", "text": "بله، FastAPI هم بررسی می‌کنیم"}
  ]
}
```

## Platform-Specific Notes

### NVIDIA GPU (Linux)

```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker info | grep -i nvidia
docker run --rm --gpus all nvidia/cuda:12.4.1-base nvidia-smi
```

### Apple Silicon (macOS)

- Use **Docker Desktop 4.25+** (Settings → General → "Use Rosetta for x86/amd64 emulation on Apple Silicon")
- MPS backend auto-detected; no extra config needed
- `WHISPER_DEVICE=mps DIARIZATION_DEVICE=mps` works automatically
- Gemma/vLLM **not supported** (no CUDA)

### CPU Only (Any Platform)

```bash
# Force CPU mode
WHISPER_DEVICE=cpu DIARIZATION_DEVICE=cpu MODEL_MODE=both docker compose up -d audio

# Or let auto-detection handle it (no NVIDIA GPU = CPU)
docker compose up -d audio
```

## Health Checks & Monitoring

```bash
# Service health
./scripts/healthcheck.sh

# GPU/Backend status
curl http://127.0.0.1:8001/gpu

# Logs
./scripts/logs.sh audio

# Real-time GPU monitoring (NVIDIA only)
./scripts/gpu-status.sh
# Or: watch -n 1 nvidia-smi
```

## Version Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| CUDA (base) | 12.4.1 | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` |
| PyTorch | 2.4.1 | CUDA 12.4 / MPS / CPU |
| vLLM | 0.6.3 | Gemma 4 W4A16 (NVIDIA only) |
| Transformers | 4.44.2 | |
| Whisper | 20240930 | openai-whisper |
| pyannote.audio | 3.3.2 | Low VRAM, stable |
| pyannote.core | 6.1.0 | |

## Production Notes

- Ports bound to `127.0.0.1` (not `0.0.0.0`)
- `HF_TOKEN` never exposed via API or logs
- Temp files cleaned up after processing
- Upload size limited (default 100 MB)
- Structured logging
- Health checks on both services
- Automatic restart on failure (`unless-stopped`)

## License

MIT License - see LICENSE file.