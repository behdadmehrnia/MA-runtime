import logging
import torch
from typing import Optional, Any
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)

_whisper_model: Optional[Any] = None
_diarization_pipeline: Optional[Any] = None
_models_loaded = False


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_gpu_info() -> dict:
    """Get GPU information if available."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        free_mem = total_mem - reserved
        return {
            "cuda": True,
            "backend": "cuda",
            "gpu": gpu_name,
            "total_vram_gb": round(total_mem / (1024**3), 2),
            "allocated_vram_gb": round(allocated / (1024**3), 2),
            "reserved_vram_gb": round(reserved / (1024**3), 2),
            "free_vram_gb": round(free_mem / (1024**3), 2),
        }
    elif torch.backends.mps.is_available():
        return {
            "cuda": False,
            "backend": "mps",
            "gpu": "Apple Silicon (MPS)",
            "total_vram_gb": None,
            "allocated_vram_gb": None,
            "reserved_vram_gb": None,
            "free_vram_gb": None,
        }
    return {"cuda": False, "backend": "cpu", "gpu": None}


def load_whisper_model() -> Any:
    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model

    if settings.model_mode == "diarization":
        logger.info("Skipping Whisper load (MODEL_MODE=diarization)")
        return None

    logger.info(f"Loading Whisper model: {settings.whisper_model}")

    import whisper

    device = get_device()

    _whisper_model = whisper.load_model(
        settings.whisper_model.replace("openai/", ""),
        device=device,
        download_root="/root/.cache/huggingface/hub"
    )

    # Apply compute type optimization
    if settings.whisper_compute_type == "float16" and device.type == "cuda":
        _whisper_model = _whisper_model.half()
    elif settings.whisper_compute_type == "int8" and device.type == "cuda":
        # Note: int8 requires bitsandbytes or similar, whisper.cpp handles this better
        logger.warning("int8 compute type not fully supported with openai-whisper, using float16")
        _whisper_model = _whisper_model.half()

    _whisper_model.eval()

    logger.info(f"Whisper model loaded successfully on {device}")
    return _whisper_model


def load_diarization_pipeline() -> Any:
    global _diarization_pipeline

    if _diarization_pipeline is not None:
        return _diarization_pipeline

    if settings.model_mode == "whisper":
        logger.info("Skipping diarization load (MODEL_MODE=whisper)")
        return None

    logger.info(f"Loading pyannote diarization pipeline: {settings.diarization_model}")

    from pyannote.audio import Pipeline

    _diarization_pipeline = Pipeline.from_pretrained(
        settings.diarization_model,
        use_auth_token=settings.hf_token or None
    )

    device = get_device()
    _diarization_pipeline.to(device)

    logger.info(f"Diarization pipeline loaded successfully on {device}")
    return _diarization_pipeline


def load_all_models() -> None:
    global _models_loaded

    if _models_loaded:
        return

    logger.info(f"Starting model loading (MODEL_MODE={settings.model_mode})...")

    if settings.model_mode in ("both", "whisper"):
        load_whisper_model()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if settings.model_mode in ("both", "diarization"):
        load_diarization_pipeline()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _models_loaded = True
    logger.info("All requested models loaded successfully")


def get_whisper_model() -> Any:
    if settings.model_mode == "diarization":
        return None
    if _whisper_model is None:
        load_whisper_model()
    return _whisper_model


def get_diarization_pipeline() -> Any:
    if settings.model_mode == "whisper":
        return None
    if _diarization_pipeline is None:
        load_diarization_pipeline()
    return _diarization_pipeline


def is_healthy() -> dict:
    gpu_info = get_gpu_info()
    return {
        "whisper_loaded": _whisper_model is not None or settings.model_mode == "diarization",
        "diarization_loaded": _diarization_pipeline is not None or settings.model_mode == "whisper",
        "model_mode": settings.model_mode,
        **gpu_info,
    }