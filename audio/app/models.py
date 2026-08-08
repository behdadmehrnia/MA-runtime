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
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    free_mem = total_mem - reserved
    
    return {
        "cuda": True,
        "gpu": gpu_name,
        "total_vram_gb": round(total_mem / (1024**3), 2),
        "allocated_vram_gb": round(allocated / (1024**3), 2),
        "reserved_vram_gb": round(reserved / (1024**3), 2),
        "free_vram_gb": round(free_mem / (1024**3), 2),
    }


def load_whisper_model() -> Any:
    global _whisper_model
    
    if _whisper_model is not None:
        return _whisper_model
    
    logger.info(f"Loading Whisper model: {settings.whisper_model}")
    
    import whisper
    
    device = get_device()
    
    _whisper_model = whisper.load_model(
        settings.whisper_model.replace("openai/", ""),
        device=device,
        download_root="/root/.cache/huggingface/hub"
    )
    
    if settings.whisper_compute_type == "float16" and device.type == "cuda":
        _whisper_model = _whisper_model.half()
    
    _whisper_model.eval()
    
    logger.info("Whisper model loaded successfully")
    return _whisper_model


def load_diarization_pipeline() -> Any:
    global _diarization_pipeline
    
    if _diarization_pipeline is not None:
        return _diarization_pipeline
    
    logger.info(f"Loading pyannote diarization pipeline: {settings.diarization_model}")
    
    from pyannote.audio import Pipeline
    
    _diarization_pipeline = Pipeline.from_pretrained(
        settings.diarization_model,
        use_auth_token=settings.hf_token or None
    )
    
    device = get_device()
    _diarization_pipeline.to(device)
    
    logger.info("Diarization pipeline loaded successfully")
    return _diarization_pipeline


def load_all_models() -> None:
    global _models_loaded
    
    if _models_loaded:
        return
    
    logger.info("Starting model loading...")
    
    load_whisper_model()
    torch.cuda.empty_cache()
    
    load_diarization_pipeline()
    torch.cuda.empty_cache()
    
    _models_loaded = True
    logger.info("All models loaded successfully")


def get_whisper_model() -> Any:
    if _whisper_model is None:
        load_whisper_model()
    return _whisper_model


def get_diarization_pipeline() -> Any:
    if _diarization_pipeline is None:
        load_diarization_pipeline()
    return _diarization_pipeline


def is_healthy() -> dict:
    return {
        "whisper_loaded": _whisper_model is not None,
        "diarization_loaded": _diarization_pipeline is not None,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }