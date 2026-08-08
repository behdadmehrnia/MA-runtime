import logging
import tempfile
import os
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch

from app.config import settings
from app.models import (
    load_all_models, 
    is_healthy, 
    get_gpu_info,
    get_whisper_model,
    get_diarization_pipeline
)
from app.transcription import transcribe_audio
from app.diarization import diarize_audio
from app.merging import align_speakers_to_transcript

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up audio service...")
    try:
        load_all_models()
        logger.info("Audio service startup complete")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise
    yield
    logger.info("Shutting down audio service...")


app = FastAPI(
    title="Meeting AI Audio Service",
    description="Persian speech-to-text with speaker diarization",
    version="1.0.0",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    status: str
    whisper_loaded: bool
    diarization_loaded: bool
    cuda: bool
    gpu: str | None


class GPUResponse(BaseModel):
    cuda: bool
    gpu: str | None
    total_vram_gb: float | None
    allocated_vram_gb: float | None
    reserved_vram_gb: float | None
    free_vram_gb: float | None


class TranscribeResponse(BaseModel):
    segments: List[Dict[str, Any]]


class DiarizeResponse(BaseModel):
    segments: List[Dict[str, Any]]


class ProcessResponse(BaseModel):
    segments: List[Dict[str, Any]]


def validate_audio_file(file: UploadFile) -> None:
    max_size = settings.max_audio_file_size_mb * 1024 * 1024
    
    if file.size and file.size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_audio_file_size_mb} MB"
        )
    
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )


async def save_upload_file(file: UploadFile) -> str:
    validate_audio_file(file)
    
    ext = os.path.splitext(file.filename or "")[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        return tmp.name


def cleanup_file(filepath: str) -> None:
    try:
        if os.path.exists(filepath):
            os.unlink(filepath)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {filepath}: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    health = is_healthy()
    return HealthResponse(
        status="ok" if health["whisper_loaded"] and health["diarization_loaded"] else "degraded",
        whisper_loaded=health["whisper_loaded"],
        diarization_loaded=health["diarization_loaded"],
        cuda=health["cuda"],
        gpu=health["gpu"],
    )


@app.get("/gpu", response_model=GPUResponse)
async def gpu_info():
    info = get_gpu_info()
    return GPUResponse(**info)


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_endpoint(file: UploadFile = File(...)):
    async with semaphore:
        temp_path = await save_upload_file(file)
        try:
            segments = transcribe_audio(temp_path, language="persian", task="transcribe")
            return TranscribeResponse(segments=segments)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cleanup_file(temp_path)


@app.post("/diarize", response_model=DiarizeResponse)
async def diarize_endpoint(file: UploadFile = File(...)):
    async with semaphore:
        temp_path = await save_upload_file(file)
        try:
            segments = diarize_audio(temp_path)
            return DiarizeResponse(segments=segments)
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cleanup_file(temp_path)


@app.post("/process", response_model=ProcessResponse)
async def process_endpoint(file: UploadFile = File(...)):
    async with semaphore:
        temp_path = await save_upload_file(file)
        try:
            logger.info(f"Processing audio file: {file.filename}")
            
            whisper_segments = transcribe_audio(temp_path, language="persian", task="transcribe")
            logger.info(f"Whisper produced {len(whisper_segments)} segments")
            
            diarization_segments = diarize_audio(temp_path)
            logger.info(f"Diarization produced {len(diarization_segments)} segments")
            
            aligned = align_speakers_to_transcript(whisper_segments, diarization_segments)
            
            return ProcessResponse(segments=aligned)
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cleanup_file(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)