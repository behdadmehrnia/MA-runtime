import logging
import tempfile
import os
from typing import List, Dict, Any
import torch

from app.models import get_diarization_pipeline, get_device

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: str) -> List[Dict[str, Any]]:
    pipeline = get_diarization_pipeline()
    device = get_device()
    
    logger.info(f"Running speaker diarization on: {audio_path}")
    
    with torch.inference_mode():
        diarization = pipeline(audio_path)
    
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker,
        })
    
    logger.info(f"Diarization complete: {len(segments)} speaker segments")
    return segments