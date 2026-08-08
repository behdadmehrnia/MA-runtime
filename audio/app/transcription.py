import logging
import tempfile
import os
from typing import List, Dict, Any
import torch
import numpy as np

from app.models import get_whisper_model, get_device

logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_path: str,
    language: str = "persian",
    task: str = "transcribe"
) -> List[Dict[str, Any]]:
    model = get_whisper_model()
    device = get_device()
    
    logger.info(f"Transcribing audio: {audio_path} (language={language}, task={task})")
    
    with torch.inference_mode():
        result = model.transcribe(
            audio_path,
            language=language,
            task=task,
            verbose=False,
            fp16=(device.type == "cuda"),
            condition_on_previous_text=True,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
    
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg["text"].strip(),
        })
    
    logger.info(f"Transcription complete: {len(segments)} segments")
    return segments


def load_audio_file(audio_path: str) -> np.ndarray:
    import whisper
    audio = whisper.load_audio(audio_path)
    return audio