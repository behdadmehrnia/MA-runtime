import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def calculate_overlap(seg_start: float, seg_end: float, turn_start: float, turn_end: float) -> float:
    overlap_start = max(seg_start, turn_start)
    overlap_end = min(seg_end, turn_end)
    return max(0.0, overlap_end - overlap_start)


def align_speakers_to_transcript(
    whisper_segments: List[Dict[str, Any]],
    diarization_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not diarization_segments:
        logger.warning("No diarization segments, assigning default speaker")
        for seg in whisper_segments:
            seg["speaker"] = "SPEAKER_00"
        return whisper_segments
    
    if not whisper_segments:
        logger.warning("No whisper segments")
        return []
    
    aligned = []
    
    for whisper_seg in whisper_segments:
        ws = whisper_seg["start"]
        we = whisper_seg["end"]
        
        best_speaker = None
        best_overlap = 0.0
        
        for dia_seg in diarization_segments:
            ds = dia_seg["start"]
            de = dia_seg["end"]
            
            overlap = calculate_overlap(ws, we, ds, de)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dia_seg["speaker"]
        
        if best_speaker is None:
            best_speaker = "SPEAKER_00"
            logger.debug(f"No overlap found for segment {ws:.2f}-{we:.2f}, using default speaker")
        
        aligned.append({
            "start": ws,
            "end": we,
            "speaker": best_speaker,
            "text": whisper_seg["text"],
        })
    
    logger.info(f"Alignment complete: {len(aligned)} speaker-aware segments")
    return aligned