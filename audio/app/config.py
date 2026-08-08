from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    hf_token: str = Field(default="", alias="HF_TOKEN")
    whisper_model: str = Field(default="openai/whisper-large-v3", alias="WHISPER_MODEL")
    diarization_model: str = Field(default="pyannote/speaker-diarization-3.1", alias="DIARIZATION_MODEL")
    whisper_device: str = Field(default="cuda", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="float16", alias="WHISPER_COMPUTE_TYPE")
    diarization_device: str = Field(default="cuda", alias="DIARIZATION_DEVICE")
    max_audio_file_size_mb: int = Field(default=100, alias="MAX_AUDIO_FILE_SIZE_MB")
    max_concurrent_jobs: int = Field(default=1, alias="MAX_CONCURRENT_JOBS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()