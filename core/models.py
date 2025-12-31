from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class ASRResult:
    text: str
    language: Optional[str]
    segments: List[Segment]


@dataclass
class ASRJob:
    audio_bytes: bytes
    task: str                 # "transcribe" | "translate"
    model: str
    language: Optional[str]
    stream: bool
    response_format: str
    options: Dict[str, Any]
