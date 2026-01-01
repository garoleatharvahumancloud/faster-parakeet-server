from abc import ABC, abstractmethod
from typing import Iterable
from core.models import Segment

class ASREngine(ABC):

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        pass

    @abstractmethod
    def transcribe(self, audio: bytes) -> str:
        pass

    @abstractmethod
    def stream_transcribe(self, audio: bytes) -> Iterable[Segment]:
        pass
