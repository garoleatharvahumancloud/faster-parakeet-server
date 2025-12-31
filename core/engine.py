from abc import ABC, abstractmethod
from typing import Iterable, List
from core.models import Segment


class ASREngine(ABC):
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        ...

    @abstractmethod
    def transcribe(self, pcm16: bytes) -> List[Segment]:
        ...

    @abstractmethod
    def stream_transcribe(
        self, pcm16_iter: Iterable[bytes]
    ) -> Iterable[Segment]:
        ...
