from abc import ABC, abstractmethod
from typing import AsyncGenerator
from .models import ASRJob, Segment

class ASREngine(ABC):

    @abstractmethod
    async def transcribe(self, job: ASRJob):
        pass

    @abstractmethod
    async def stream_transcribe(
        self, job: ASRJob
    ) -> AsyncGenerator[Segment, None]:
        pass

    @abstractmethod
    def supports_streaming(self) -> bool:
        pass
