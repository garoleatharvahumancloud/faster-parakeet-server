from core.engine import ASREngine
from core.models import ASRJob, ASRResult, Segment


class StubEngine(ASREngine):

    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, job: ASRJob) -> ASRResult:
        return ASRResult(
            text="hello world",
            language="en",
            segments=[
                Segment(start=0.0, end=1.0, text="hello world")
            ],
        )

    async def stream_transcribe(self, job: ASRJob):
        yield Segment(start=0.0, end=0.5, text="hello")
        yield Segment(start=0.5, end=1.0, text="world")
