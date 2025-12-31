from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from core.models import ASRJob
from core.stub_engine import StubEngine
from utils.formatters import format_response
from core.parakeet_engine import ParakeetEngine


from utils.sse import sse_event

router = APIRouter()   # ✅ THIS WAS MISSING
# engine = StubEngine()
engine = ParakeetEngine()

@router.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str | None = Form(None),
    response_format: str = Form("json"),
    stream: bool = Form(False),
):
    audio_bytes = await file.read()

    job = ASRJob(
        audio_bytes=audio_bytes,
        task="transcribe",
        model=model,
        language=language,
        stream=stream,
        response_format=response_format,
        options={},
    )

    if stream:
        if not engine.supports_streaming():
            raise HTTPException(400, "Streaming not supported")

        async def event_generator():
            async for seg in engine.stream_transcribe(job):
                yield sse_event({
                    "type": "segment",
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "final": True,
                })
            yield sse_event({"type": "done"})

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
        )

    result = await engine.transcribe(job)
    return format_response(result, response_format)
