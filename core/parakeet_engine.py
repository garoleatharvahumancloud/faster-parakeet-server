from typing import Iterable, List
import numpy as np

from core.engine import ASREngine
from core.models import Segment


class ParakeetEngine(ASREngine):
    """
    Stateful, zero-overlap streaming ASR engine.
    Decoder is stubbed for now (flow testing).
    """

    def __init__(self):
        self.sample_rate = 16000
        self._init_model()
        self._init_state()

    # ---------- required ----------

    @property
    def supports_streaming(self) -> bool:
        return True

    # ---------- lifecycle ----------

    def _init_model(self):
        """
        Model loading will go here later.
        For now, keep decoder stubbed.
        """
        self.decoder = None

    def _init_state(self):
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.processed_samples = 0

    # ---------- offline ----------

    def transcribe(self, pcm16: bytes) -> List[Segment]:
        """
        Offline transcription (required by base class).
        Currently stubbed.
        """
        audio = (
            np.frombuffer(pcm16, dtype=np.int16)
            .astype(np.float32)
            / 32768.0
        )

        duration = len(audio) / self.sample_rate

        return [
            Segment(
                start=0.0,
                end=duration,
                text="[stub transcription]",
                final=True,
            )
        ]

    # ---------- streaming ----------

    def stream_transcribe(
        self, pcm16_iter: Iterable[bytes]
    ) -> Iterable[Segment]:
        for pcm16 in pcm16_iter:
            self.accept_audio(pcm16)
            for seg in self.stream():
                yield seg

    def accept_audio(self, pcm16: bytes):
        audio = (
            np.frombuffer(pcm16, dtype=np.int16)
            .astype(np.float32)
            / 32768.0
        )
        self.audio_buffer = np.concatenate(
            [self.audio_buffer, audio]
        )

    def stream(self) -> Iterable[Segment]:
        CHUNK_SAMPLES = int(0.5 * self.sample_rate)  # 500 ms

        while (
            len(self.audio_buffer) - self.processed_samples
            >= CHUNK_SAMPLES
        ):
            chunk = self.audio_buffer[
                self.processed_samples :
                self.processed_samples + CHUNK_SAMPLES
            ]

            segments = self._decode_chunk(chunk)

            self.processed_samples += CHUNK_SAMPLES

            for seg in segments:
                yield seg

    def _decode_chunk(self, audio_chunk: np.ndarray) -> List[Segment]:
        """
        Stub decoder: proves flow, buffering, zero-overlap.
        """
        duration = len(audio_chunk) / self.sample_rate

        return [
            Segment(
                start=0.0,
                end=duration,
                text="[stub stream chunk]",
                final=False,
            )
        ]
