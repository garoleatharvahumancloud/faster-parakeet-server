import numpy as np
import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel

class ParakeetStatefulEngine:
    def __init__(
        self,
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        sample_rate=16000,
        device="cuda",
        max_buffer_sec=8.0,        # keep buffer small → speed
        min_emit_sec=0.6,          # don't decode too often
    ):
        self.sample_rate = sample_rate
        self.device = device
        self.max_samples = int(sample_rate * max_buffer_sec)
        self.min_emit_samples = int(sample_rate * min_emit_sec)

        self.model = EncDecRNNTBPEModel.from_pretrained(model_name)
        self.model.to(device).eval()

        self.audio_buffer = np.zeros((0,), dtype=np.float32)
        self.last_text = ""
        self.samples_since_last_decode = 0

    @torch.no_grad()
    def process_chunk(self, chunk: np.ndarray) -> str | None:
        """
        chunk: float32 mono PCM @ 16kHz
        returns: newly decoded text OR None
        """

        # 1) Append audio
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        self.samples_since_last_decode += len(chunk)

        # 2) Trim buffer (STATEFUL)
        if len(self.audio_buffer) > self.max_samples:
            self.audio_buffer = self.audio_buffer[-self.max_samples :]

        # 3) Throttle decoding (THIS FIXES SLOWNESS)
        if self.samples_since_last_decode < self.min_emit_samples:
            return None

        self.samples_since_last_decode = 0

        # 4) Transcribe whole buffer
        hyps = self.model.transcribe(
            [self.audio_buffer],
            batch_size=1,
            return_hypotheses=True,
        )

        # RNNT returns Hypothesis
        text = hyps[0].text.strip()

        # 5) Emit only new text (diff)
        if text.startswith(self.last_text):
            delta = text[len(self.last_text):].strip()
        else:
            # model revised earlier words → reset
            delta = text

        self.last_text = text
        return delta if delta else None
