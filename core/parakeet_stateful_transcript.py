import time
import numpy as np
import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel

def hyp_to_text(h):
    return h.text if hasattr(h, "text") else str(h)

class ParakeetStatefulEngine:
    def __init__(
        self,
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        sample_rate=16000,
        device="cuda",
        max_buffer_sec=12.0,
        min_update_sec=0.5,
    ):
        self.sample_rate = sample_rate
        self.device = device

        self.max_samples = int(sample_rate * max_buffer_sec)
        self.min_update_samples = int(sample_rate * min_update_sec)

        self.model = EncDecRNNTBPEModel.from_pretrained(model_name)
        self.model.to(device).eval()

        self.audio_buffer = np.zeros((0,), dtype=np.float32)
        self.samples_since_last = 0

        # committed transcript
        self.committed_text = ""

    @torch.no_grad()
    def process_chunk(self, chunk: np.ndarray):
        start = time.time()

        # 1) append audio
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        self.samples_since_last += len(chunk)

        # 2) trim buffer
        if len(self.audio_buffer) > self.max_samples:
            self.audio_buffer = self.audio_buffer[-self.max_samples:]

        # 3) throttle
        if self.samples_since_last < self.min_update_samples:
            return None

        self.samples_since_last = 0

        # 4) transcribe
        hyps = self.model.transcribe(
            [self.audio_buffer],
            batch_size=1,
            return_hypotheses=True,
        )

        full_text = hyp_to_text(hyps[0]).strip()

        # 5) compute delta (NO DROPPED WORDS)
        if full_text.startswith(self.committed_text):
            delta = full_text[len(self.committed_text):].strip()
        else:
            # RNNT revision → re‑emit everything
            delta = full_text

        self.committed_text = full_text

        if not delta:
            return None

        return {
            "partial": delta,
            "full": full_text,
            "latency_sec": round(time.time() - start, 3),
        }

    def finalize(self):
        """Call once at end of stream"""
        return self.committed_text
