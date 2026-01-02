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
        max_buffer_sec=8.0,
        min_update_sec=0.6,
        stability_threshold=2,   # must repeat N times before commit
    ):
        self.sample_rate = sample_rate
        self.device = device

        self.max_samples = int(sample_rate * max_buffer_sec)
        self.min_update_samples = int(sample_rate * min_update_sec)

        self.model = EncDecRNNTBPEModel.from_pretrained(model_name)
        self.model.to(device).eval()

        self.audio_buffer = np.zeros((0,), dtype=np.float32)
        self.samples_since_last = 0

        # streaming state
        self.prev_text = ""
        self.candidate_text = ""
        self.candidate_count = 0
        self.stability_threshold = stability_threshold

    @torch.no_grad()
    def process_chunk(self, chunk: np.ndarray):
        t0 = time.time()

        # 1) append audio
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        self.samples_since_last += len(chunk)

        # 2) trim buffer
        if len(self.audio_buffer) > self.max_samples:
            self.audio_buffer = self.audio_buffer[-self.max_samples :]

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

        text = hyp_to_text(hyps[0]).strip()

        # 5) stability check
        if text == self.candidate_text:
            self.candidate_count += 1
        else:
            self.candidate_text = text
            self.candidate_count = 1

        # 6) commit only when stable
        if self.candidate_count >= self.stability_threshold:
            if text.startswith(self.prev_text):
                delta = text[len(self.prev_text):].strip()
            else:
                delta = text

            self.prev_text = text
            self.candidate_count = 0

            if delta:
                return {
                    "partial": delta,
                    "full": text,
                    "latency_sec": round(time.time() - t0, 3),
                }

        return None
