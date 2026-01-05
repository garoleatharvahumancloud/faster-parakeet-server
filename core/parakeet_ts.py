import time
import numpy as np
import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel

class ParakeetStatefulEngine:
    def __init__(
        self,
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        sample_rate=16000,
        device="cuda",
        max_buffer_sec=4.0,
        min_update_sec=0.6,
        encoder_frame_sec=0.02,  # 20ms per encoder frame
    ):
        self.sample_rate = sample_rate
        self.device = device
        self.encoder_frame_sec = encoder_frame_sec

        self.max_samples = int(sample_rate * max_buffer_sec)
        self.min_update_samples = int(sample_rate * min_update_sec)

        self.model = EncDecRNNTBPEModel.from_pretrained(model_name)
        self.model.to(device).eval()

        self.audio_buffer = np.zeros((0,), dtype=np.float32)
        self.samples_since_last = 0

        self.committed_text = ""
        self.committed_tokens = 0

    @torch.no_grad()
    def process_chunk(self, chunk: np.ndarray):
        start = time.time()

        # 1) append audio
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        self.samples_since_last += len(chunk)

        # 2) trim buffer
        if len(self.audio_buffer) > self.max_samples:
            self.audio_buffer = self.audio_buffer[-self.max_samples:]

        # 3) throttle updates
        if self.samples_since_last < self.min_update_samples:
            return None
        self.samples_since_last = 0

        # 4) RNNT decode
        hyps = self.model.transcribe(
            [self.audio_buffer],
            batch_size=1,
            return_hypotheses=True,
        )

        hyp = hyps[0]
        full_text = hyp.text.strip()

        # 5) delta text logic
        if full_text.startswith(self.committed_text):
            delta_text = full_text[len(self.committed_text):].strip()
        else:
            delta_text = full_text  # RNNT revision

        # 6) timestamps (TOKEN‑LEVEL)
        token_ts = hyp.timestamp
        if token_ts is None:
            token_ts = []

        token_ts = token_ts.tolist()

        new_token_ts = token_ts[self.committed_tokens :]
        new_token_ts_sec = [
            round(t * self.encoder_frame_sec, 3) for t in new_token_ts
        ]

        self.committed_text = full_text
        self.committed_tokens = len(token_ts)

        if not delta_text:
            return None

        return {
            "partial": delta_text,
            "full": full_text,
            "token_timestamps_sec": new_token_ts_sec,
            "latency_sec": round(time.time() - start, 3),
        }

    def finalize(self):
        return self.committed_text
