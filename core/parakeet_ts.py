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
        encoder_frame_sec=0.02,  # 20ms
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
        self.committed_token_count = 0

    @torch.no_grad()
    def process_chunk(self, chunk: np.ndarray):
        start = time.time()

        # 1) buffer audio
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        self.samples_since_last += len(chunk)

        if len(self.audio_buffer) > self.max_samples:
            self.audio_buffer = self.audio_buffer[-self.max_samples:]

        if self.samples_since_last < self.min_update_samples:
            return None
        self.samples_since_last = 0

        # 2) decode
        hyps = self.model.transcribe(
            [self.audio_buffer],
            batch_size=1,
            return_hypotheses=True,
        )

        hyp = hyps[0]
        full_text = hyp.text.strip()

        # 3) delta text
        if full_text.startswith(self.committed_text):
            delta_text = full_text[len(self.committed_text):].strip()
        else:
            delta_text = full_text  # RNNT revision

        # 4) token timestamps (SAFE)
        token_ts = []
        if hasattr(hyp, "timestamp") and hyp.timestamp is not None:
            token_ts = [
                int(t) for t in hyp.timestamp.tolist()
                if t is not None
            ]

        new_frames = token_ts[self.committed_token_count:]
        new_words = self._frames_to_words(delta_text, new_frames)

        self.committed_text = full_text
        self.committed_token_count = len(token_ts)

        if not delta_text:
            return None

        return {
            "partial": delta_text,
            "full": full_text,
            "words": new_words,
            "latency_sec": round(time.time() - start, 3),
        }

    def finalize(self):
        return self.committed_text

    def _frames_to_words(self, text, frames):
        """
        Converts token frame indices to word timestamps.
        Defensive: works even with missing frames.
        """
        words = text.split()
        if not frames or not words:
            return []

        # spread frames across words (best possible heuristic)
        step = max(1, len(frames) // len(words))
        out = []

        idx = 0
        for w in words:
            if idx >= len(frames):
                break

            start_f = frames[idx]
            end_f = frames[min(idx + step - 1, len(frames) - 1)]

            out.append({
                "word": w,
                "start": round(start_f * self.encoder_frame_sec, 3),
                "end": round(end_f * self.encoder_frame_sec, 3),
            })
            idx += step

        return out
