# import numpy as np
# import torch
# from nemo.collections.asr.models import EncDecRNNTBPEModel

# class ParakeetStatefulEngine:
#     def __init__(
#         self,
#         model_name="nvidia/parakeet-tdt-0.6b-v3",
#         sample_rate=16000,
#         device="cuda",
#         max_buffer_sec=8.0,        # keep buffer small → speed
#         min_emit_sec=0.6,          # don't decode too often
#     ):
#         self.sample_rate = sample_rate
#         self.device = device
#         self.max_samples = int(sample_rate * max_buffer_sec)
#         self.min_emit_samples = int(sample_rate * min_emit_sec)

#         self.model = EncDecRNNTBPEModel.from_pretrained(model_name)
#         self.model.to(device).eval()

#         self.audio_buffer = np.zeros((0,), dtype=np.float32)
#         self.last_text = ""
#         self.samples_since_last_decode = 0

#     @torch.no_grad()
#     def process_chunk(self, chunk: np.ndarray) -> str | None:
#         """
#         chunk: float32 mono PCM @ 16kHz
#         returns: newly decoded text OR None
#         """

#         # 1) Append audio
#         self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
#         self.samples_since_last_decode += len(chunk)

#         # 2) Trim buffer (STATEFUL)
#         if len(self.audio_buffer) > self.max_samples:
#             self.audio_buffer = self.audio_buffer[-self.max_samples :]

#         # 3) Throttle decoding (THIS FIXES SLOWNESS)
#         if self.samples_since_last_decode < self.min_emit_samples:
#             return None

#         self.samples_since_last_decode = 0

#         # 4) Transcribe whole buffer
#         hyps = self.model.transcribe(
#             [self.audio_buffer],
#             batch_size=1,
#             return_hypotheses=True,
#         )

#         # RNNT returns Hypothesis
#         text = hyps[0].text.strip()

#         # 5) Emit only new text (diff)
#         if text.startswith(self.last_text):
#             delta = text[len(self.last_text):].strip()
#         else:
#             # model revised earlier words → reset
#             delta = text

#         self.last_text = text
#         return delta if delta else None


# import time
# import numpy as np
# import torch
# from nemo.collections.asr.models import EncDecRNNTBPEModel


# def hyp_to_text(h):
#     if isinstance(h, str):
#         return h
#     return h.text


# class ParakeetStatefulEngine:
#     def __init__(
#         self,
#         model_name="nvidia/parakeet-tdt-0.6b-v3",
#         sample_rate=16000,
#         device="cuda",
#         max_buffer_sec=15.0,
#         min_update_sec=0.6,   # IMPORTANT: throttling
#     ):
#         self.sample_rate = sample_rate
#         self.device = device

#         self.max_samples = int(sample_rate * max_buffer_sec)
#         self.min_update_samples = int(sample_rate * min_update_sec)

#         self.model = EncDecRNNTBPEModel.from_pretrained(model_name)
#         self.model.to(device).eval()

#         self.audio_buffer = np.zeros((0,), dtype=np.float32)
#         self.last_text = ""
#         self.samples_since_last = 0

#     @torch.no_grad()
#     def process_chunk(self, chunk: np.ndarray):
#         t0 = time.time()

#         # 1) Append audio
#         self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
#         self.samples_since_last += len(chunk)

#         # 2) Trim buffer
#         if len(self.audio_buffer) > self.max_samples:
#             self.audio_buffer = self.audio_buffer[-self.max_samples :]

#         # 3) Throttle ASR calls (THIS FIXES SLOWNESS)
#         if self.samples_since_last < self.min_update_samples:
#             return None

#         self.samples_since_last = 0

#         # 4) Transcribe full buffer
#         hyps = self.model.transcribe(
#             [self.audio_buffer],
#             batch_size=1,
#             return_hypotheses=True,
#         )

#         text = hyp_to_text(hyps[0]).strip()

#         # 5) Emit only NEW text
#         if text.startswith(self.last_text):
#             delta = text[len(self.last_text):].strip()
#         else:
#             delta = text

#         self.last_text = text

#         latency = time.time() - t0

#         return {
#             "partial": delta,
#             "full": text,
#             "latency_sec": round(latency, 3),
#         }


import numpy as np
import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel

class ParakeetStatefulEngine:
    def __init__(
        self,
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        sample_rate=16000,
        device="cuda",
        max_buffer_sec=8.0,      # smaller = faster
        min_emit_sec=0.6,        # throttle decoding
    ):
        self.sample_rate = sample_rate
        self.device = device

        self.max_samples = int(sample_rate * max_buffer_sec)
        self.min_emit_samples = int(sample_rate * min_emit_sec)

        self.model = EncDecRNNTBPEModel.from_pretrained(model_name)
        self.model.to(device).eval()

        self.audio_buffer = np.zeros((0,), dtype=np.float32)
        self.samples_since_emit = 0
        self.last_text = ""

    @torch.no_grad()
    def process_chunk(self, chunk: np.ndarray):
        # append audio
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        self.samples_since_emit += len(chunk)

        # trim buffer (STATEFUL)
        if len(self.audio_buffer) > self.max_samples:
            self.audio_buffer = self.audio_buffer[-self.max_samples:]

        # throttle decoding
        if self.samples_since_emit < self.min_emit_samples:
            return None

        self.samples_since_emit = 0

        # transcribe full buffer
        hyps = self.model.transcribe(
            [self.audio_buffer],
            batch_size=1,
            return_hypotheses=True,
        )

        hyp = hyps[0]
        text = hyp.text.strip()

        # emit only new text
        if text.startswith(self.last_text):
            delta = text[len(self.last_text):].strip()
        else:
            delta = text

        self.last_text = text
        return delta if delta else None
