import torch
import numpy as np
from nemo.collections.asr.models import EncDecRNNTBPEModel


class ParakeetEngine:
    def __init__(self, device="cuda"):
        self.device = device

        self.model = EncDecRNNTBPEModel.from_pretrained(
            "nvidia/parakeet-tdt-0.6b-v3"
        )
        self.model.to(self.device)
        self.model.eval()

        # Decoder state for streaming
        self._decoder_state = None

    @torch.no_grad()
    def stream_transcribe(self, audio_chunks, sample_rate=16000):
        """
        audio_chunks: iterable of 1D float32 numpy arrays
        """
        for chunk in audio_chunks:
            if len(chunk) == 0:
                continue

            # NeMo expects list of audio arrays
            hyps = self.model.transcribe(
                audio=[chunk],
                batch_size=1,
                return_hypotheses=True
            )

            hyp = hyps[0]

            if hyp.text:
                yield hyp.text
