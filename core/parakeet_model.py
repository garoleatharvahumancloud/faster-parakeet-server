import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel

class ParakeetStreamingModel:
    def __init__(self, device="cuda"):
        self.device = device

        self.model = EncDecRNNTBPEModel.from_pretrained(
            "nvidia/parakeet-tdt-0.6b-v3"
        )
        self.model.to(device).eval()

        # THIS is the streaming decoder (critical)
        self.decoder = self.model.decoding.rnnt_decoder
        self.decoder.reset()

    @torch.no_grad()
    def decode(self, audio: torch.Tensor):
        """
        audio: (T,) float32 tensor @16kHz
        """
        audio = audio.unsqueeze(0).to(self.device)
        lengths = torch.tensor([audio.shape[1]], device=self.device)

        encoded, encoded_len = self.model.encoder(audio, lengths)

        results = self.decoder.forward(
            encoder_output=encoded,
            encoded_lengths=encoded_len
        )

        return results
