import torch
import numpy as np

class RNNTStreamingDecoder:
    def __init__(self, model):
        self.model = model
        self.model.eval()

        self.encoder_state = None
        self.decoder_state = None

    @torch.no_grad()
    def decode_chunk(self, audio_chunk: np.ndarray):
        """
        TRUE RNNT streaming decode
        """
        audio = torch.tensor(audio_chunk).unsqueeze(0).to(self.model.device)

        enc_out, self.encoder_state = self.model.encoder(
            audio,
            state=self.encoder_state,
            return_state=True,
        )

        results = self.model.decoding.rnnt_decoder_predictions_tensor(
            enc_out,
            decoder_state=self.decoder_state,
            return_hypotheses=True,
        )

        self.decoder_state = results[0].decoder_state
        return results[0].text
