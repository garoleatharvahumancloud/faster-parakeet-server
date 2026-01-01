import soundfile as sf
import numpy as np
from core.parakeet_stateful_engine import ParakeetStatefulEngine

AUDIO_PATH = "output.wav"   # 16kHz mono wav
CHUNK_MS = 200            # realistic streaming chunk

def main():
    engine = ParakeetStatefulEngine(device="cuda")

    audio, sr = sf.read(AUDIO_PATH, dtype="float32")
    assert sr == 16000

    chunk_size = int(sr * CHUNK_MS / 1000)

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]

        text = engine.process_chunk(chunk)
        if text:
            print(text, end=" ", flush=True)

if __name__ == "__main__":
    main()
