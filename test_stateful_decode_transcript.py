import time
import soundfile as sf
from core.parakeet_stateful_transcript import ParakeetStatefulEngine

CHUNK_SEC = 0.2

def main():
    engine = ParakeetStatefulEngine()

    audio, sr = sf.read("output.wav", dtype="float32")
    assert sr == 16000

    chunk_size = int(sr * CHUNK_SEC)

    with open("streaming_transcription.txt", "w", encoding="utf-8") as f:
        print("--- Streaming ASR start ---")
        t_start = time.time()

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            out = engine.process_chunk(chunk)

            if out:
                t_now = time.time() - t_start
                print(f"[{t_now:6.2f}s] {out['partial']}")
                f.write(out["partial"] + " ")
                f.flush()

    print("--- Done ---")

if __name__ == "__main__":
    main()
