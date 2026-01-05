import time
import soundfile as sf
from core.parakeet_stateful_transcript import ParakeetStatefulEngine

CHUNK_SEC = 0.2

def main():
    engine = ParakeetStatefulEngine()

    audio, sr = sf.read("output.wav", dtype="float32")
    assert sr == 16000

    chunk_size = int(sr * CHUNK_SEC)

    print("--- Streaming ASR start ---")
    t0 = time.time()

    with open("streaming_transcription.txt", "w", encoding="utf-8") as f:
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            out = engine.process_chunk(chunk)

            if out is None:
                continue

            # SAFE: out is ALWAYS a dict here
            now = time.time() - t0
            print(f"[{now:6.2f}s] {out['partial']}")
            f.write(out["partial"] + " ")
            f.flush()

        # final commit
        final_text = engine.finalize()
        f.write("\n\n=== FINAL TRANSCRIPT ===\n")
        f.write(final_text)

    print("--- Done ---")

if __name__ == "__main__":
    main()
