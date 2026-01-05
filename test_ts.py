import time
import soundfile as sf
from core.parakeet_ts import ParakeetStatefulEngine

CHUNK_SEC = 0.2

def main():
    engine = ParakeetStatefulEngine(device="cuda")

    audio, sr = sf.read("output.wav", dtype="float32")
    assert sr == 16000

    chunk_size = int(sr * CHUNK_SEC)

    print("--- Streaming ASR start ---")
    t0 = time.time()

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        out = engine.process_chunk(chunk)

        if out is None:
            continue

        now = time.time() - t0
        print(f"[{now:6.2f}s] {out['partial']}")
        print("   token_ts(sec):", out["token_timestamps_sec"])

    print("\n=== FINAL ===")
    print(engine.finalize())

if __name__ == "__main__":
    main()
