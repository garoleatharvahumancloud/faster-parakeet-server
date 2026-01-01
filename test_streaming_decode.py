# import soundfile as sf
# import numpy as np
# from core.parakeet_stateful_engine import ParakeetStatefulEngine

# AUDIO_PATH = "output.wav"   # 16kHz mono wav
# CHUNK_MS = 200            # realistic streaming chunk

# def main():
#     engine = ParakeetStatefulEngine(device="cuda")

#     audio, sr = sf.read(AUDIO_PATH, dtype="float32")
#     assert sr == 16000

#     chunk_size = int(sr * CHUNK_MS / 1000)

#     for i in range(0, len(audio), chunk_size):
#         chunk = audio[i : i + chunk_size]

#         text = engine.process_chunk(chunk)
#         if text:
#             print(text, end=" ", flush=True)

# if __name__ == "__main__":
#     main()

# import time
# import numpy as np
# import soundfile as sf
# from core.parakeet_stateful_engine import ParakeetStatefulEngine


# def main():
#     engine = ParakeetStatefulEngine(
#         device="cuda",
#         min_update_sec=0.7,   # tweak this (0.5–1.0)
#     )

#     audio, sr = sf.read("output.wav", dtype="float32")
#     assert sr == 16000

#     chunk_size = int(0.1 * sr)  # 100 ms

#     print("\n--- Streaming ASR start ---\n")

#     t_start = time.time()

#     for i in range(0, len(audio), chunk_size):
#         chunk = audio[i : i + chunk_size]

#         result = engine.process_chunk(chunk)
#         if result is None:
#             continue

#         print(
#             f"[{time.time()-t_start:6.2f}s] "
#             f"Δ: {result['partial']} "
#             f"(latency {result['latency_sec']}s)"
#         )

#     print("\n--- Final Transcript ---\n")
#     print(engine.last_text)


# if __name__ == "__main__":
#     main()


import time
import soundfile as sf
import numpy as np
from core.parakeet_stateful_engine import ParakeetStatefulEngine

CHUNK_SEC = 0.2  # 200ms chunks

def main():
    engine = ParakeetStatefulEngine()

    audio, sr = sf.read("output.wav", dtype="float32")
    assert sr == 16000

    chunk_size = int(sr * CHUNK_SEC)

    print("--- Streaming ASR start ---")

    t_start = time.time()

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        out = engine.process_chunk(chunk)

        if out:
            t_now = time.time() - t_start
            print(f"[{t_now:6.2f}s] {out}")

    print("--- Done ---")

if __name__ == "__main__":
    main()
