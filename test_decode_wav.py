import time
import soundfile as sf
from core.parakeet_engine import ParakeetEngine


def chunk_audio(audio, sr, chunk_ms=500):
    chunk_size = int(sr * chunk_ms / 1000)

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]

        # Skip tiny tail chunks (prevents normalize_batch crash)
        if len(chunk) < 320:
            continue

        yield chunk


def main():
    wav_path = "output.wav"
    audio, sr = sf.read(wav_path)

    engine = ParakeetEngine(device="cuda")

    transcript = []
    start_total = time.time()

    for idx, chunk in enumerate(chunk_audio(audio, sr)):
        start_chunk = time.time()

        for text in engine.stream_transcribe([chunk], sr):
            if text:
                transcript.append(text)

        print(f"Chunk {idx} took {time.time() - start_chunk:.3f}s")

    total_time = time.time() - start_total

    final_text = " ".join(transcript)

    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(final_text)
        f.write("\n\n")
        f.write(f"Total time: {total_time:.2f}s\n")

    print("\n=== FINAL TRANSCRIPT ===")
    print(final_text)
    print(f"\nTotal time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
