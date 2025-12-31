from core.parakeet_engine import ParakeetEngine

engine = ParakeetEngine()

# 1 second of silence
pcm16 = b"\x00\x00" * 16000

engine.accept_audio(pcm16)

segments = list(engine.stream())

print("Segments:")
for s in segments:
    print(s)

print("OK: engine flow works")
