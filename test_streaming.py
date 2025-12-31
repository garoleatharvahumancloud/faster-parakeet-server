import requests

url = "http://127.0.0.1:8000/v1/audio/transcriptions"

files = {"file": open("output.wav", "rb")}
data = {"model": "whisper-1", "stream": "true"}

with requests.post(url, files=files, data=data, stream=True) as r:
    for line in r.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            if decoded.startswith("data: "):
                print(decoded[6:])
