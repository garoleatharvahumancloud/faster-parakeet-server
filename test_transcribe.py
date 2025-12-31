import requests

url = "http://127.0.0.1:8000/v1/audio/transcriptions"
files = {"file": open("output.wav", "rb")}
data = {"model": "whisper-1"}

r = requests.post(url, files=files, data=data)
print(r.json())
