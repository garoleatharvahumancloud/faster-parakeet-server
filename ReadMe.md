```
faster-parakeet-server/

├─ main.py                   # Entry point for FastAPI server
├─ requirements.txt          # Python dependencies
├─ api/
│  ├─ __init__.py
│  └─ transcriptions.py      # API routes, handles streaming & normal
├─ core/
│  ├─ __init__.py
│  ├─ engine.py              # Abstract ASREngine class
│  ├─ stub_engine.py         # StubEngine for testing
│  └─ models.py              # ASRJob, ASRResult, Segment
└─ utils/
   ├─ __init__.py
   ├─ formatters.py          # JSON formatting helpers
   └─ sse.py                 # Helper for Server-Sent Events (SSE)
   ```
