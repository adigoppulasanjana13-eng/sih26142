# GeoSpecSR Backend

FastAPI prototype backend.

## Run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --reload
```

Health check:
http://127.0.0.1:8000/health

The `/process` endpoint currently returns a prototype response. Replace its body with the real preprocessing and trained super-resolution inference later.
