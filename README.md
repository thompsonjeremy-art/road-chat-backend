
# Road Chat Backend (PoC)

FastAPI app with a `/chat` endpoint that:
- accepts photo + text,
- extracts EXIF GPS if present,
- asks minimal follow-ups (location → issue type → severity),
- stores a row in SQLite (`data/reports.db`),
- lets you download a CSV at `/export.csv`.

## Run locally
```bash
cd road_chat_backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
```

## Deploy on Render (Web Service)
- Environment: Python 3.11
- Build Command:
```
pip install -r requirements.txt
```
- Start Command:
```
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Notes
- CORS is open (`*`) for PoC; restrict in production.
- Session state is in-memory; persist later if needed.
- Route/milepost snapping not implemented yet (PostGIS later).
