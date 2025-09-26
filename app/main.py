
import os, io, re, sqlite3, uuid, pathlib, datetime
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import piexif

APP_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = APP_DIR / ".." / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "reports.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Road Chat Backend (PoC)")

# ---- CORS (allow everything for PoC; tighten later) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- SQLite setup ----
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            route TEXT,
            milepost REAL,
            issue_type TEXT,
            severity TEXT,
            lane_blocked INTEGER,
            lat REAL,
            lon REAL,
            photo_path TEXT,
            created_at TEXT
        );
    """)
    conn.commit()
    conn.close()

init_db()

# ---- In-memory session state ----
# For PoC only (not persistent). Keyed by session_id.
SESSIONS = {}

ISSUE_CHOICES = ["pothole", "cracking", "shoulder_drop", "guardrail", "sign", "drainage", "debris", "snow_ice"]

def parse_exif_gps(image_bytes: bytes):
    try:
        im = Image.open(io.BytesIO(image_bytes))
        exif_bytes = im.info.get("exif", b"")
        if not exif_bytes:
            return None, None
        exif_dict = piexif.load(exif_bytes)
        gps = exif_dict.get("GPS", {})
        lat_val = gps.get(piexif.GPSIFD.GPSLatitude)
        lon_val = gps.get(piexif.GPSIFD.GPSLongitude)
        lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef, b'N')
        lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef, b'E')
        if not lat_val or not lon_val:
            return None, None
        def to_deg(val):
            d = val[0][0] / val[0][1]
            m = val[1][0] / val[1][1]
            s = val[2][0] / val[2][1]
            return d + m/60 + s/3600
        lat = to_deg(lat_val)
        lon = to_deg(lon_val)
        if lat_ref == b'S':
            lat = -lat
        if lon_ref == b'W':
            lon = -lon
        return lat, lon
    except Exception:
        return None, None

def save_photo(file: UploadFile) -> tuple[str, bytes]:
    ext = pathlib.Path(file.filename).suffix.lower() or ".jpg"
    name = f"{uuid.uuid4().hex}{ext}"
    path = UPLOAD_DIR / name
    data = file.file.read()
    with open(path, "wb") as f:
        f.write(data)
    return str(path), data

def ensure_session(session_id: str):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "route": None,
            "milepost": None,
            "issue_type": None,
            "severity": None,
            "lane_blocked": None,
            "lat": None,
            "lon": None,
            "photo_path": None,
            "step": "start"
        }
    return SESSIONS[session_id]

def parse_route_mp(text: str):
    # Very naive parse. Looks for "mile 23" or "mp 12.4", returns float if found.
    t = text.lower()
    mp = None
    m = re.search(r"(mile|mp)\s*([0-9]+(?:\.[0-9]+)?)", t)
    if m:
        try:
            mp = float(m.group(2))
        except:
            mp = None
    # naive route capture: keep whole text as route for PoC
    route = text.strip()
    return route, mp

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(session_id: str = Form(...), text: Optional[str] = Form(None), photo: Optional[UploadFile] = File(None)):
    s = ensure_session(session_id)

    # If photo uploaded, save it and try EXIF GPS
    if photo is not None:
        path, bytes_data = save_photo(photo)
        s["photo_path"] = path
        lat, lon = parse_exif_gps(bytes_data)
        s["lat"], s["lon"] = lat, lon
        if lat is None or lon is None:
            s["step"] = "ask_location"
            return JSONResponse({"reply": "Thanks for the photo. I couldn’t read its GPS location. What road and nearest milepost or intersection is this?", "done": False})
        else:
            s["step"] = "classify_or_confirm"
            return JSONResponse({"reply": "Photo received ✔️. I detected a location. What is the problem type? (pothole, cracking, shoulder_drop, guardrail, sign, drainage, debris, snow_ice)", "done": False})

    # Handle text depending on state
    if s["step"] in ["start", None]:
        return JSONResponse({"reply": "Please attach a photo of the issue if you can. Otherwise, describe the road and nearest milepost/intersection.", "done": False})

    if s["step"] == "ask_location":
        if not text:
            return JSONResponse({"reply": "Please provide the road name and the nearest milepost or intersection.", "done": False})
        route, mp = parse_route_mp(text)
        s["route"], s["milepost"] = route, mp
        s["step"] = "classify_or_confirm"
        return JSONResponse({"reply": "Got it. What is the problem type? (pothole, cracking, shoulder_drop, guardrail, sign, drainage, debris, snow_ice)", "done": False})

    if s["step"] == "classify_or_confirm":
        if not text:
            return JSONResponse({"reply": "What is the problem type? (pothole, cracking, shoulder_drop, guardrail, sign, drainage, debris, snow_ice)", "done": False})
        t = text.strip().lower().replace(" ", "_")
        match = None
        for ch in ISSUE_CHOICES:
            if ch in t:
                match = ch; break
        s["issue_type"] = match or "unknown"
        s["step"] = "ask_severity"
        return JSONResponse({"reply": "Thanks. How severe is it? (minor, moderate, severe). Is a lane blocked? You can reply like: 'severe, lane not blocked'.", "done": False})

    if s["step"] == "ask_severity":
        if not text:
            return JSONResponse({"reply": "Please rate severity (minor, moderate, severe) and whether a lane is blocked.", "done": False})
        t = text.lower()
        sev = "moderate"
        if "minor" in t: sev = "minor"
        if "severe" in t: sev = "severe"
        s["severity"] = sev
        s["lane_blocked"] = 1 if ("blocked" in t and "not blocked" not in t and "unblocked" not in t and "no" not in t) else 0

        # Save to DB
        rid = str(uuid.uuid4())
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""INSERT INTO reports 
            (id, session_id, route, milepost, issue_type, severity, lane_blocked, lat, lon, photo_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""" ,
            (rid, session_id, s["route"], s["milepost"], s["issue_type"], s["severity"], s["lane_blocked"],
             s["lat"], s["lon"], s["photo_path"],
             datetime.datetime.utcnow().isoformat(timespec="seconds")+"Z"
            ))
        conn.commit()
        conn.close()
        s["step"] = "done"
        route_disp = s["route"] or "(unknown road)"
        mp_disp = s["milepost"] if s["milepost"] is not None else "(unknown MP)"
        return JSONResponse({"reply": f"Logged ✅\nRoute: {route_disp}\nMilepost: {mp_disp}\nIssue: {s['issue_type']}\nSeverity: {s['severity']}\nLane blocked: {'yes' if s['lane_blocked'] else 'no'}", "done": True})

    if s["step"] == "done":
        return JSONResponse({"reply": "Report already completed for this session. You can reset the session in the UI to start a new report.", "done": True})

    return JSONResponse({"reply": "I didn’t catch that. Try attaching a photo or answer the last question.", "done": False})

@app.get("/export.csv")
def export_csv():
    import csv, tempfile
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM reports ORDER BY created_at DESC")
    rows = cur.fetchall()
    headers = [d[0] for d in cur.description]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    with open(tmp.name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    conn.close()
    return FileResponse(tmp.name, filename="reports_export.csv")
