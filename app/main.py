import os, io, re, sqlite3, uuid, pathlib, datetime, base64
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import piexif
from openai import OpenAI

# Setup paths
APP_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = APP_DIR / ".." / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "reports.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

app = FastAPI(title="Road Chat Backend (PoC)")

# ---- CORS (open for PoC) ----
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
SESSIONS = {}
ISSUE_CHOICES = ["pothole", "cracking", "shoulder_drop", "guardrail", "sign", "drainage", "debris", "snow_ice"]

# ---- Helpers ----
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
    t = text.lower()
    mp = None
    m = re.search(r"(mile|mp)\s*([0-9]+(?:\.[0-9]+)?)", t)
    if m:
        try:
            mp = float(m.group(2))
        except:
            mp = None
    route = text.strip()
    return route, mp

def classify_issue_from_photo(photo_bytes: bytes) -> str:
    """Classify a road issue from a photo using OpenAI Vision.
    Returns exactly one of ISSUE_CHOICES or 'unknown'.
    """
    b64 = base64.b64encode(photo_bytes).decode("utf-8")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",          # vision-capable
            temperature=0,                # deterministic
            max_tokens=10,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a road maintenance inspector. "
                        "When given a photo, respond with EXACTLY one word from this list: "
                        "pothole, cracking, shoulder_drop, guardrail, sign, drainage, debris, snow_ice, unknown. "
                        "If unsure, answer 'unknown'. No sentences, no punctuation."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Classify this road photo. One word only from the list."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }
            ],
        )
        guess = resp.choices[0].message.content.strip().lower()
    except Exception:
        return "unknown"

    # Normalize common variants to our allowed labels
    if guess not in ISSUE_CHOICES:
        if "pothole" in guess:
            guess = "pothole"
        elif "crack" in guess:
            guess = "cracking"
        elif "shoulder" in guess:
            guess = "shoulder_drop"
        elif "guard" in guess:
            guess = "guardrail"
        elif "sign" in guess:
            guess = "sign"
        elif "drain" in guess or "culvert" in guess:
            guess = "drainage"
        elif "debris" in guess or "rock" in guess or "tree" in guess:
            guess = "debris"
        elif "snow" in guess or "ice" in guess:
            guess = "snow_ice"
        else:
            guess = "unknown"

    return guess

# ---- Routes ----
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(session_id: str = Form(...), text: Optional[str] = Form(None), photo: Optional[UploadFile] = File(None)):
    s = ensure_session(session_id)

    # Handle photo upload
    if photo is not None:
        path, bytes_data = save_photo(photo)
        s["photo_path"] = path
        lat, lon = parse_exif_gps(bytes_data)
        s["lat"], s["lon"] = lat, lon

        # Run AI vision model
        guess = classify_issue_from_photo(bytes_data)
        s["issue_type"] = guess
        s["step"] = "ask_severity"

        loc_msg = "I detected a GPS location." if lat and lon else "I couldn’t read GPS from the photo."
        return JSONResponse({
            "reply": f"Photo received ✔️. {loc_msg}\nI think this looks like: {guess}.\nCan you confirm severity? (minor, moderate, severe). Is a lane blocked?",
            "done": False
        })

    # Handle conversation flow
    if s["step"] in ["start", None]:
        return JSONResponse({"reply": "Please attach a photo of the issue or describe the road and nearest milepost/intersection.", "done": False})

    if s["step"] == "ask_location":
        if not text:
            return JSONResponse({"reply": "Please provide the road name and nearest milepost/intersection.", "done": False})
        route, mp = parse_route_mp(text)
        s["route"], s["milepost"] = route, mp
        s["step"] = "ask_severity"
        return JSONResponse({"reply": "Got it. How severe is it? (minor, moderate, severe). Is a lane blocked?", "done": False})

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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (rid, session_id, s["route"], s["milepost"], s["issue_type"], s["severity"], s["lane_blocked"],
             s["lat"], s["lon"], s["photo_path"],
             datetime.datetime.utcnow().isoformat(timespec="seconds")+"Z"
            ))
        conn.commit()
        conn.close()
        s["step"] = "done"
        return JSONResponse({
            "reply": f"Logged ✅\nRoute: {s['route'] or '(unknown road)'}\nMilepost: {s['milepost'] or '(unknown MP)'}\nIssue: {s['issue_type']}\nSeverity: {s['severity']}\nLane blocked: {'yes' if s['lane_blocked'] else 'no'}",
            "done": True
        })

    if s["step"] == "done":
        return JSONResponse({"reply": "Report already completed for this session. Reset to start a new one.", "done": True})

    return JSONResponse({"reply": "I didn’t catch that. Try attaching a photo or answering the last question.", "done": False})

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
