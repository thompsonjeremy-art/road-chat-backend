import os, io, re, sqlite3, uuid, pathlib, datetime, base64, logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import piexif
from openai import OpenAI

# ---------- Config ----------
APP_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = APP_DIR / ".." / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "reports.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("road-chat")

# OpenAI client (requires OPENAI_API_KEY env var)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------- App ----------
app = FastAPI(title="Road Chat Backend (PoC)")

# CORS open for PoC
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DB ----------
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

# ---------- Session ----------
SESSIONS = {}
ISSUE_CHOICES = ["pothole", "cracking", "shoulder_drop", "guardrail", "sign", "drainage", "debris", "snow_ice"]

def ensure_session(session_id: str):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "route": None,
            "milepost": None,
            "issue_type": None,        # FINAL confirmed label
            "ai_guess": None,          # AI's proposed label
            "severity": None,
            "lane_blocked": None,
            "lat": None,
            "lon": None,
            "photo_path": None,
            "step": "start"
        }
    return SESSIONS[session_id]

# ---------- Helpers ----------
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
        if lat_ref == b'S': lat = -lat
        if lon_ref == b'W': lon = -lon
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

def parse_route_mp(text: str):
    t = text.lower()
    mp = None
    m = re.search(r"(mile|mp)\s*([0-9]+(?:\.[0-9]+)?)", t)
    if m:
        try: mp = float(m.group(2))
        except: mp = None
    route = text.strip()
    return route, mp

def normalize_issue(label: str) -> str:
    """Map free text to our canonical set."""
    lab = (label or "").strip().lower()
    if lab in ISSUE_CHOICES: return lab
    # common synonyms / phrases
    if "pothole" in lab or "hole" in lab or "pitted" in lab:
        return "pothole"
    if "crack" in lab or "split" in lab or "alligator" in lab:
        return "cracking"
    if "shoulder" in lab or "edge drop" in lab or "edge-drop" in lab or "drop off" in lab:
        return "shoulder_drop"
    if "guard" in lab or "rail" in lab:
        return "guardrail"
    if "sign" in lab or "marker" in lab:
        return "sign"
    if "drain" in lab or "culvert" in lab or "ditch" in lab or "flood" in lab:
        return "drainage"
    if "debris" in lab or "rock" in lab or "tree" in lab or "fallen" in lab:
        return "debris"
    if "snow" in lab or "ice" in lab or "packed" in lab or "slick" in lab:
        return "snow_ice"
    return "unknown"

def prepare_image_b64(image_bytes: bytes) -> str:
    """Downscale & JPEG-encode to keep payload small and consistent."""
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        im.thumbnail((1280, 1280))  # keep aspect ratio; max side 1280px
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return base64.b64encode(image_bytes).decode("utf-8")

def classify_issue_from_photo(photo_bytes: bytes) -> tuple[str, str]:
    """
    Returns (normalized_label, raw_model_reply).
    Uses OpenAI vision with few-shot and deterministic settings.
    """
    b64 = prepare_image_b64(photo_bytes)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=10,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a road maintenance inspector. "
                        "Look only at the roadway in the photo and respond with ONE word from: "
                        "pothole, cracking, shoulder_drop, guardrail, sign, drainage, debris, snow_ice, unknown. "
                        "If unsure, reply unknown. No punctuation."
                    ),
                },
                {"role": "user", "content": "Example: hole in asphalt (often with water) → pothole"},
                {"role": "user", "content": "Example: many thin linear fissures in asphalt → cracking"},
                {"role": "user", "content": "Example: snow or ice covering lanes → snow_ice"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Classify this road photo. One word only."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ],
                },
            ],
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        raw = f"error:{e}"

    label = normalize_issue(raw)
    if DEBUG:
        log.info(f"[VISION] raw='{raw}' -> normalized='{label}'")
    return label, raw

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "debug": DEBUG}

@app.post("/vision_test")
async def vision_test(photo: UploadFile = File(...)):
    """Direct vision test (bypasses chat flow)."""
    path, bytes_data = save_photo(photo)
    lat, lon = parse_exif_gps(bytes_data)
    label, raw = classify_issue_from_photo(bytes_data)
    return {
        "label": label,
        "raw": raw if DEBUG else "(hidden; set DEBUG=true to expose)",
        "has_gps": bool(lat and lon),
        "saved_to": path
    }

@app.post("/chat")
async def chat(
    session_id: str = Form(...),
    text: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
):
    s = ensure_session(session_id)

    # PHOTO RECEIVED → detect GPS + AI classify → ask for confirmation
    if photo is not None:
        path, bytes_data = save_photo(photo)
        s["photo_path"] = path
        lat, lon = parse_exif_gps(bytes_data)
        s["lat"], s["lon"] = lat, lon

        ai_label, raw = classify_issue_from_photo(bytes_data)
        s["ai_guess"] = ai_label
        s["step"] = "confirm_issue"

        loc_msg = "I detected a GPS location." if (lat and lon) else "I couldn’t read GPS from the photo."
        reply = (
            f"Photo received ✔️. {loc_msg}\n"
            f"I think this looks like: {ai_label}.\n"
            "Is that right? (yes/no, or tell me the correct type)"
        )
        payload = {"reply": reply, "done": False}
        if DEBUG:
            payload["model_raw"] = raw
        return JSONResponse(payload)

    # START (no photo yet)
    if s["step"] in ["start", None]:
        return JSONResponse({
            "reply": "Please attach a photo of the issue if you can. Otherwise, tell me the road and nearest milepost/intersection.",
            "done": False
        })

    # CONFIRMATION STEP
    if s["step"] == "confirm_issue":
        if not text:
            return JSONResponse({"reply": "Is my guess correct? (yes/no, or name the correct type)", "done": False})
        t = text.strip().lower()
        if t in ["yes", "y", "correct", "right"]:
            s["issue_type"] = s.get("ai_guess") or "unknown"
        elif t in ["no", "n", "incorrect", "wrong"]:
            s["issue_type"] = "unknown"
        else:
            # user gave a label like "hole in the road" → normalize
            s["issue_type"] = normalize_issue(t)

        # If we still have unknown AND no GPS, ask for location; else severity next
        if s["lat"] is None or s["lon"] is None:
            s["step"] = "ask_location"
            return JSONResponse({"reply": "What road and nearest milepost or intersection is this?", "done": False})
        else:
            s["step"] = "ask_severity"
            return JSONResponse({"reply": "Thanks. How severe is it? (minor, moderate, severe). Is a lane blocked?", "done": False})

    # ASK LOCATION (no GPS in photo)
    if s["step"] == "ask_location":
        if not text:
            return JSONResponse({"reply": "Please provide the road name and nearest milepost or intersection.", "done": False})
        route, mp = parse_route_mp(text)
        s["route"], s["milepost"] = route, mp
        s["step"] = "ask_severity"
        return JSONResponse({"reply": "Got it. How severe is it? (minor, moderate, severe). Is a lane blocked?", "done": False})

    # ASK SEVERITY (final before save)
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
             datetime.datetime.utcnow().isoformat(timespec="seconds")+"Z"))
        conn.commit()
        conn.close()
        s["step"] = "done"
        return JSONResponse({
            "reply": (
                "Logged ✅\n"
                f"Route: {s['route'] or '(unknown road)'}\n"
                f"Milepost: {s['milepost'] or '(unknown MP)'}\n"
                f"Issue: {s['issue_type'] or (s.get('ai_guess') or 'unknown')}\n"
                f"Severity: {s['severity']}\n"
                f"Lane blocked: {'yes' if s['lane_blocked'] else 'no'}"
            ),
            "done": True
        })

    # DONE
    if s["step"] == "done":
        return JSONResponse({"reply": "Report already completed for this session. Reset to start a new one.", "done": True})

    # Fallback
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
        w = csv.writer(f); w.writerow(headers); w.writerows(rows)
    conn.close()
    return FileResponse(tmp.name, filename="reports_export.csv")
