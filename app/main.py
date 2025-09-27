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
ISSUE_CHOICES = [
    "pothole", "cracking", "shoulder_drop", "guardrail",
    "sign", "drainage", "debris", "snow_ice"
]

def ensure_session(session_id: str):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "route": None,
            "milepost": None,
            "issue_type": None,    # final confirmed label
            "ai_guess": None,      # AI proposed label
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

SEV_WORDS = {"minor":"minor","moderate":"moderate","medium":"moderate","severe":"severe","major":"severe"}

def parse_severity(text: str) -> Optional[str]:
    t = text.lower()
    for k,v in SEV_WORDS.items():
        if k in t:
            return v
    return None

# ✅ Improved: handle plain yes/no and phrases
def parse_lane_blocked(text: str) -> Optional[int]:
    t = text.lower().strip()
    # explicit phrases
    if "lane" in t or "lanes" in t or "traffic" in t:
        if any(w in t for w in ["not blocked","unblocked","open","flowing","moving","clear","no issues"]):
            return 0
        if any(w in t for w in ["blocked","closed","shut","impassable","stopped"]):
            return 1
    # plain yes/no answers (when we're prompting about lanes)
    if t in ["no","nope","nah","negative","n"]:
        return 0
    if t in ["yes","yep","yeah","affirmative","y"]:
        return 1
    return None

def normalize_issue(label: str) -> str:
    """Map free text to our canonical set."""
    lab = (label or "").strip().lower()
    if lab in ISSUE_CHOICES: return lab
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
    if "snow" in lab or "ice" in lab or "slick" in lab:
        return "snow_ice"
    return "unknown"

def prepare_image_b64(image_bytes: bytes) -> str:
    """Downscale & JPEG-encode to keep payload small and consistent."""
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        im.thumbnail((1280, 1280))  # keep aspect ratio
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

def extract_all_from_text(s: dict, text: str):
    """
    Fill any missing fields from a free-form user message.
    - issue_type via normalize_issue
    - route & milepost via parse_route_mp
    - severity via parse_severity
    - lane_blocked via parse_lane_blocked
    """
    if not s.get("issue_type") or s["issue_type"] == "unknown":
        cand = normalize_issue(text)
        if cand != "unknown":
            s["issue_type"] = cand

    route, mp = parse_route_mp(text)
    if route and (not s.get("route")):
        s["route"] = route
    if mp is not None and s.get("milepost") is None:
        s["milepost"] = mp

    sev = parse_severity(text)
    if sev and not s.get("severity"):
        s["severity"] = sev

    lb = parse_lane_blocked(text)
    if lb is not None and s.get("lane_blocked") is None:
        s["lane_blocked"] = lb

def next_missing_fields(s: dict) -> list[str]:
    fields = []
    if not s.get("issue_type") or s["issue_type"] == "unknown":
        fields.append("issue_type")
    if s.get("lat") is None or s.get("lon") is None:
        if not s.get("route"): fields.append("route")
        if s.get("milepost") is None: fields.append("milepost")
    if not s.get("severity"): fields.append("severity")
    if s.get("lane_blocked") is None: fields.append("lane_blocked")
    return fields

def issue_prompt() -> str:
    return (
        "What best describes the issue? "
        "(pothole, cracking, shoulder_drop, guardrail, sign, drainage, debris, snow_ice)"
    )

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

    # PHOTO RECEIVED → detect GPS + AI classify
    if photo is not None:
        path, bytes_data = save_photo(photo)
        s["photo_path"] = path
        lat, lon = parse_exif_gps(bytes_data)
        s["lat"], s["lon"] = lat, lon

        ai_label, raw = classify_issue_from_photo(bytes_data)
        s["ai_guess"] = ai_label

        loc_msg = "I detected a GPS location." if (lat and lon) else "I couldn’t read GPS from the photo."

        if ai_label == "unknown":
            s["step"] = "ask_issue_type"
            payload = {
                "reply": f"Photo received ✔️. {loc_msg}\nI couldn’t confidently classify the issue.\n{issue_prompt()}",
                "done": False
            }
            if DEBUG: payload["model_raw"] = raw
            return JSONResponse(payload)

        # otherwise proceed to confirmation
        s["step"] = "confirm_issue"
        payload = {
            "reply": (
                f"Photo received ✔️. {loc_msg}\n"
                f"I think this looks like: {ai_label}.\n"
                "Is that right? (yes/no, or tell me the correct type)"
            ),
            "done": False
        }
        if DEBUG: payload["model_raw"] = raw
        return JSONResponse(payload)

    # TEXT: always try to extract facts
    if text:
        extract_all_from_text(s, text)

    # CONFIRMATION step (after photo guess)
    if s["step"] == "confirm_issue":
        if text:
            t = text.strip().lower()
            if t in ["yes","y","correct","right","yep","yeah"]:
                s["issue_type"] = s.get("ai_guess") or s.get("issue_type") or "unknown"
            elif t in ["no","n","incorrect","wrong"]:
                s["step"] = "ask_issue_type"
                return JSONResponse({"reply": "Thanks. " + issue_prompt(), "done": False})
            else:
                # free-form correction already normalized above
                if not s.get("issue_type") or s["issue_type"] == "unknown":
                    s["step"] = "ask_issue_type"
                    return JSONResponse({"reply": issue_prompt(), "done": False})
        s["step"] = "ask_details"  # move on

    # ASK USER FOR TYPE (AI couldn't classify OR user said no)
    if s["step"] == "ask_issue_type":
        if not text:
            return JSONResponse({"reply": issue_prompt(), "done": False})
        s["issue_type"] = normalize_issue(text)
        if s["issue_type"] == "unknown":
            return JSONResponse({"reply": "Sorry, I still didn’t catch that.\n" + issue_prompt(), "done": False})
        s["step"] = "ask_details"

    # START (no photo yet)
    if s["step"] in ["start", None]:
        if text:
            s["step"] = "ask_details"
        else:
            return JSONResponse({"reply": "Attach a photo if you can. Otherwise, tell me the issue and the road + milepost/intersection.", "done": False})

    # ASK DETAILS: only ask for what’s missing, save when complete
    if s["step"] in ["ask_details","ask_location","ask_severity"]:
        missing = next_missing_fields(s)
        if not missing:
            # save to DB
            rid = str(uuid.uuid4())
            conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
            cur.execute("""INSERT INTO reports 
                (id, session_id, route, milepost, issue_type, severity, lane_blocked, lat, lon, photo_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (rid, session_id, s.get("route"), s.get("milepost"), s.get("issue_type"),
                 s.get("severity"), s.get("lane_blocked"), s.get("lat"), s.get("lon"),
                 s.get("photo_path"), datetime.datetime.utcnow().isoformat(timespec="seconds")+"Z"))
            conn.commit(); conn.close()
            s["step"] = "done"
            return JSONResponse({
                "reply": (
                    "Logged ✅\n"
                    f"Issue: {s['issue_type'] or 'unknown'}\n"
                    f"Route: {s.get('route') or '(unknown road)'}\n"
                    f"Milepost: {s.get('milepost') if s.get('milepost') is not None else '(unknown MP)'}\n"
                    f"Severity: {s.get('severity') or '(unspecified)'}\n"
                    f"Lane blocked: {'yes' if s.get('lane_blocked') else 'no' if s.get('lane_blocked') == 0 else '(unspecified)'}"
                ),
                "done": True
            })

        asks = []
        if "issue_type" in missing:
            asks.append("the problem type (pothole, cracking, shoulder_drop, guardrail, sign, drainage, debris, snow_ice)")
        if "route" in missing or "milepost" in missing:
            asks.append("the road name and nearest milepost/intersection")
        if "severity" in missing:
            asks.append("severity (minor, moderate, severe)")
        if "lane_blocked" in missing:
            asks.append("whether a lane is blocked")
        return JSONResponse({"reply": "Got it. Could you tell me " + "; and ".join(asks) + "?", "done": False})

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
