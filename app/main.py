import os, io, re, json, sqlite3, uuid, pathlib, datetime, base64, logging
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
            "meta": {},            # for proposals/debug
            "step": "start"
        }
    return SESSIONS[session_id]

# ---------- Helpers: EXIF / images ----------
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

# ---------- Helpers: parsing & normalization ----------
SEV_WORDS = {"minor":"minor","moderate":"moderate","medium":"moderate","severe":"severe","major":"severe"}

def parse_severity(text: str) -> Optional[str]:
    t = (text or "").lower()
    for k,v in SEV_WORDS.items():
        if k in t:
            return v
    return None

def normalize_severity(label: Optional[str]) -> Optional[str]:
    if not label: return None
    t = label.lower().strip()
    if "minor" in t: return "minor"
    if "moderate" in t or "medium" in t: return "moderate"
    if "severe" in t or "major" in t: return "severe"
    return None

# ✅ Improved lane parsing (supports plain yes/no and phrases)
def parse_lane_blocked(text: str) -> Optional[int]:
    t = (text or "").lower().strip()
    # explicit phrases
    if "lane" in t or "lanes" in t or "traffic" in t:
        if any(w in t for w in ["not blocked","unblocked","open","flowing","moving","clear","no issues"]):
            return 0
        if any(w in t for w in ["lane blocked","lanes blocked","blocked","closed","shut","impassable","stopped"]):
            return 1
    # plain yes/no
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

# ---- Better route/MP parsing (only pick real road names) ----
ROAD_WORDS = r"(?:highway|hwy|road|rd|street|st|avenue|ave|drive|dr|blvd|boulevard|lane|ln|way|pkwy|parkway|court|ct|place|pl|trail|trl|circle|cir)"
ROUTE_TOKEN  = r"[A-Za-z0-9 .'\-]+?"

def normalize_route(s: Optional[str]) -> Optional[str]:
    if not s: return None
    if not re.search(rf"\b{ROAD_WORDS}\b", s, re.IGNORECASE):
        return None
    s = re.sub(r"\s+", " ", s).strip(" .,-")
    s = " ".join(w if w.isupper() and len(w) <= 4 else w.capitalize() for w in s.split())
    return s[:80]

def parse_route_mp(text: str) -> tuple[Optional[str], Optional[float]]:
    t = text.strip()

    # Milepost anywhere
    mp = None
    m_mp = re.search(r"\b(?:mile|mp)\s*([0-9]+(?:\.[0-9]+)?)\b", t, re.IGNORECASE)
    if m_mp:
        try:
            mp = float(m_mp.group(1))
        except:
            mp = None

    route = None

    # Pattern A: “… on/along/near/at <ROUTE> …”
    m_on = re.search(rf"\b(?:on|along|near|at)\s+({ROUTE_TOKEN}\b{ROAD_WORDS}\b)", t, re.IGNORECASE)
    if m_on:
        route = normalize_route(m_on.group(1))
        if route:
            return route, mp

    # Pattern B: “<ROUTE> … mile/mp …”
    m_before_mp = re.search(rf"\b({ROUTE_TOKEN}\b{ROAD_WORDS}\b)[^\.]*?\b(?:mile|mp)\b", t, re.IGNORECASE)
    if m_before_mp:
        route = normalize_route(m_before_mp.group(1))
        if route:
            return route, mp

    # Pattern C: trailing “… at <ROUTE>”
    m_at = re.search(rf"\bat\s+({ROUTE_TOKEN}\b{ROAD_WORDS}\b)\b", t, re.IGNORECASE)
    if m_at:
        route = normalize_route(m_at.group(1))
        if route:
            return route, mp

    return None, mp

# ---------- Vision: issue classification ----------
def classify_issue_from_photo(photo_bytes: bytes) -> tuple[str, str]:
    """Returns (normalized_label, raw_model_reply)."""
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
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raw = f"error:{e}"

    label = normalize_issue(raw)
    if DEBUG:
        log.info(f"[VISION issue] raw='{raw}' -> '{label}'")
    return label, raw

# ---------- NEW: severity & blockage proposals ----------
def parse_severity_from_text(text: str) -> tuple[Optional[str], float]:
    t = (text or "").lower()
    if "severe" in t or "major" in t: return ("severe", 0.9)
    if "moderate" in t or "medium" in t: return ("moderate", 0.8)
    if "minor" in t or "small" in t or "low" in t: return ("minor", 0.8)
    return (None, 0.0)

def parse_blocked_from_text(text: str) -> tuple[Optional[int], float]:
    t = (text or "").lower().strip()
    if any(k in t for k in ["not blocked","unblocked","open","flowing","moving","no blockage"]): return (0, 0.9)
    if any(k in t for k in ["lane blocked","lanes blocked","blocked","closed","shut","impassable","stopped"]): return (1, 0.9)
    if t in ["no","nope","nah","negative","n"]: return (0, 0.9)
    if t in ["yes","yep","yeah","affirmative","y"]: return (1, 0.9)
    return (None, 0.0)

def vision_severity_and_blocked(photo_b64: str) -> dict:
    """
    Returns: {"severity": str|None, "severity_conf": float,
              "blocked": int|None, "blocked_conf": float, "raw": str}
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=80,
            messages=[
                {"role":"system","content":
                 ("You are a road maintenance inspector. "
                  "From the photo, estimate: (1) severity (minor, moderate, severe) "
                  "based on hazard/damage extent/vehicle impact; "
                  "(2) is a traffic lane blocked (yes/no/unknown). "
                  "Return JSON ONLY with keys: severity, severity_conf (0-1), "
                  "blocked (yes/no/unknown), blocked_conf (0-1).")},
                {"role":"user","content":[
                    {"type":"text","text":"Analyze this photo and output JSON only."},
                    {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{photo_b64}"}}
                ]}
            ]
        )
        raw = (resp.choices[0].message.content or "").strip()
        d = json.loads(raw) if raw.startswith("{") else {}
        sev = normalize_severity(d.get("severity"))
        blk = (d.get("blocked","unknown") or "").lower()
        blocked = 1 if blk=="yes" else 0 if blk=="no" else None
        sev_conf = float(d.get("severity_conf", 0))
        blk_conf = float(d.get("blocked_conf", 0))
        return {"severity": sev, "severity_conf": sev_conf, "blocked": blocked, "blocked_conf": blk_conf, "raw": raw}
    except Exception as e:
        return {"severity": None, "severity_conf": 0.0, "blocked": None, "blocked_conf": 0.0, "raw": f"error:{e}"}

def propose_severity_and_blocked(photo_bytes: Optional[bytes], user_text: Optional[str]) -> dict:
    """
    Combines text cues and (if available) vision to propose severity & blocked with confidence.
    Returns: {"severity": str|None, "severity_conf": float,
              "blocked": int|None, "blocked_conf": float, "raw_vision": str|None}
    """
    best_sev, best_sev_conf = None, 0.0
    best_blk, best_blk_conf = None, 0.0

    # 1) text first (cheap)
    ts, ts_conf = parse_severity_from_text(user_text or "")
    tb, tb_conf = parse_blocked_from_text(user_text or "")
    if ts: best_sev, best_sev_conf = ts, ts_conf
    if tb is not None: best_blk, best_blk_conf = tb, tb_conf

    raw_vis = None
    # 2) vision if we have a photo and confidence is low
    if photo_bytes and (best_sev_conf < 0.85 or best_blk_conf < 0.85):
        b64 = prepare_image_b64(photo_bytes)
        v = vision_severity_and_blocked(b64)
        raw_vis = v.get("raw")
        if v.get("severity") and v.get("severity_conf",0) > best_sev_conf:
            best_sev, best_sev_conf = v["severity"], v["severity_conf"]
        if v.get("blocked") is not None and v.get("blocked_conf",0) > best_blk_conf:
            best_blk, best_blk_conf = v["blocked"], v["blocked_conf"]

    return {
        "severity": best_sev, "severity_conf": best_sev_conf,
        "blocked": best_blk, "blocked_conf": best_blk_conf,
        "raw_vision": raw_vis
    }

# ---------- Extract all from text ----------
def extract_all_from_text(s: dict, text: str):
    """
    Fill any missing fields from a free-form user message.
    - issue_type via normalize_issue
    - route & milepost via parse_route_mp
    - severity via parse_severity/parse_severity_from_text
    - lane_blocked via parse_lane_blocked/parse_blocked_from_text
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

    # primary rules
    sev = parse_severity(text)
    if sev and not s.get("severity"):
        s["severity"] = sev

    lb = parse_lane_blocked(text)
    if lb is not None and s.get("lane_blocked") is None:
        s["lane_blocked"] = lb

    # secondary (confidence-scored) text cues
    if not s.get("severity"):
        ts, _ = parse_severity_from_text(text or "")
        if ts: s["severity"] = ts
    if s.get("lane_blocked") is None:
        tb, _ = parse_blocked_from_text(text or "")
        if tb is not None: s["lane_blocked"] = tb

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
    # propose severity/blocked
    sv = propose_severity_and_blocked(photo_bytes=bytes_data, user_text=None)
    return {
        "label": label,
        "raw": raw if DEBUG else "(hidden; set DEBUG=true to expose)",
        "severity_guess": sv["severity"],
        "severity_conf": sv["severity_conf"],
        "blocked_guess": sv["blocked"],
        "blocked_conf": sv["blocked_conf"],
        "sv_raw": sv["raw_vision"] if DEBUG else "(hidden)",
        "has_gps": bool(lat and lon),
        "saved_to": path
    }

@app.post("/chat")
async def chat(
    session_id: str = Form(...),
    text: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
    # Optional future: device lat/lon via form
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
):
    s = ensure_session(session_id)

    # Device-provided coords (optional)
    if lat is not None and lon is not None:
        s["lat"], s["lon"] = float(lat), float(lon)

    # PHOTO RECEIVED → detect GPS + classify + propose severity/blocked
    if photo is not None:
        path, bytes_data = save_photo(photo)
        s["photo_path"] = path
        exif_lat, exif_lon = parse_exif_gps(bytes_data)
        if exif_lat and exif_lon:
            s["lat"], s["lon"] = exif_lat, exif_lon

        ai_label, raw = classify_issue_from_photo(bytes_data)
        s["ai_guess"] = ai_label

        # Proposals
        sv = propose_severity_and_blocked(photo_bytes=bytes_data, user_text=None)
        s["meta"]["sv"] = sv
        if sv["severity"] and sv["severity_conf"] >= 0.75 and not s.get("severity"):
            s["severity"] = sv["severity"]
        if sv["blocked"] is not None and sv["blocked_conf"] >= 0.75 and s.get("lane_blocked") is None:
            s["lane_blocked"] = sv["blocked"]

        loc_msg = "I detected a GPS location." if (s.get("lat") and s.get("lon")) else "I couldn’t read GPS from the photo."

        if ai_label == "unknown":
            s["step"] = "ask_issue_type"
            more = ""
            if sv.get("severity"):
                more += f"\n(Severity looks {sv['severity']} ~{sv['severity_conf']:.2f})"
            if sv.get("blocked") is not None:
                yn = "yes" if sv["blocked"]==1 else "no"
                more += f"\n(Lane blocked looks {yn} ~{sv['blocked_conf']:.2f})"
            payload = {
                "reply": f"Photo received ✔️. {loc_msg}\nI couldn’t confidently classify the issue.{more}\n{issue_prompt()}",
                "done": False
            }
            if DEBUG:
                payload["model_raw"] = raw
                if sv.get("raw_vision"): payload["sv_raw"] = sv["raw_vision"]
            return JSONResponse(payload)

        # otherwise proceed to confirmation
        s["step"] = "confirm_issue"
        extra = ""
        if not s.get("severity") and sv.get("severity"):
            extra += f"\nI estimate severity: {sv['severity']} ({sv['severity_conf']:.2f})."
        if s.get("lane_blocked") is None and sv.get("blocked") is not None:
            yn = "yes" if sv["blocked"]==1 else "no"
            extra += f"\nI think a lane is blocked: {yn} ({sv['blocked_conf']:.2f})."

        payload = {
            "reply": (
                f"Photo received ✔️. {loc_msg}\n"
                f"I think this looks like: {ai_label}.{extra}\n"
                "Is that right? (yes/no, or tell me the correct type)"
            ),
            "done": False
        }
        if DEBUG:
            payload["model_raw"] = raw
            if sv.get("raw_vision"): payload["sv_raw"] = sv["raw_vision"]
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
                # Apply severity proposal if still missing
                sv = (s.get("meta") or {}).get("sv") or {}
                if not s.get("severity") and sv.get("severity") and sv.get("severity_conf",0) >= 0.5:
                    s["severity"] = sv["severity"]
                if s.get("lane_blocked") is None and sv.get("blocked") is not None and sv.get("blocked_conf",0) >= 0.5:
                    s["lane_blocked"] = sv["blocked"]
            elif t in ["no","n","incorrect","wrong"]:
                s["step"] = "ask_issue_type"
                return JSONResponse({"reply": "Thanks. " + issue_prompt(), "done": False})
            else:
                # free-form correction already normalized by extract_all_from_text above
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
            mp_txt = f"MP {round(s['milepost'], 2)}" if s.get('milepost') is not None else "(MP unknown)"
            route_txt = s.get('route') or '(unknown road)'
            return JSONResponse({
                "reply": (
                    "Logged ✅\n"
                    f"Issue: {s['issue_type'] or 'unknown'}\n"
                    f"Route: {route_txt} {mp_txt}\n"
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
            sv = (s.get("meta") or {}).get("sv") or {}
            if sv.get("severity"):
                asks.append(f"severity (minor, moderate, severe) — I’d guess {sv['severity']}")
            else:
                asks.append("severity (minor, moderate, severe)")
        if "lane_blocked" in missing:
            sv = (s.get("meta") or {}).get("sv") or {}
            if sv.get("blocked") is not None:
                yn = "yes" if sv["blocked"]==1 else "no"
                asks.append(f"whether a lane is blocked — I’d guess {yn}")
            else:
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