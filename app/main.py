import os, io, re, json, sqlite3, uuid, pathlib, datetime, base64, logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import piexif
from openai import OpenAI

# ---------- Paths / config ----------
APP_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = APP_DIR / ".." / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "reports.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("road-chat")

# OpenAI (requires OPENAI_API_KEY)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------- FastAPI ----------
app = FastAPI(title="Road Issue Reporter (PoC)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
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
    conn.commit(); conn.close()
init_db()

# ---------- Sessions ----------
SESSIONS = {}
ISSUE_CHOICES = ["pothole","cracking","shoulder_drop","guardrail","sign","drainage","debris","snow_ice"]

def ensure_session(session_id: str):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "route": None,
            "milepost": None,
            "issue_type": None,   # final
            "ai_guess": None,     # vision type guess
            "severity": None,
            "lane_blocked": None, # 1/0
            "lat": None, "lon": None,
            "photo_path": None,
            "meta": {},           # stash proposals/raw
            "step": "start"
        }
    return SESSIONS[session_id]

# ---------- Image & EXIF ----------
def parse_exif_gps(image_bytes: bytes):
    try:
        im = Image.open(io.BytesIO(image_bytes))
        exif_bytes = im.info.get("exif", b"")
        if not exif_bytes: return None, None
        exif = piexif.load(exif_bytes).get("GPS", {})
        lat_val, lon_val = exif.get(piexif.GPSIFD.GPSLatitude), exif.get(piexif.GPSIFD.GPSLongitude)
        lat_ref, lon_ref = exif.get(piexif.GPSIFD.GPSLatitudeRef, b'N'), exif.get(piexif.GPSIFD.GPSLongitudeRef, b'E')
        if not lat_val or not lon_val: return None, None
        def to_deg(v):
            d=v[0][0]/v[0][1]; m=v[1][0]/v[1][1]; s=v[2][0]/v[2][1]; return d+m/60+s/3600
        lat, lon = to_deg(lat_val), to_deg(lon_val)
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
    with open(path, "wb") as f: f.write(data)
    return str(path), data

def prepare_image_b64(image_bytes: bytes) -> str:
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        im.thumbnail((1280,1280))
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return base64.b64encode(image_bytes).decode("utf-8")

# ---------- Parsing / normalization ----------
SEV_WORDS = {"minor":"minor","moderate":"moderate","medium":"moderate","severe":"severe","major":"severe"}

def normalize_issue(label: str) -> str:
    lab = (label or "").strip().lower()
    if lab in ISSUE_CHOICES: return lab
    if "pothole" in lab or "hole" in lab: return "pothole"
    if "crack" in lab or "alligator" in lab: return "cracking"
    if "shoulder" in lab or "edge drop" in lab or "drop off" in lab: return "shoulder_drop"
    if "guard" in lab and "rail" in lab: return "guardrail"
    if "sign" in lab or "marker" in lab: return "sign"
    if "drain" in lab or "culvert" in lab or "flood" in lab: return "drainage"
    if "debris" in lab or "tree" in lab or "rock" in lab or "fallen" in lab: return "debris"
    if "snow" in lab or "ice" in lab or "slick" in lab: return "snow_ice"
    return "unknown"

def normalize_severity(label: Optional[str]) -> Optional[str]:
    if not label: return None
    t = label.lower().strip()
    if "minor" in t: return "minor"
    if "moderate" in t or "medium" in t: return "moderate"
    if "severe" in t or "major" in t: return "severe"
    return None

def parse_severity(text: str) -> Optional[str]:
    t = (text or "").lower()
    for k,v in SEV_WORDS.items():
        if k in t: return v
    return None

# Lane blocked: understand yes/no and phrasing
def parse_lane_blocked(text: str) -> Optional[int]:
    t = (text or "").lower().strip()
    if "lane" in t or "lanes" in t or "traffic" in t:
        if any(w in t for w in ["not blocked","unblocked","open","flowing","moving","clear","no issues"]): return 0
        if any(w in t for w in ["lane blocked","lanes blocked","blocked","closed","shut","impassable","stopped"]): return 1
    if t in ["no","nope","nah","negative","n"]: return 0
    if t in ["yes","yep","yeah","affirmative","y"]: return 1
    return None

# Route + milepost (smarter)
ROAD_WORDS = r"(?:highway|hwy|road|rd|street|st|avenue|ave|drive|dr|blvd|boulevard|lane|ln|way|pkwy|parkway|court|ct|place|pl|trail|trl|circle|cir)"
ROUTE_TOKEN  = r"[A-Za-z0-9 .'\-]+?"

def normalize_route(s: Optional[str]) -> Optional[str]:
    if not s: return None
    if not re.search(rf"\b{ROAD_WORDS}\b", s, re.IGNORECASE): return None
    s = re.sub(r"\s+"," ",s).strip(" .,-")
    s = " ".join(w if w.isupper() and len(w)<=4 else w.capitalize() for w in s.split())
    return s[:80]

def parse_route_mp(text: str) -> tuple[Optional[str], Optional[float]]:
    t = text.strip()
    mp = None
    m_mp = re.search(r"\b(?:mile|mp)\s*([0-9]+(?:\.[0-9]+)?)\b", t, re.IGNORECASE)
    if m_mp:
        try: mp = float(m_mp.group(1))
        except: mp = None
    route = None
    m_on = re.search(rf"\b(?:on|along|near|at)\s+({ROUTE_TOKEN}\b{ROAD_WORDS}\b)", t, re.IGNORECASE)
    if m_on: 
        route = normalize_route(m_on.group(1)); 
        if route: return route, mp
    m_before = re.search(rf"\b({ROUTE_TOKEN}\b{ROAD_WORDS}\b)[^\.]*?\b(?:mile|mp)\b", t, re.IGNORECASE)
    if m_before:
        route = normalize_route(m_before.group(1)); 
        if route: return route, mp
    m_at = re.search(rf"\bat\s+({ROUTE_TOKEN}\b{ROAD_WORDS}\b)\b", t, re.IGNORECASE)
    if m_at:
        route = normalize_route(m_at.group(1)); 
        if route: return route, mp
    return None, mp

# ---------- Unified vision extractor (type + severity + blocked) ----------
def vision_extract_all(photo_bytes: bytes) -> dict:
    """
    Returns:
      {
        "type":        pothole|cracking|shoulder_drop|guardrail|sign|drainage|debris|snow_ice|unknown,
        "type_conf":   0..1,
        "severity":    minor|moderate|severe|unknown,
        "severity_conf":0..1,
        "blocked":     yes|no|partial|unknown,
        "blocked_conf":0..1,
        "notes":       str
      }
    """
    b64 = prepare_image_b64(photo_bytes)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=220,
            messages=[
                {"role":"system","content":
                 ("You are a road maintenance inspector. Look ONLY at the photo.\n"
                  "Output strict JSON (no code-fence) with keys:\n"
                  "type (pothole|cracking|shoulder_drop|guardrail|sign|drainage|debris|snow_ice|unknown),\n"
                  "type_conf (0-1), severity (minor|moderate|severe|unknown), severity_conf (0-1),\n"
                  "blocked (yes|no|partial|unknown), blocked_conf (0-1), notes.\n"
                  "Rules:\n"
                  "- Tree/log or large object across full carriageway => type=debris, severity=severe, blocked=yes.\n"
                  "- Object occupies part of a lane but vehicles can pass cautiously => blocked=partial, severity=moderate.\n"
                  "- Small item on shoulder => blocked=no, severity=minor.\n"
                  "- Widespread cracking => cracking; depth/extent influences severity.\n"
                  "- Deep wheel-damaging hole => pothole, severity=severe.\n"
                  "- Standing water across a lane => drainage, severity=moderate (blocked=partial if lane compromised); across full lane(s) => severe (blocked=yes).\n"
                  "- Snow/ice covering travel lanes => snow_ice; impassable => blocked=yes.\n"
                  "- If uncertain, choose 'unknown' and keep confidences ≤ 0.5.\n"
                  "Return ONLY the JSON object."
                 )},
                # Few-shot hints (for biasing, dummy images)
                {"role":"user","content":[
                    {"type":"text","text":"Example: Large tree fully across a two-lane road."},
                    {"type":"image_url","image_url":{"url":"https://picsum.photos/seed/roadtree/400/220"}}
                ]},
                {"role":"assistant","content":"{\"type\":\"debris\",\"type_conf\":0.95,\"severity\":\"severe\",\"severity_conf\":0.95,\"blocked\":\"yes\",\"blocked_conf\":0.95,\"notes\":\"tree across entire road\"}"},
                {"role":"user","content":[
                    {"type":"text","text":"Example: Small branch on shoulder; cars pass unhindered."},
                    {"type":"image_url","image_url":{"url":"https://picsum.photos/seed/branch/400/220"}}
                ]},
                {"role":"assistant","content":"{\"type\":\"debris\",\"type_conf\":0.85,\"severity\":\"minor\",\"severity_conf\":0.8,\"blocked\":\"no\",\"blocked_conf\":0.9,\"notes\":\"small object on shoulder\"}"},
                {"role":"user","content":[
                    {"type":"text","text":"Analyze this photo and return JSON only."},
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
                ]}
            ]
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw) if raw.startswith("{") else {}
    except Exception as e:
        data = {"type":"unknown","type_conf":0.0,"severity":"unknown","severity_conf":0.0,"blocked":"unknown","blocked_conf":0.0,"notes":f"error:{e}"}

    out = {
        "type": normalize_issue(data.get("type") or "unknown"),
        "type_conf": float(data.get("type_conf", 0) or 0),
        "severity": normalize_severity(data.get("severity") or "unknown") or "unknown",
        "severity_conf": float(data.get("severity_conf", 0) or 0),
        "blocked": (data.get("blocked") or "unknown").lower(),
        "blocked_conf": float(data.get("blocked_conf", 0) or 0),
        "notes": data.get("notes") or ""
    }
    # clamps
    for k in ["type_conf","severity_conf","blocked_conf"]:
        out[k] = max(0.0, min(1.0, out[k]))
    if DEBUG: log.info(f"[VISION all] {out}")
    return out

# ---------- Text extraction ----------
def extract_all_from_text(s: dict, text: str):
    if not s.get("issue_type") or s["issue_type"] == "unknown":
        cand = normalize_issue(text)
        if cand != "unknown": s["issue_type"] = cand
    route, mp = parse_route_mp(text)
    if route and not s.get("route"): s["route"] = route
    if mp is not None and s.get("milepost") is None: s["milepost"] = mp
    sev = parse_severity(text)
    if sev and not s.get("severity"): s["severity"] = sev
    lb = parse_lane_blocked(text)
    if lb is not None and s.get("lane_blocked") is None: s["lane_blocked"] = lb

def next_missing_fields(s: dict) -> list[str]:
    missing = []
    if not s.get("issue_type") or s["issue_type"] == "unknown": missing.append("issue_type")
    if s.get("lat") is None or s.get("lon") is None:
        if not s.get("route"): missing.append("route")
        if s.get("milepost") is None: missing.append("milepost")
    if not s.get("severity"): missing.append("severity")
    if s.get("lane_blocked") is None: missing.append("lane_blocked")
    return missing

def issue_prompt() -> str:
    return "What best describes the issue? (pothole, cracking, shoulder_drop, guardrail, sign, drainage, debris, snow_ice)"

# ---------- Endpoints ----------
@app.get("/health")
def health(): return {"status":"ok","debug":DEBUG}

@app.post("/vision_test")
async def vision_test(photo: UploadFile = File(...)):
    path, bytes_data = save_photo(photo)
    lat, lon = parse_exif_gps(bytes_data)
    vision = vision_extract_all(bytes_data)
    return {
        "vision": vision,
        "has_gps": bool(lat and lon),
        "saved_to": path
    }

@app.post("/chat")
async def chat(
    session_id: str = Form(...),
    text: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
):
    s = ensure_session(session_id)

    # Device-provided coords (optional)
    if lat is not None and lon is not None:
        s["lat"], s["lon"] = float(lat), float(lon)

    # ---- PHOTO path ----
    if photo is not None:
        path, bytes_data = save_photo(photo)
        s["photo_path"] = path
        exif_lat, exif_lon = parse_exif_gps(bytes_data)
        if exif_lat and exif_lon: s["lat"], s["lon"] = exif_lat, exif_lon

        vision = vision_extract_all(bytes_data)
        s["meta"]["vision_all"] = vision
        s["ai_guess"] = vision.get("type") or "unknown"

        # Hard override for obvious contradictions (e.g., notes mention tree across road)
        notes = (vision.get("notes") or "").lower()
        if "tree" in notes or "log" in notes:
            if "across" in notes or "across road" in notes or "full width" in notes:
                vision["blocked"] = "yes"; vision["blocked_conf"] = max(vision.get("blocked_conf",0), 0.9)
                if vision.get("severity") == "unknown":
                    vision["severity"] = "severe"; vision["severity_conf"] = max(vision.get("severity_conf",0), 0.85)

        # Auto-fill if confident
        if not s.get("issue_type") and vision["type"] != "unknown" and vision["type_conf"] >= 0.75:
            s["issue_type"] = vision["type"]
        if not s.get("severity") and vision["severity"] != "unknown" and vision["severity_conf"] >= 0.75:
            s["severity"] = vision["severity"]
        if s.get("lane_blocked") is None:
            blk, bc = vision["blocked"], vision["blocked_conf"]
            if blk in ["yes","partial"] and bc >= 0.75: s["lane_blocked"] = 1
            elif blk == "no" and bc >= 0.75: s["lane_blocked"] = 0

        loc_msg = "I detected a GPS location." if (s.get('lat') and s.get('lon')) else "I couldn’t read GPS from the photo."

        # Build proposals line
        props = []
        if vision["type"] != "unknown": props.append(f"type **{vision['type']}** ({vision['type_conf']:.2f})")
        if vision["severity"] != "unknown": props.append(f"severity **{vision['severity']}** ({vision['severity_conf']:.2f})")
        if vision["blocked"] != "unknown":
            yn = "yes" if vision["blocked"] in ["yes","partial"] else "no"
            props.append(f"lane blocked **{yn}** ({vision['blocked_conf']:.2f})")
        prose = "I think this looks like: " + ", ".join(props) + "." if props else "I couldn’t confidently classify the issue."

        if not s.get("issue_type"):
            s["step"] = "ask_issue_type"
            return JSONResponse({"reply": f"Photo received ✔️. {loc_msg}\n{prose}\n{issue_prompt()}", "done": False})

        s["step"] = "confirm_issue"
        return JSONResponse({"reply": f"Photo received ✔️. {loc_msg}\n{prose}\nIs that right? (yes/no, or tell me the correct type)", "done": False})

    # ---- TEXT path ----
    if text: extract_all_from_text(s, text)

    # Confirmation step
    if s["step"] == "confirm_issue":
        if text:
            t = text.strip().lower()
            if t in ["yes","y","correct","right","yep","yeah"]:
                s["issue_type"] = s.get("ai_guess") or s.get("issue_type") or "unknown"
            elif t in ["no","n","incorrect","wrong"]:
                s["step"] = "ask_issue_type"
                return JSONResponse({"reply": "Thanks. " + issue_prompt(), "done": False})
            else:
                if not s.get("issue_type") or s["issue_type"] == "unknown":
                    s["step"] = "ask_issue_type"
                    return JSONResponse({"reply": issue_prompt(), "done": False})
        s["step"] = "ask_details"

    # Ask for type explicitly
    if s["step"] == "ask_issue_type":
        if not text: return JSONResponse({"reply": issue_prompt(), "done": False})
        s["issue_type"] = normalize_issue(text)
        if s["issue_type"] == "unknown":
            return JSONResponse({"reply": "Sorry, I still didn’t catch that.\n" + issue_prompt(), "done": False})
        s["step"] = "ask_details"

    # Start
    if s["step"] in ["start", None]:
        if text:
            s["step"] = "ask_details"
        else:
            return JSONResponse({"reply": "Attach a photo if you can. Otherwise, tell me the issue and the road + milepost/intersection.", "done": False})

    # Ask details / Save when complete
    if s["step"] in ["ask_details","ask_location","ask_severity"]:
        missing = next_missing_fields(s)
        if not missing:
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
            mp_txt = f"MP {round(s['milepost'],2)}" if s.get('milepost') is not None else "(MP unknown)"
            route_txt = s.get('route') or '(unknown road)'
            return JSONResponse({
                "reply": ("Logged ✅\n"
                          f"Issue: {s['issue_type'] or 'unknown'}\n"
                          f"Route: {route_txt} {mp_txt}\n"
                          f"Severity: {s.get('severity') or '(unspecified)'}\n"
                          f"Lane blocked: {'yes' if s.get('lane_blocked') else 'no' if s.get('lane_blocked') == 0 else '(unspecified)'}"),
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
        return JSONResponse({"reply":"Got it. Could you tell me " + "; and ".join(asks) + "?", "done": False})

    if s["step"] == "done":
        return JSONResponse({"reply":"Report already completed for this session. Reset to start a new one.", "done": True})

    return JSONResponse({"reply":"I didn’t catch that. Try attaching a photo or answering the last question.", "done": False})

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