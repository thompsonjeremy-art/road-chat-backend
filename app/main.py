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
            "issue_type": None,
            "ai_guess": None,
            "severity": None,
            "lane_blocked": None, # 1/0
            "lat": None, "lon": None,
            "photo_path": None,
            "meta": {},
            "step": "start",
            "_last_req_id": None,
            "_last_reply_json": None,
        }
    return SESSIONS[session_id]

# ---------- Helpers ----------
def send_json(s: dict, reply: str, done: bool = False) -> JSONResponse:
    payload = {"reply": reply, "done": done}
    s["_last_reply_json"] = payload
    return JSONResponse(payload)

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
    # minor-ish
    if any(w in t for w in ["minor","small","little","tiny","hairline","shallow","not bad","no big deal","light"]):
        return "minor"
    # severe-ish
    if any(w in t for w in ["severe","major","big","huge","massive","deep","dangerous","bad","really bad","critical"]):
        return "severe"
    # moderate-ish
    if any(w in t for w in ["moderate","medium","in between","so-so","ok-ish"]):
        return "moderate"
    # numeric hint (1–10)
    m = re.search(r"\b([1-9]|10)\/?10\b", t)
    if m:
        n = int(m.group(1))
        if n <= 3: return "minor"
        if n >= 8: return "severe"
        return "moderate"
    return None

def parse_lane_blocked(text: str) -> Optional[int]:
    t = (text or "").lower().strip()
    if "lane" in t or "lanes" in t or "traffic" in t:
        if any(w in t for w in ["not blocked","unblocked","open","flowing","moving","clear","no issues"]): return 0
        if any(w in t for w in ["lane blocked","lanes blocked","blocked","closed","shut","impassable","stopped"]): return 1
    if t in ["no","nope","nah","negative","n"]: return 0
    if t in ["yes","yep","yeah","affirmative","y"]: return 1
    return None

ROAD_WORDS  = r"(?:highway|hwy|road|rd|street|st|avenue|ave|drive|dr|blvd|boulevard|lane|ln|way|pkwy|parkway|court|ct|place|pl|trail|trl|circle|cir)"
ROUTE_TOKEN = r"[A-Za-z0-9 .'\-]+?"

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

# ---------- Vision extractor (type + severity + blocked) ----------
def vision_extract_all(photo_bytes: bytes) -> dict:
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
    for k in ["type_conf","severity_conf","blocked_conf"]:
        out[k] = max(0.0, min(1.0, out[k]))
    if DEBUG: log.info(f"[VISION all] {out}")
    return out

# ---------- Natural-sounding reply writer ----------
def natural_reply(state: dict, *, proposals: dict | None = None, asks: list[str] | None = None, context_note: str = "") -> str:
    try:
        asks = asks or []
        p = proposals or {}
        payload = {
            "known": {
                "issue_type": state.get("issue_type"),
                "route": state.get("route"),
                "milepost": state.get("milepost"),
                "lat": state.get("lat"),
                "lon": state.get("lon"),
                "severity": state.get("severity"),
                "lane_blocked": state.get("lane_blocked"),
            },
            "proposals": p,
            "asks": asks[:2],
            "note": context_note or ""
        }
        sys = (
            "You are a concise, warm road-issue intake agent. "
            "Write ONE short message (1–2 sentences). "
            "Acknowledge what you understood, show any guessed values as soft suggestions "
            "and ask at most ONE clear follow-up. Avoid lists and multiple questions."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=90,
            messages=[{"role":"system","content":sys},{"role":"user","content":json.dumps(payload, ensure_ascii=False)}]
        )
        text = (resp.choices[0].message.content or "").strip()
        return text[:600] if text else "Got it. Could you help me with one more detail?"
    except Exception:
        ask = asks[0] if asks else "one more detail"
        return f"Thanks, I’ve logged what I can. Could you tell me {ask}?"

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
    return "the problem type (pothole, cracking, shoulder_drop, guardrail, sign, drainage, debris, snow_ice)"

# ---------- Endpoints ----------
@app.get("/health")
def health(): return {"status":"ok","debug":DEBUG}

@app.post("/vision_test")
async def vision_test(photo: UploadFile = File(...)):
    path, bytes_data = save_photo(photo)
    lat, lon = parse_exif_gps(bytes_data)
    vision = vision_extract_all(bytes_data)
    return {"vision": vision, "has_gps": bool(lat and lon), "saved_to": path}

@app.post("/chat")
async def chat(
    session_id: str = Form(...),
    text: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    req_id: Optional[str] = Form(None),   # for de-dupe
):
    s = ensure_session(session_id)

    # --- de-dupe identical retries from client ---
    if req_id:
        if s.get("_last_req_id") == req_id and s.get("_last_reply_json"):
            return JSONResponse(s["_last_reply_json"])
        s["_last_req_id"] = req_id

    # Device GPS (optional)
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

        # Obvious tree-across-road override
        notes = (vision.get("notes") or "").lower()
        if ("tree" in notes or "log" in notes) and ("across" in notes or "full width" in notes or "across road" in notes):
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

        loc_msg = "GPS found" if (s.get('lat') and s.get('lon')) else "no GPS from the photo"

        props = {}
        if vision["type"] != "unknown": props["issue_type"] = vision["type"]
        if vision["severity"] != "unknown": props["severity"] = vision["severity"]
        if vision["blocked"] != "unknown": props["lane_blocked"] = 1 if vision["blocked"] in ["yes","partial"] else 0

        if not s.get("issue_type"):
            s["step"] = "ask_issue_type"
            reply = natural_reply(s, proposals=props, asks=[issue_prompt()], context_note=f"Photo received; {loc_msg}.")
            return send_json(s, reply, False)

        s["step"] = "confirm_issue"
        follow = []
        if not s.get("severity"): follow.append("the severity (minor, moderate, or severe)")
        if s.get("lane_blocked") is None: follow.append("whether a lane is blocked (yes or no)")
        note = "Please confirm the issue type I suggested, or tell me the correct one. " + f"({loc_msg})"
        reply = natural_reply(s, proposals={"issue_type": s.get("ai_guess"), **props}, asks=follow[:1], context_note=note)
        return send_json(s, reply, False)

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
                reply = natural_reply(s, asks=[issue_prompt()], context_note="Thanks for the correction.")
                return send_json(s, reply, False)
            else:
                if not s.get("issue_type") or s["issue_type"] == "unknown":
                    s["step"] = "ask_issue_type"
                    reply = natural_reply(s, asks=[issue_prompt()], context_note="I didn’t quite catch the type.")
                    return send_json(s, reply, False)
        s["step"] = "ask_details"

    # Ask for type explicitly
    if s["step"] == "ask_issue_type":
        if not text:
            reply = natural_reply(s, asks=[issue_prompt()], context_note="")
            return send_json(s, reply, False)
        s["issue_type"] = normalize_issue(text)
        if s["issue_type"] == "unknown":
            reply = natural_reply(s, asks=[issue_prompt()], context_note="Sorry, I still didn’t catch that.")
            return send_json(s, reply, False)
        s["step"] = "ask_details"

    # Start
    if s["step"] in ["start", None]:
        if text:
            s["step"] = "ask_details"
        else:
            reply = natural_reply(s, asks=["a short description and the road + nearest milepost or intersection"], context_note="Ready when you are.")
            return send_json(s, reply, False)

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
            out = (
                f"Logged ✅ I’ve saved your report.\n"
                f"Issue: {s['issue_type'] or 'unknown'}\n"
                f"Route: {route_txt} {mp_txt}\n"
                f"Severity: {s.get('severity') or '(unspecified)'}\n"
                f"Lane blocked: {'yes' if s.get('lane_blocked') else 'no' if s.get('lane_blocked') == 0 else '(unspecified)'}"
            )
            return send_json(s, out, True)

        asks = []
        if "issue_type" in missing: asks.append(issue_prompt())
        if "route" in missing or "milepost" in missing: asks.append("the road name and the nearest milepost or intersection")
        if "severity" in missing: asks.append("the severity (minor, moderate, or severe)")
        if "lane_blocked" in missing: asks.append("whether a lane is blocked (yes or no)")
        reply = natural_reply(s, asks=[asks[0]], context_note="")
        return send_json(s, reply, False)

    if s["step"] == "done":
        return send_json(s, "Report already completed for this session. Send another photo or describe a new issue to start a fresh report.", True)

    return send_json(s, "I didn’t catch that. Try attaching a photo or answering the last question.", False)

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