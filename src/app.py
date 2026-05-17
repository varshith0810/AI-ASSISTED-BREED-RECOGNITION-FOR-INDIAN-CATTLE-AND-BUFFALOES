import base64
import json
import os
import tarfile
import tempfile
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from PIL import Image
from starlette.middleware.sessions import SessionMiddleware
from torchvision import models, transforms

app = FastAPI(title="Cattle Breed Recognition Frontend + API")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "change-me"))

MODEL = None
CLASSES = None
MODEL_META = None
USERS = {}
TFMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

BASE_STYLE = """<style>body{font-family:Inter,Arial,sans-serif;background:#0b1020;color:#e5e7eb;margin:0}.shell{max-width:1100px;margin:0 auto;padding:24px}.card{background:#121a31;border:1px solid #263252;border-radius:16px;padding:20px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.nav a{color:#c7d2fe;text-decoration:none;margin-right:10px}.result{background:#f8fafc;color:#0f172a;border-radius:12px;padding:12px}.img-preview{max-width:360px;width:100%;border-radius:12px;border:1px solid #cbd5e1}input,button{width:100%;padding:10px;border-radius:10px;border:1px solid #334155}button{background:#6366f1;color:#fff;border:none}.muted{color:#94a3b8}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>"""


def _current_user(request: Request):
    return request.session.get("user")


def page_template(content: str, request: Request) -> str:
    user = _current_user(request)
    auth_links = "<a href='/logout'>Logout</a>" if user else "<a href='/signin'>Sign In</a> <a href='/create-account'>Create Account</a>"
    return f"""<html><head><meta name='viewport' content='width=device-width, initial-scale=1'>{BASE_STYLE}</head><body><div class='shell'>
    <div class='nav'><a href='/'>Home</a>{auth_links}</div>
    {content}</div></body></html>"""


def render_home(request: Request):
    user = _current_user(request)
    if not user:
        return RedirectResponse(url="/signin", status_code=303)
    content = """
    <div class='card'><h2>Indian Cattle & Buffalo Breed Classifier</h2>
    <p class='muted'>Prediction is available only after sign in.</p>
    <form action='/predict' method='post' enctype='multipart/form-data'>
      <div class='grid'>
        <div><label>Upload Animal Image</label><input type='file' name='file' required></div>
        <div><label>Animal ID (optional)</label><input type='text' name='animal_id'></div>
      </div>
      <div style='margin-top:10px'><label>GPS Coordinates (lat,long)</label><input type='text' name='gps_coordinates' placeholder='30.8717,75.8520'></div>
      <div style='margin-top:12px'><button type='submit'>Predict Breed</button></div>
    </form></div>"""
    return page_template(content, request)


def render_signin(request: Request, message: str = ""):
    flash = f"<p>{message}</p>" if message else ""
    content = f"""<div class='card'><h2>Sign In</h2>{flash}
    <form action='/signin' method='post'>
    <label>Username</label><input name='identity' required>
    <label style='margin-top:8px;display:block'>Password</label><input type='password' name='password' required>
    <div style='margin-top:12px'><button type='submit'>Sign In</button></div>
    </form><p>New user? <a href='/create-account'>Create account</a></p></div>"""
    return page_template(content, request)


def render_create_account(request: Request, message: str = ""):
    flash = f"<p>{message}</p>" if message else ""
    content = f"""<div class='card'><h2>Create Account</h2>{flash}
    <form action='/create-account' method='post'>
    <label>Email</label><input type='email' name='email' required>
    <label style='margin-top:8px;display:block'>Username</label><input name='username' required>
    <label style='margin-top:8px;display:block'>Password</label><input type='password' name='password' required>
    <div style='margin-top:12px'><button type='submit'>Create Account</button></div>
    </form></div>"""
    return page_template(content, request)


def render_result(request: Request, top, conf, animal_id, location_label, rows, image_b64):
    content = f"""<div class='card'><h2>Prediction Result</h2>
    <div class='result'>
      <p><b>Predicted Breed:</b> {top}</p>
      <p><b>Confidence:</b> {conf:.2f}%</p>
      <p><b>Animal ID:</b> {animal_id or 'N/A'}</p>
      <p><b>Detected Location:</b> {location_label}</p>
      <h4>Top-5 Scores</h4><ul>{rows}</ul>
    </div>
    <div style='margin-top:12px'>
      <h4>Uploaded Image</h4>
      <img class='img-preview' src='data:image/jpeg;base64,{image_b64}' alt='Uploaded animal'>
    </div>
    </div>"""
    return page_template(content, request)


def resolve_location_label(gps_coordinates: str) -> str:
    gps = (gps_coordinates or "").strip()
    if not gps:
        return "N/A"
    try:
        lat_str, lon_str = [x.strip() for x in gps.split(",", 1)]
        lat, lon = float(lat_str), float(lon_str)
    except Exception:
        return f"Invalid GPS format: {gps}. Use 'lat,long'."

    try:
        url = "https://nominatim.openstreetmap.org/reverse?" + urllib.parse.urlencode({
            "lat": lat,
            "lon": lon,
            "format": "jsonv2",
            "zoom": 14,
            "addressdetails": 1,
        })
        req = urllib.request.Request(url, headers={"User-Agent": "cattle-breed-app/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        addr = data.get("address", {})
        village_or_town = addr.get("village") or addr.get("town") or addr.get("city") or addr.get("hamlet")
        state = addr.get("state") or addr.get("county") or ""
        country = addr.get("country") or ""
        if village_or_town:
            parts = [village_or_town, state, country]
            return ", ".join([p for p in parts if p])
        return data.get("display_name", gps)
    except Exception:
        return f"{gps} (location lookup unavailable)"


def _load_from_bundle(bundle_path: Path):
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    tmp_dir = Path(tempfile.gettempdir()) / "cattle_model_bundle"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "r:gz") as tar:
        tar.extractall(tmp_dir)
    all_files = [p for p in tmp_dir.rglob("*") if p.is_file()]
    int8_candidates = [p for p in all_files if p.name == "breed_classifier_int8.pt"]
    ts_candidates = [p for p in all_files if p.name == "breed_classifier_ts.pt"]
    classes_candidates = [p for p in all_files if p.name == "class_names.json"]
    if not classes_candidates or (not int8_candidates and not ts_candidates):
        raise FileNotFoundError("Model bundle missing required files")
    with open(classes_candidates[0], "r", encoding="utf-8") as f:
        classes = json.load(f)
    if int8_candidates:
        model_path = int8_candidates[0]
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, len(classes))
        qmodel = torch.quantization.quantize_dynamic(base.eval(), {nn.Linear}, dtype=torch.qint8)
        state = torch.load(model_path, map_location="cpu")
        qmodel.load_state_dict(state)
        return qmodel.eval(), classes, {"type": "int8"}
    ts_model = torch.jit.load(str(ts_candidates[0]), map_location="cpu").eval()
    return ts_model, classes, {"type": "torchscript"}


def _normalize_loaded(loaded):
    if isinstance(loaded, tuple):
        if len(loaded) == 3:
            return loaded[0], loaded[1], loaded[2]
        if len(loaded) == 2:
            return loaded[0], loaded[1], {"type": "unknown"}
    if isinstance(loaded, dict):
        return loaded.get("model"), loaded.get("classes"), loaded.get("meta", {"type": "unknown"})
    raise RuntimeError(f"Unexpected loader output type={type(loaded)}")


def get_model():
    global MODEL, CLASSES, MODEL_META
    if MODEL is None:
        bundle = Path(os.getenv("MODEL_BUNDLE", "cattle_model_low_hw.tar.gz"))
        MODEL, CLASSES, MODEL_META = _normalize_loaded(_load_from_bundle(bundle))
    return MODEL, CLASSES


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "model_type": (MODEL_META or {}).get("type")}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render_home(request)


@app.get("/signin", response_class=HTMLResponse)
def signin_page(request: Request):
    return render_signin(request)


@app.post("/signin")
async def signin(request: Request, identity: str = Form(...), password: str = Form(...)):
    stored = USERS.get(identity.strip().lower())
    if not stored or stored["password"] != password:
        return HTMLResponse(render_signin(request, "Invalid credentials"), status_code=401)
    request.session["user"] = stored["username"]
    return RedirectResponse(url="/", status_code=303)


@app.get("/create-account", response_class=HTMLResponse)
def create_account_page(request: Request):
    return render_create_account(request)


@app.post("/create-account", response_class=HTMLResponse)
async def create_account(request: Request, email: str = Form(...), username: str = Form(...), password: str = Form(...)):
    key = username.strip().lower()
    if key in USERS:
        return HTMLResponse(render_create_account(request, "Username already exists"), status_code=400)
    USERS[key] = {"email": email.strip(), "username": username.strip(), "password": password}
    return HTMLResponse(render_signin(request, f"Account created for {username}. Please sign in."))


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/signin", status_code=303)


@app.post("/predict", response_class=HTMLResponse)
async def predict_page(
    request: Request,
    file: UploadFile = File(...),
    animal_id: str = Form(default=""),
    gps_coordinates: str = Form(default=""),
):
    if not _current_user(request):
        return RedirectResponse(url="/signin", status_code=303)

    try:
        content = await file.read()
        image = Image.open(BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        model, classes = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    x = TFMS(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        vals, idxs = torch.topk(probs, 5)

    top = classes[idxs[0].item()]
    conf = vals[0].item() * 100
    rows = "".join([f"<li>{classes[i]}: {v*100:.2f}%</li>" for v, i in zip(vals.tolist(), idxs.tolist())])
    location_label = resolve_location_label(gps_coordinates)
    image_b64 = base64.b64encode(content).decode("utf-8")
    return render_result(request, top, conf, animal_id, location_label, rows, image_b64)
