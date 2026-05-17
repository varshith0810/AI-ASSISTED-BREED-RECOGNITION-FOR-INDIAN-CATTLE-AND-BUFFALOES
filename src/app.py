import json
import os
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from PIL import Image
from torchvision import models, transforms

app = FastAPI(title="Cattle Breed Recognition Frontend + API")

MODEL = None
CLASSES = None
MODEL_META = None
TFMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

BASE_STYLE = """
<style>
:root {
  --bg: #0b1020;
  --card: rgba(255,255,255,0.12);
  --card-solid: #ffffff;
  --text: #e5e7eb;
  --muted: #94a3b8;
  --primary: #7c3aed;
  --secondary: #06b6d4;
  --success: #22c55e;
}
*{box-sizing:border-box}
body{
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
  margin:0;
  color:var(--text);
  background: radial-gradient(1200px 700px at 10% -5%, #3b82f6 0%, transparent 60%),
              radial-gradient(900px 650px at 95% 8%, #9333ea 0%, transparent 55%),
              linear-gradient(140deg, #020617 0%, #0b1020 35%, #111827 100%);
  min-height:100vh;
}
.shell{max-width:1120px;margin:0 auto;padding:34px 20px 28px}
.topbar{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:26px}
.logo{display:flex;align-items:center;gap:10px;font-weight:700;letter-spacing:.2px}
.logo-dot{width:12px;height:12px;border-radius:50%;background:linear-gradient(120deg,var(--secondary),var(--primary));box-shadow:0 0 22px #22d3ee}
.nav{display:flex;gap:10px;flex-wrap:wrap}
.nav a,.ghost-link{text-decoration:none;padding:10px 14px;border-radius:12px;color:#dbeafe;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.18);font-size:14px}
.nav a:hover,.ghost-link:hover{background:rgba(255,255,255,.16)}
.hero{display:grid;grid-template-columns:1.2fr .8fr;gap:22px;align-items:stretch}
.card{background:var(--card);backdrop-filter: blur(10px);border-radius:20px;border:1px solid rgba(255,255,255,.18);box-shadow:0 18px 35px rgba(2,6,23,.45);padding:24px}
h1,h2,h3,p{margin:0}
.headline{font-size:34px;font-weight:800;line-height:1.2}
.subtitle{color:#bfdbfe;margin-top:10px;max-width:760px}
.badge{display:inline-flex;align-items:center;gap:8px;margin:4px 0 14px;background:rgba(34,211,238,.12);border:1px solid rgba(34,211,238,.35);color:#67e8f9;padding:7px 12px;border-radius:999px;font-size:12px;font-weight:600;letter-spacing:.2px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
label{display:block;font-size:13px;color:#dbeafe;margin-bottom:6px}
input,button{width:100%;padding:12px 14px;border-radius:12px;border:1px solid rgba(255,255,255,.22);outline:none;font-size:14px}
input{background:rgba(255,255,255,.1);color:#f8fafc}
input::placeholder{color:#94a3b8}
input:focus{border-color:#60a5fa;box-shadow:0 0 0 3px rgba(96,165,250,.2)}
button{background:linear-gradient(120deg,var(--secondary),var(--primary));color:#fff;font-weight:700;cursor:pointer;border:none;letter-spacing:.2px}
button:hover{filter:brightness(1.08)}
.btn-secondary{background:rgba(255,255,255,.11);border:1px solid rgba(255,255,255,.2)}
.side-stats{display:grid;grid-template-columns:1fr;gap:12px}
.stat{padding:14px;border-radius:14px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.14)}
.stat b{display:block;font-size:22px;margin-bottom:4px;color:#fff}
.result{margin-top:16px;padding:14px;border-radius:14px;background:#f8fafc;color:#0f172a;border:1px solid #e2e8f0}
.result p{margin:6px 0}
ul{margin:8px 0 0 18px;padding:0}
.flash{padding:10px 12px;border-radius:12px;margin-bottom:12px;font-size:14px}
.flash.success{background:rgba(34,197,94,.18);border:1px solid rgba(34,197,94,.45);color:#bbf7d0}
.auth-wrap{max-width:580px;margin:24px auto 0}
.footer{margin-top:20px;color:var(--muted);font-size:13px}
@media(max-width:900px){.hero{grid-template-columns:1fr}.grid{grid-template-columns:1fr}}
</style>
"""


def page_template(content: str, active: str = "home") -> str:
    def current(path: str) -> str:
        return "style='outline:2px solid rgba(125,211,252,.45)'" if active == path else ""

    return f"""<html><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1'>{BASE_STYLE}</head>
    <body><div class='shell'>
      <div class='topbar'>
        <div class='logo'><span class='logo-dot'></span> CattleVision AI</div>
        <div class='nav'>
          <a href='/' {current('home')}>Home</a>
          <a href='/signin' {current('signin')}>Sign In</a>
          <a href='/create-account' {current('create')}>Create Account</a>
        </div>
      </div>
      {content}
      <div class='footer'>Designed for farmers, vets, and breeding centers • Modern AI-assisted workflow.</div>
    </div></body></html>"""


def render_home():
    content = """
    <div class='hero'>
      <div class='card'>
        <div class='badge'>AI-Assisted Breed Recognition</div>
        <h1 class='headline'>Indian Cattle & Buffalo Breed Classifier</h1>
        <p class='subtitle'>Upload an animal image to predict breed with confidence scores. Best results come from clear side or front profile photos.</p>
        <form action='/predict' method='post' enctype='multipart/form-data' style='margin-top:16px'>
          <div class='grid'>
            <div><label>Upload Animal Image</label><input type='file' name='file' required></div>
            <div><label>Animal ID (optional)</label><input type='text' name='animal_id' placeholder='COW-2024-0042'></div>
          </div>
          <div style='margin-top:12px'><label>GPS Coordinates (optional)</label><input type='text' name='gps_coordinates' placeholder='30.8717N, 75.8520E'></div>
          <div class='grid' style='margin-top:16px'>
            <button type='submit'>Predict Breed</button>
            <a href='/create-account' class='ghost-link' style='display:flex;align-items:center;justify-content:center'>New user? Create account</a>
          </div>
        </form>
      </div>
      <div class='card side-stats'>
        <div class='stat'><b>Top-5</b><span>See ranked breed probabilities.</span></div>
        <div class='stat'><b>Fast Inference</b><span>Optimized for low hardware model serving.</span></div>
        <div class='stat'><b>Field Ready</b><span>Optional Animal ID and GPS metadata capture.</span></div>
      </div>
    </div>
    """
    return page_template(content, active="home")


def render_signin(message: str = ""):
    flash = f"<div class='flash success'>{message}</div>" if message else ""
    content = f"""
    <div class='auth-wrap'>
      <div class='card'>
        <div class='badge'>Welcome Back</div>
        <h2 class='headline' style='font-size:30px'>Sign In</h2>
        <p class='subtitle'>Access your classifier workspace and continue where you left off.</p>
        {flash}
        <form action='/signin' method='post' style='margin-top:16px'>
          <label>Email or Username</label>
          <input type='text' name='identity' placeholder='you@example.com or username' required>
          <div style='height:10px'></div>
          <label>Password</label>
          <input type='password' name='password' placeholder='••••••••' required>
          <div class='grid' style='margin-top:16px'>
            <button type='submit'>Sign In</button>
            <a href='/create-account' class='ghost-link' style='display:flex;align-items:center;justify-content:center'>Create new account</a>
          </div>
        </form>
      </div>
    </div>
    """
    return page_template(content, active="signin")


def render_create_account(message: str = ""):
    flash = f"<div class='flash success'>{message}</div>" if message else ""
    content = f"""
    <div class='auth-wrap'>
      <div class='card'>
        <div class='badge'>New User Registration</div>
        <h2 class='headline' style='font-size:30px'>Create Account</h2>
        <p class='subtitle'>Register with your email, username, and password to start using the platform.</p>
        {flash}
        <form action='/create-account' method='post' style='margin-top:16px'>
          <label>Email</label>
          <input type='email' name='email' placeholder='you@example.com' required>
          <div style='height:10px'></div>
          <label>Username</label>
          <input type='text' name='username' placeholder='farmer_raj' required>
          <div style='height:10px'></div>
          <label>Password</label>
          <input type='password' name='password' placeholder='Create a strong password' required>
          <div class='grid' style='margin-top:16px'>
            <button type='submit'>Create Account</button>
            <a href='/signin' class='ghost-link' style='display:flex;align-items:center;justify-content:center'>Back to sign in</a>
          </div>
        </form>
      </div>
    </div>
    """
    return page_template(content, active="create")

def render_result(top, conf, animal_id, gps_coordinates, rows):
    content = f"""
    <div class='auth-wrap' style='max-width:860px'>
      <div class='card'>
        <div class='badge'>Prediction Complete</div>
        <h2 class='headline' style='font-size:30px'>Breed Recognition Result</h2>
        <div class='result'>
          <p><b>Predicted Breed:</b> {top}</p>
          <p><b>Confidence:</b> {conf:.2f}%</p>
          <p><b>Animal ID:</b> {animal_id or 'N/A'}</p>
          <p><b>GPS Coordinates:</b> {gps_coordinates or 'N/A'}</p>
          <h4>Top-5 Scores</h4><ul>{rows}</ul>
        </div>
        <div class='grid' style='margin-top:14px'>
          <a href='/' class='ghost-link' style='display:flex;align-items:center;justify-content:center'>Try Another Image</a>
          <a href='/signin' class='ghost-link' style='display:flex;align-items:center;justify-content:center'>Go to Sign In</a>
        </div>
      </div>
    </div>
    """
    return page_template(content, active="home")

def render_result(top, conf, animal_id, gps_coordinates, rows):
    content = f"""
    <div class='auth-wrap' style='max-width:860px'>
      <div class='card'>
        <div class='badge'>Prediction Complete</div>
        <h2 class='headline' style='font-size:30px'>Breed Recognition Result</h2>
        <div class='result'>
          <p><b>Predicted Breed:</b> {top}</p>
          <p><b>Confidence:</b> {conf:.2f}%</p>
          <p><b>Animal ID:</b> {animal_id or 'N/A'}</p>
          <p><b>GPS Coordinates:</b> {gps_coordinates or 'N/A'}</p>
          <h4>Top-5 Scores</h4><ul>{rows}</ul>
        </div>
        <div class='grid' style='margin-top:14px'>
          <a href='/' class='ghost-link' style='display:flex;align-items:center;justify-content:center'>Try Another Image</a>
          <a href='/signin' class='ghost-link' style='display:flex;align-items:center;justify-content:center'>Go to Sign In</a>
        </div>
      </div>
    </div>
    """
    return page_template(content, active="home")

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
        extracted = [str(p.relative_to(tmp_dir)) for p in all_files]
        raise FileNotFoundError(
            "Bundle must contain class_names.json and at least one of "
            "breed_classifier_int8.pt or breed_classifier_ts.pt. "
            f"Extracted files: {extracted}"
        )

    classes_path = classes_candidates[0]
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    if int8_candidates:
        model_path = int8_candidates[0]
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, len(classes))
        base.eval()
        qmodel = torch.quantization.quantize_dynamic(base, {nn.Linear}, dtype=torch.qint8)
        state = torch.load(model_path, map_location="cpu")
        qmodel.load_state_dict(state)
        qmodel.eval()
        return qmodel, classes, {"type": "int8", "model_path": str(model_path), "classes_path": str(classes_path)}

    ts_path = ts_candidates[0]
    ts_model = torch.jit.load(str(ts_path), map_location="cpu")
    ts_model.eval()
    return ts_model, classes, {"type": "torchscript", "model_path": str(ts_path), "classes_path": str(classes_path)}


def _normalize_loaded(loaded):
    if isinstance(loaded, tuple):
        if len(loaded) == 3:
            return loaded[0], loaded[1], loaded[2]
        if len(loaded) == 2:
            return loaded[0], loaded[1], {"type": "unknown"}
    if isinstance(loaded, dict):
        return loaded.get("model"), loaded.get("classes"), loaded.get("meta", {"type": "unknown"})
    raise RuntimeError(f"Unexpected loader output type={type(loaded)} value={loaded}")


def get_model():
    global MODEL, CLASSES, MODEL_META
    if MODEL is None:
        bundle = Path(os.getenv("MODEL_BUNDLE", "cattle_model_low_hw.tar.gz"))
        loaded = _load_from_bundle(bundle)
        model, classes, meta = _normalize_loaded(loaded)
        MODEL, CLASSES, MODEL_META = model, classes, meta
    return MODEL, CLASSES


def inspect_bundle_files():
    bundle = Path(os.getenv("MODEL_BUNDLE", "cattle_model_low_hw.tar.gz"))
    info = {"bundle_path": str(bundle), "exists": bundle.exists(), "files": []}
    if not bundle.exists():
        return info

    tmp_dir = Path(tempfile.gettempdir()) / "cattle_model_bundle_debug"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tar:
        tar.extractall(tmp_dir)

    info["files"] = sorted(str(p.relative_to(tmp_dir)) for p in tmp_dir.rglob("*") if p.is_file())
    return info


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "model_type": (MODEL_META or {}).get("type")}


@app.get("/debug/bundle")
def debug_bundle():
    if os.getenv("DEBUG_BUNDLE", "false").lower() != "true":
        raise HTTPException(status_code=403, detail="Enable DEBUG_BUNDLE=true to use this endpoint")
    return inspect_bundle_files()


@app.get("/", response_class=HTMLResponse)
def home():
    return render_home()

@app.get("/signin", response_class=HTMLResponse)
def signin_page():
    return render_signin()

@app.get("/signin", response_class=HTMLResponse)
def signin_page():
    return render_signin()

@app.post("/signin")
async def signin(identity: str = Form(...), password: str = Form(...)):
    if not identity.strip() or not password.strip():
        raise HTTPException(status_code=400, detail="Identity and password are required.")
    return RedirectResponse(url="/", status_code=303)


@app.get("/create-account", response_class=HTMLResponse)
def create_account_page():
    return render_create_account()

@app.post("/create-account", response_class=HTMLResponse)
async def create_account(email: str = Form(...), username: str = Form(...), password: str = Form(...)):
    if not email.strip() or not username.strip() or not password.strip():
        raise HTTPException(status_code=400, detail="Email, username, and password are required.")
    return render_signin(message=f"Account created for {username}. Please sign in.")

@app.post("/create-account", response_class=HTMLResponse)
async def create_account(email: str = Form(...), username: str = Form(...), password: str = Form(...)):
    if not email.strip() or not username.strip() or not password.strip():
        raise HTTPException(status_code=400, detail="Email, username, and password are required.")
    return render_signin(message=f"Account created for {username}. Please sign in.")
@app.post("/predict", response_class=HTMLResponse)
async def predict_page(
    request: Request,
    file: UploadFile = File(...),
    animal_id: str = Form(default=""),
    gps_coordinates: str = Form(default=""),
):
    try:
        content = await file.read()
        image = Image.open(BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        model, classes = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}. Ensure latest deployment is active.")

    x = TFMS(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        vals, idxs = torch.topk(probs, 5)

    top = classes[idxs[0].item()]
    conf = vals[0].item() * 100
    rows = "".join([f"<li>{classes[i]}: {v*100:.2f}%</li>" for v, i in zip(vals.tolist(), idxs.tolist())])
    return render_result(top, conf, animal_id, gps_coordinates, rows)
