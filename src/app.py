import json
import os
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
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
body{font-family:Inter,Arial,sans-serif;background:linear-gradient(120deg,#f5f7ff,#eefaf6);margin:0;color:#1f2937}
.container{max-width:960px;margin:40px auto;padding:24px}
.card{background:white;border-radius:18px;box-shadow:0 10px 30px rgba(17,24,39,.08);padding:24px}
.title{font-size:28px;font-weight:700;margin-bottom:8px}
.subtitle{color:#6b7280;margin-bottom:20px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
input,button{width:100%;padding:12px;border-radius:10px;border:1px solid #d1d5db}
button{background:#111827;color:#fff;font-weight:600;cursor:pointer}
button:hover{background:#0b1220}
.badge{display:inline-block;background:#ecfeff;color:#155e75;border:1px solid #a5f3fc;padding:6px 10px;border-radius:999px;font-size:12px}
.result{margin-top:18px;padding:16px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:12px}
ul{margin-top:8px}
.footer{margin-top:16px;color:#6b7280;font-size:13px}
@media(max-width:768px){.grid{grid-template-columns:1fr}}
</style>
"""


def render_home():
    return f"""<html><head>{BASE_STYLE}</head><body><div class='container'><div class='card'>
    <div class='badge'>AI-Assisted Breed Recognition</div>
    <div class='title'>Indian Cattle & Buffalo Breed Classifier</div>
    <div class='subtitle'>Upload animal image to predict breed (software-only model).</div>
    <form action='/predict' method='post' enctype='multipart/form-data'>
      <div class='grid'>
        <div><label>Upload Animal Image</label><input type='file' name='file' required></div>
        <div><label>Animal ID (optional)</label><input type='text' name='animal_id' placeholder='COW-2024-0042'></div>
      </div>
      <div style='margin-top:12px'><label>GPS Coordinates (optional)</label><input type='text' name='gps_coordinates' placeholder='30.8717N, 75.8520E'></div>
      <div style='margin-top:16px'><button type='submit'>Predict Breed</button></div>
    </form>
    <div class='footer'>Tip: Use clear side/front profile image for better accuracy.</div>
    </div></div></body></html>"""


def render_result(top, conf, animal_id, gps_coordinates, rows):
    return f"""<html><head>{BASE_STYLE}</head><body><div class='container'><div class='card'>
    <div class='badge'>Prediction Complete</div>
    <div class='title'>Breed Recognition Result</div>
    <div class='result'>
      <p><b>Predicted Breed:</b> {top}</p>
      <p><b>Confidence:</b> {conf:.2f}%</p>
      <p><b>Animal ID:</b> {animal_id or 'N/A'}</p>
      <p><b>GPS Coordinates:</b> {gps_coordinates or 'N/A'}</p>
      <h4>Top-5 Scores</h4><ul>{rows}</ul>
    </div>
    <div style='margin-top:16px'><a href='/'><button>Try Another Image</button></a></div>
    </div></div></body></html>"""


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
