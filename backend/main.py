import io
import time
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image

from model import SRInferenceEngine
from dataset import SAMPLES_META, get_sample_data, generate_sample_images_if_missing

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "static_outputs"
DIST_DIR = BASE_DIR.parent / "frontend" / "dist"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

engine = SRInferenceEngine()
generate_sample_images_if_missing()

app = FastAPI(
    title="GeoSpecSR API — Deep Learning Satellite Super-Resolution",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Serve frontend build assets if available
if (DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

class ProcessRequest(BaseModel):
    sample_id: str = "hyderabad"
    target_resolution_m: float = 2.0
    scale_factor: int = 4
    mode_layer: str = "Satellite RGB"
    lat: float = None
    lng: float = None

@app.get("/")
def read_root():
    if (DIST_DIR / "index.html").exists():
        return FileResponse(DIST_DIR / "index.html")
    return {
        "status": "ok",
        "service": "GeoSpecSR API",
        "ui": "http://localhost:5173"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "GeoSpecSR API",
        "device": str(engine.device),
        "model": "GeoSpecSRNet Residual Channel Attention & Uncertainty Engine",
        "features": [
            "Sentinel-2 10m to <4m Super-Resolution",
            "RGB Color Channel Attention",
            "Sobel Edge Geometry Object Detection",
            "Spatial Uncertainty & Model Confidence Map"
        ],
        "supported_resolutions": ["1.0m", "2.0m", "2.5m", "4.0m"],
        "available_samples": list(SAMPLES_META.keys())
    }

@app.get("/samples")
def get_samples():
    samples_list = []
    for sid, meta in SAMPLES_META.items():
        samples_list.append({
            **meta,
            "lr_image_url": f"/outputs/{sid}_lr.png",
            "hr_reference_url": f"/outputs/{sid}_hr.png"
        })
    return {"samples": samples_list}

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    target_resolution_m: float = 2.0,
    blur_sigma: float = 2.0
):
    contents = await file.read()
    try:
        input_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    scale = 4 if target_resolution_m <= 2.5 else (2 if target_resolution_m >= 4.0 else 4)
    job_id = uuid4().hex[:12]
    start_time = time.time()
    
    results = engine.process_image(input_pil, scale_factor=scale)
    exec_time_sec = round(time.time() - start_time, 3)

    lr_fn = f"{job_id}_input_lr.png"
    sr_fn = f"{job_id}_output_sr.png"
    agri_fn = f"{job_id}_agriculture.png"
    edge_fn = f"{job_id}_edge.png"
    nir_fn = f"{job_id}_nir.png"
    bicubic_fn = f"{job_id}_bicubic.png"
    heatmap_fn = f"{job_id}_heatmap.png"
    uncert_fn = f"{job_id}_uncertainty.png"
    grid_fn = f"{job_id}_differentiation_grid.png"

    input_pil.save(OUTPUT_DIR / lr_fn)
    results["sr_image"].save(OUTPUT_DIR / sr_fn)
    results["agriculture_image"].save(OUTPUT_DIR / agri_fn)
    results["edge_image"].save(OUTPUT_DIR / edge_fn)
    results["nir_image"].save(OUTPUT_DIR / nir_fn)
    results["bicubic_image"].save(OUTPUT_DIR / bicubic_fn)
    results["heatmap_image"].save(OUTPUT_DIR / heatmap_fn)
    results["uncertainty_image"].save(OUTPUT_DIR / uncert_fn)
    results["differentiation_grid"].save(OUTPUT_DIR / grid_fn)

    sigma_urls = {}
    for s_name, s_pil in results.get("sigma_tiles", {}).items():
        s_fn = f"{job_id}_{s_name}.png"
        s_pil.save(OUTPUT_DIR / s_fn)
        sigma_urls[s_name] = f"/outputs/{s_fn}"

    return {
        "job_id": job_id,
        "status": "completed",
        "mode": "custom_user_upload",
        "filename": file.filename,
        "input_resolution_m": 10.0,
        "target_resolution_m": target_resolution_m,
        "blur_sigma": blur_sigma,
        "execution_time_seconds": exec_time_sec,
        "metrics": results["metrics"],
        "sigma_metrics": results.get("sigma_metrics", []),
        "uncertainty": results.get("uncertainty", {}),
        "object_breakdown": results.get("object_breakdown", {}),
        "urls": {
            "input_lr": f"/outputs/{lr_fn}",
            "output_sr": f"/outputs/{sr_fn}",
            "differentiation_grid": f"/outputs/{grid_fn}",
            "agriculture": f"/outputs/{agri_fn}",
            "edge": f"/outputs/{edge_fn}",
            "nir": f"/outputs/{nir_fn}",
            "bicubic": f"/outputs/{bicubic_fn}",
            "heatmap": f"/outputs/{heatmap_fn}",
            "uncertainty": f"/outputs/{uncert_fn}",
            **sigma_urls
        },
        "note": "PyTorch Super-Resolution & Uncertainty Evaluation successfully completed."
    }

@app.post("/process")
def process_study_area(req: ProcessRequest):
    meta, lr_img, hr_img = get_sample_data(req.sample_id, lat=req.lat, lng=req.lng)
    
    scale = 4 if req.target_resolution_m <= 2.5 else (2 if req.target_resolution_m >= 4.0 else 4)
    job_id = f"{req.sample_id}_{uuid4().hex[:8]}"
    start_time = time.time()
    
    results = engine.process_image(lr_img, reference_pil_img=hr_img, scale_factor=scale, mode_layer=req.mode_layer)
    exec_time_sec = round(time.time() - start_time, 3)

    lr_fn = f"{job_id}_input_lr.png"
    sr_fn = f"{job_id}_output_sr.png"
    agri_fn = f"{job_id}_agriculture.png"
    edge_fn = f"{job_id}_edge.png"
    nir_fn = f"{job_id}_nir.png"
    bicubic_fn = f"{job_id}_bicubic.png"
    heatmap_fn = f"{job_id}_heatmap.png"
    uncert_fn = f"{job_id}_uncertainty.png"
    hr_ref_fn = f"{job_id}_reference_hr.png"
    grid_fn = f"{job_id}_differentiation_grid.png"

    lr_img.save(OUTPUT_DIR / lr_fn)
    results["sr_image"].save(OUTPUT_DIR / sr_fn)
    results["agriculture_image"].save(OUTPUT_DIR / agri_fn)
    results["edge_image"].save(OUTPUT_DIR / edge_fn)
    results["nir_image"].save(OUTPUT_DIR / nir_fn)
    results["bicubic_image"].save(OUTPUT_DIR / bicubic_fn)
    results["heatmap_image"].save(OUTPUT_DIR / heatmap_fn)
    results["uncertainty_image"].save(OUTPUT_DIR / uncert_fn)
    results["differentiation_grid"].save(OUTPUT_DIR / grid_fn)
    hr_img.save(OUTPUT_DIR / hr_ref_fn)

    sigma_urls = {}
    for s_name, s_pil in results.get("sigma_tiles", {}).items():
        s_fn = f"{job_id}_{s_name}.png"
        s_pil.save(OUTPUT_DIR / s_fn)
        sigma_urls[s_name] = f"/outputs/{s_fn}"

    return {
        "job_id": job_id,
        "status": "completed",
        "mode": "marked_location_inference" if req.lat else "real_dataset_inference",
        "sample_info": meta,
        "input_resolution_m": 10.0,
        "target_resolution_m": req.target_resolution_m,
        "execution_time_seconds": exec_time_sec,
        "metrics": results["metrics"],
        "sigma_metrics": results.get("sigma_metrics", []),
        "uncertainty": results.get("uncertainty", {}),
        "object_breakdown": results.get("object_breakdown", {}),
        "urls": {
            "input_lr": f"/outputs/{lr_fn}",
            "output_sr": f"/outputs/{sr_fn}",
            "differentiation_grid": f"/outputs/{grid_fn}",
            "agriculture": f"/outputs/{agri_fn}",
            "edge": f"/outputs/{edge_fn}",
            "nir": f"/outputs/{nir_fn}",
            "bicubic": f"/outputs/{bicubic_fn}",
            "heatmap": f"/outputs/{heatmap_fn}",
            "uncertainty": f"/outputs/{uncert_fn}",
            "reference_hr": f"/outputs/{hr_ref_fn}",
            **sigma_urls
        },
        "validation": f"Scientific accuracy assessment & uncertainty map computed at {req.target_resolution_m}m target resolution."
    }
