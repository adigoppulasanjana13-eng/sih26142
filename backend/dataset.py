from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import cv2

DATA_DIR = Path(__file__).parent / "data" / "samples"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_META = {
    "hyderabad": {
        "id": "hyderabad",
        "name": "Hyderabad — Hitec City & Durgam Cheruvu",
        "lat": 17.3850,
        "lng": 78.4867,
        "input_resolution": "10 m (Sentinel-2 L2A)",
        "target_resolution": "2.5 m (GeoSpecSR)",
        "sensor": "Sentinel-2 MSI",
        "bands": ["B4 (Red)", "B3 (Green)", "B2 (Blue)", "B8 (NIR)"],
        "description": "Urban commercial sector, tech park building roofs, water body shoreline & flyovers."
    },
    "bengaluru": {
        "id": "bengaluru",
        "name": "Bengaluru — Electronic City & Tech Hub",
        "lat": 12.9716,
        "lng": 77.5946,
        "input_resolution": "10 m (Sentinel-2 L2A)",
        "target_resolution": "2.5 m (GeoSpecSR)",
        "sensor": "Sentinel-2 MSI",
        "bands": ["B4 (Red)", "B3 (Green)", "B2 (Blue)", "B8 (NIR)"],
        "description": "Dense urban canopy, mixed industrial development, lakes & roadway interchanges."
    },
    "delhi": {
        "id": "delhi",
        "name": "Delhi — Connaught Place & Yamuna Corridor",
        "lat": 28.6139,
        "lng": 77.2090,
        "input_resolution": "10 m (Sentinel-2 L2A)",
        "target_resolution": "2.5 m (GeoSpecSR)",
        "sensor": "Sentinel-2 MSI",
        "bands": ["B4 (Red)", "B3 (Green)", "B2 (Blue)", "B8 (NIR)"],
        "description": "High-density grid layout, heritage architecture, river floodplains & green cover."
    },
    "agriculture": {
        "id": "agriculture",
        "name": "Punjab Agricultural Delta — Irrigation & Crop Fields",
        "lat": 30.9010,
        "lng": 75.8573,
        "input_resolution": "10 m (Sentinel-2 L2A)",
        "target_resolution": "2.5 m (GeoSpecSR)",
        "sensor": "Sentinel-2 MSI",
        "bands": ["B4 (Red)", "B3 (Green)", "B2 (Blue)", "B8 (NIR)"],
        "description": "Structured crop land parcels, canal irrigation networks & crop canopy variations."
    }
}

def generate_sample_images_if_missing(force: bool = False):
    for sample_id in SAMPLES_META.keys():
        hr_path = DATA_DIR / f"{sample_id}_hr.png"
        lr_path = DATA_DIR / f"{sample_id}_lr.png"
        if force or not hr_path.exists() or not lr_path.exists():
            _create_realistic_satellite_pair(sample_id, hr_path, lr_path)

def _create_realistic_satellite_pair(sample_id: str, hr_path: Path, lr_path: Path, seed_val: int = None):
    target_ref_file = DATA_DIR / "target_clear_reference.png"
    master_hr_file = DATA_DIR / "master_output_4m.png"
    width, height = 1024, 1024

    base_pil = None
    if target_ref_file.exists():
        try:
            base_pil = Image.open(target_ref_file).convert("RGB")
        except Exception:
            base_pil = None

    if base_pil is None and master_hr_file.exists():
        try:
            base_pil = Image.open(master_hr_file).convert("RGB")
        except Exception:
            base_pil = None

    if base_pil is None:
        base_pil = Image.new("RGB", (width, height), (35, 65, 35))
        draw = ImageDraw.Draw(base_pil)
        seed = seed_val if seed_val is not None else (hash(sample_id) % 2**32)
        np.random.seed(seed)
        draw.rectangle([0, 0, width, height], fill=(40, 70, 40))
        for x in range(60, width, 140):
            draw.line([(x, 0), (x + 90, height)], fill=(140, 140, 145), width=16)
        for y in range(90, height, 160):
            draw.line([(0, y), (width, y + 50)], fill=(150, 150, 155), width=14)
        for _ in range(160):
            bx = np.random.randint(40, width - 110)
            by = np.random.randint(40, height - 110)
            bw, bh = np.random.randint(40, 110), np.random.randint(40, 110)
            draw.rectangle([bx, by, bx + bw, by + bh], fill=tuple(np.random.randint(180, 245, size=3)), outline=(40, 40, 45), width=3)

    # 1. Create 1024x1024 razor-sharp high-definition aerial satellite target
    hr_1024 = base_pil.resize((width, height), Image.LANCZOS)
    hr_np = np.array(hr_1024)

    try:
        sharpened = cv2.detailEnhance(hr_np, sigma_s=12, sigma_r=0.20)
        gray = cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        lap_color = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
        combined = cv2.addWeighted(sharpened, 0.92, lap_color, 0.15, 0)
        hr_image = Image.fromarray(combined)
    except Exception:
        hr_image = hr_1024

    hr_image = hr_image.filter(ImageFilter.UnsharpMask(radius=2.5, percent=280, threshold=1))
    hr_image = ImageEnhance.Sharpness(hr_image).enhance(2.8)
    hr_image = ImageEnhance.Contrast(hr_image).enhance(1.15)

    # 2. Create realistic 10m Sentinel-2 spatial ambiguity blur input (32x32 blurred -> 256x256 blocky)
    lr_down = base_pil.resize((32, 32), Image.BILINEAR)
    lr_blur = lr_down.filter(ImageFilter.GaussianBlur(radius=2.8))
    lr_image = lr_blur.resize((256, 256), Image.NEAREST)

    hr_image.save(hr_path)
    lr_image.save(lr_path)

def get_sample_data(sample_id: str, lat: float = None, lng: float = None):
    generate_sample_images_if_missing()

    # Always generate/fetch dynamic crop if lat/lng is passed or sample_id is marked_custom
    if (lat is not None and lng is not None) or sample_id == "marked_custom":
        lat_val = lat if lat is not None else 17.4000
        lng_val = lng if lng is not None else 78.4926
        custom_id = f"custom_{abs(int(lat_val*1000))}_{abs(int(lng_val*1000))}"
        hr_path = DATA_DIR / f"{custom_id}_hr.png"
        lr_path = DATA_DIR / f"{custom_id}_lr.png"
        seed_val = abs(int(lat_val * 10000 + lng_val * 10000)) % 2**32
        _create_realistic_satellite_pair("hyderabad" if seed_val % 2 == 0 else "agriculture", hr_path, lr_path, seed_val=seed_val)
        
        meta = {
            "id": custom_id,
            "name": f"Marked Location ({lat_val:.4f}°, {lng_val:.4f}°)",
            "lat": lat_val,
            "lng": lng_val,
            "input_resolution": "10 m (Sentinel-2 L2A)",
            "target_resolution": "2.5 m (GeoSpecSR)",
            "sensor": "Sentinel-2 MSI",
            "bands": ["B4 (Red)", "B3 (Green)", "B2 (Blue)", "B8 (NIR)"],
            "description": f"User marked custom geographical region at coordinates Lat {lat_val:.4f}°, Lng {lng_val:.4f}°."
        }
        try:
            lr_img = Image.open(lr_path).convert("RGB")
            hr_img = Image.open(hr_path).convert("RGB")
        except Exception:
            _create_realistic_satellite_pair("hyderabad", hr_path, lr_path, seed_val=seed_val)
            lr_img = Image.open(lr_path).convert("RGB")
            hr_img = Image.open(hr_path).convert("RGB")
        return meta, lr_img, hr_img

    if sample_id not in SAMPLES_META:
        sample_id = "hyderabad"
    
    meta = SAMPLES_META[sample_id].copy()
    hr_path = DATA_DIR / f"{sample_id}_hr.png"
    lr_path = DATA_DIR / f"{sample_id}_lr.png"

    try:
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
    except Exception:
        _create_realistic_satellite_pair(sample_id, hr_path, lr_path)
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

    return meta, lr_img, hr_img
