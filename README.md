# 🛰️ GeoSpecSR — Satellite Image Super-Resolution & Analytics

GeoSpecSR is a deep learning framework and geospatial platform designed to enhance low-resolution **Sentinel-2 10m satellite imagery into high-precision <2.5m sub-meter output imagery**. It includes residual channel attention neural networks, real-time spatial uncertainty estimation, blur degradation differentiation ($\sigma=1\dots5$), interactive 3D height relief, and multi-spectral GIS overlays.

---

## 🚀 Quick Start for Mentors / Judges (1-Command Run)

### Linux / macOS
```bash
./start.sh
```

### Windows
```cmd
start.bat
```

> **Open Application**: After running the script, open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser!

---

## 🛠️ Manual Step-by-Step Setup

### 1. Backend Setup (FastAPI & PyTorch)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
- API Docs (Swagger): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 2. Frontend Setup (React & Vite)
```bash
cd frontend
npm install
npm run dev
```
- Web Application UI: [http://localhost:5173](http://localhost:5173)

---

## 🔑 Key Features & Technical Highlights

1. **PyTorch Super-Resolution Engine (`GeoSpecSRNet`)**:
   - Residual Channel & Geometry Spatial Attention (`ColorSpectralAttention` & `GeometrySpatialAttention`).
   - Sobel Edge Guidance for high-frequency restoration of small building roofs, narrow roads, and field borders.
   - Hardware-accelerated auto-detection (`CUDA` for Nvidia GPUs, `MPS` for Apple Silicon Macs, or `CPU`).

2. **Geospatial Map & Location Labels**:
   - Interactive Leaflet map with **Esri World Boundaries & Transportation overlays** (city names, road names, districts).
   - Dynamic map mode switcher (`🛰️ Satellite + Labels`, `🗺️ Street Map`, `🌐 Pure Satellite`).
   - Geocoding and reverse geocoding via OpenStreetMap Nominatim API.

3. **Multi-Spectral & Uncertainty Analytics**:
   - **Agricultural NDVI Index**: False color NIR and crop canopy vegetation index map.
   - **Spatial Uncertainty**: Quantifies model-inferred fine details vs directly observed data.
   - **Blur Degradation Matrix**: Interactive $\sigma=1\dots5$ degradation comparison grid.
   - **3D Terrain Relief Mesh**: Interactive WebGL 3D mesh with sun lighting control.
   - **One-Click JSON Validation Report**: Export full scientific metric assessments.
