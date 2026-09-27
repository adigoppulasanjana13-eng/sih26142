# 🛰️ GeoSpecSR — Satellite Image Super-Resolution & Geospatial Analytics

**GeoSpecSR** is a deep learning framework and web platform that enhances low-resolution Sentinel-2 satellite imagery (10m) into high-precision, sub-meter output imagery (<2.5m). It combines a custom residual channel-attention neural network with real-time spatial uncertainty estimation, blur degradation analysis, interactive 3D terrain relief, and multi-spectral GIS overlays — served through a FastAPI backend and a React/Leaflet frontend.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Manual Setup](#manual-setup)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)

---

## Features

### Super-Resolution Engine
- **`GeoSpecSRNet`** — a custom PyTorch model combining Residual Channel Attention (`ColorSpectralAttention`) and Geometry-aware Spatial Attention (`GeometrySpatialAttention`).
- **Sobel edge guidance** for sharper restoration of fine structures — building rooftops, narrow roads, and field boundaries.
- **Automatic hardware acceleration** — detects and uses CUDA (NVIDIA GPUs), MPS (Apple Silicon), or falls back to CPU.

### Geospatial Mapping
- Interactive **Leaflet** map with Esri World Boundaries & Transportation overlays (cities, roads, districts).
- Switchable map modes: Satellite + Labels, Street Map, and Pure Satellite.
- Geocoding and reverse geocoding via the OpenStreetMap Nominatim API.

### Multi-Spectral & Uncertainty Analytics
- **NDVI (Agricultural Index)** — false-color NIR vegetation/crop canopy mapping.
- **Spatial uncertainty maps** — quantifies model-inferred detail versus directly observed data.
- **Blur degradation matrix** — interactive comparison grid across σ = 1–5 degradation levels.
- **3D terrain relief mesh** — interactive WebGL mesh with adjustable sun/lighting angle.
- **One-click JSON export** — full scientific metric validation reports.

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | FastAPI, PyTorch, torchvision, OpenCV, scikit-image, SciPy, NumPy, Pillow, Matplotlib |
| **Frontend** | React 19, Vite, Leaflet / React-Leaflet, Three.js, Axios |

---

## Quick Start

Run the full stack (backend + frontend) with one command:

**macOS / Linux**
```bash
./start.sh
```

**Windows**
```cmd
start.bat
```

Then open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## Manual Setup

### 1. Backend (FastAPI + PyTorch)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- Interactive API docs (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 2. Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

- Web app: [http://localhost:5173](http://localhost:5173)

---

## Project Structure
