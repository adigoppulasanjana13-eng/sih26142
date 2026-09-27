import React, { useEffect, useState, useRef, Component } from "react";
import { createRoot } from "react-dom/client";
import axios from "axios";
import * as THREE from "three";
import { MapContainer, TileLayer, Circle, Marker, Popup, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./styles.css";

// Fix Leaflet marker icon asset paths
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
  iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png"
});

const API = "http://127.0.0.1:8000";

const markerIcon = L.divIcon({
  className: "custom-marker",
  html: '<div class="marker-pin"><div class="pin-inner">SR</div></div>',
  iconSize: [44, 44],
  iconAnchor: [22, 44]
});

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("GeoSpecSR ErrorBoundary caught an error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: "40px", color: "#f87171", background: "#0f172a", height: "100vh", fontFamily: "sans-serif" }}>
          <h2>⚠️ GeoSpecSR UI Rendering Error</h2>
          <pre style={{ background: "#1e293b", padding: "16px", borderRadius: "8px", color: "#cbd5e1", overflow: "auto" }}>
            {this.state.error?.toString()}
          </pre>
          <button
            onClick={() => window.location.reload()}
            style={{ marginTop: "16px", padding: "10px 20px", background: "#2563eb", color: "#fff", border: 0, borderRadius: "8px", cursor: "pointer" }}
          >
            🔄 Reload Application
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const safeFixed = (val, dec = 4) => (typeof val === 'number' && !isNaN(val)) ? val.toFixed(dec) : '0.0000';

function MapLocationEvents({ onSelectLocation }) {
  useMapEvents({
    click(e) {
      onSelectLocation([e.latlng.lat, e.latlng.lng]);
    }
  });
  return null;
}

function MapController({ center }) {
  const map = useMap();
  useEffect(() => {
    map.flyTo(center, 14, { duration: 1.0 });
  }, [center, map]);
  return null;
}

function ThreeDViewer({ imageUrl, heightScale = 1.5, sunAngle = 45, onHeightChange, onSunChange }) {
  const mountRef = useRef(null);
  const [autoRotate, setAutoRotate] = useState(true);

  useEffect(() => {
    if (!mountRef.current || !imageUrl) return;

    const width = mountRef.current.clientWidth || 800;
    const height = mountRef.current.clientHeight || 600;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060913);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, -160, 130);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    mountRef.current.innerHTML = "";
    mountRef.current.appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const rad = (sunAngle * Math.PI) / 180;
    const sunLight = new THREE.DirectionalLight(0xfffaed, 1.6);
    sunLight.position.set(Math.cos(rad) * 200, Math.sin(rad) * 200, 150);
    scene.add(sunLight);

    const textureLoader = new THREE.TextureLoader();
    textureLoader.load(imageUrl, (texture) => {
      texture.colorSpace = THREE.SRGBColorSpace;
      
      const geometry = new THREE.PlaneGeometry(180, 180, 128, 128);
      
      const canvas = document.createElement("canvas");
      canvas.width = 128;
      canvas.height = 128;
      const ctx = canvas.getContext("2d");
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = imageUrl;
      img.onload = () => {
        ctx.drawImage(img, 0, 0, 128, 128);
        const imgData = ctx.getImageData(0, 0, 128, 128).data;
        const pos = geometry.attributes.position;

        for (let i = 0; i < pos.count; i++) {
          const r = imgData[i * 4];
          const g = imgData[i * 4 + 1];
          const b = imgData[i * 4 + 2];
          const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
          pos.setZ(i, lum * 22 * heightScale);
        }
        geometry.computeVertexNormals();
        pos.needsUpdate = true;
      };

      const material = new THREE.MeshStandardMaterial({
        map: texture,
        roughness: 0.4,
        metalness: 0.1,
        side: THREE.DoubleSide
      });

      const terrainMesh = new THREE.Mesh(geometry, material);
      scene.add(terrainMesh);

      const gridHelper = new THREE.GridHelper(240, 24, 0x00e5ff, 0x1e293b);
      gridHelper.rotation.x = Math.PI / 2;
      gridHelper.position.z = -1;
      scene.add(gridHelper);

      let animId;
      let rot = 0;

      const animate = () => {
        animId = requestAnimationFrame(animate);
        if (autoRotate) {
          rot += 0.005;
          camera.position.x = Math.sin(rot) * 170;
          camera.position.y = -Math.cos(rot) * 170;
          camera.lookAt(0, 0, 10);
        }
        renderer.render(scene, camera);
      };
      animate();

      let isDragging = false;
      let prevMouse = { x: 0, y: 0 };

      const onMouseDown = (e) => {
        isDragging = true;
        prevMouse = { x: e.clientX, y: e.clientY };
      };

      const onMouseMove = (e) => {
        if (!isDragging) return;
        const dx = e.clientX - prevMouse.x;
        const dy = e.clientY - prevMouse.y;
        prevMouse = { x: e.clientX, y: e.clientY };

        camera.position.x -= dx * 0.5;
        camera.position.y += dy * 0.5;
        camera.lookAt(0, 0, 0);
      };

      const onMouseUp = () => { isDragging = false; };
      const el = mountRef.current;

      el?.addEventListener("mousedown", onMouseDown);
      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", onMouseUp);

      return () => {
        cancelAnimationFrame(animId);
        el?.removeEventListener("mousedown", onMouseDown);
        window.removeEventListener("mousemove", onMouseMove);
        window.removeEventListener("mouseup", onMouseUp);
      };
    });

  }, [imageUrl, heightScale, sunAngle, autoRotate]);

  return (
    <div className="threeD-viewport-container">
      <div ref={mountRef} className="threeD-canvas-container" />
      <div className="threeD-hud-controls">
        <div className="threeD-control-row">
          <label>
            <span>🏔️ 3D Height Relief</span>
            <span>{heightScale.toFixed(1)}x</span>
          </label>
          <input
            type="range"
            min="0.2"
            max="3.0"
            step="0.1"
            value={heightScale}
            onChange={(e) => onHeightChange(parseFloat(e.target.value))}
          />
        </div>
        <div className="threeD-control-row">
          <label>
            <span>☀️ Sun Lighting Angle</span>
            <span>{sunAngle}°</span>
          </label>
          <input
            type="range"
            min="0"
            max="360"
            step="15"
            value={sunAngle}
            onChange={(e) => onSunChange(parseInt(e.target.value))}
          />
        </div>
        <button
          className={`gis-mode-btn ${autoRotate ? "active" : ""}`}
          onClick={() => setAutoRotate(!autoRotate)}
          style={{ width: "100%", justifyContent: "center" }}
        >
          {autoRotate ? "⏸️ Pause 3D Orbit" : "▶️ Auto-Rotate 3D Orbit"}
        </button>
      </div>
      <div className="threeD-badge-tag">
        <span className="pulse-dot" /> 🎮 3D SUPER REALISTIC SATELLITE TERRAIN MESH
      </div>
    </div>
  );
}

function App() {
  const [samples, setSamples] = useState([]);
  const [activeSampleId, setActiveSampleId] = useState("hyderabad");
  const [center, setCenter] = useState([17.3850, 78.4867]);
  const [circleCenter, setCircleCenter] = useState([17.3850, 78.4867]);
  const [circleRadius, setCircleRadius] = useState(1500);
  const [targetResolution, setTargetResolution] = useState(2.0);
  const [appMode, setAppMode] = useState("crop"); // "crop", "urban", "disaster"
  const [gisMode, setGisMode] = useState("2d_map"); // "2d_map", "3d_mesh"
  const [tileProvider, setTileProvider] = useState("satellite_labels"); // "satellite_labels", "osm", "satellite_pure"
  const [heightScale, setHeightScale] = useState(1.5);
  const [sunAngle, setSunAngle] = useState(45);
  const [file, setFile] = useState(null);
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [viewMode, setViewMode] = useState("output_sr");
  const [sliderPos, setSliderPos] = useState(50);
  const [searchQuery, setSearchQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [isFullInspectorOpen, setIsFullInspectorOpen] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [backendStatus, setBackendStatus] = useState("connecting");
  const [gpuDevice, setGpuDevice] = useState("Apple Silicon MPS GPU");
  const [markedLocationName, setMarkedLocationName] = useState("");

  const [magnifierPos, setMagnifierPos] = useState({ x: 0, y: 0, show: false });
  const previewRef = useRef(null);

  useEffect(() => {
    fetchBackendInfo();
  }, []);

  const fetchBackendInfo = async () => {
    try {
      const healthRes = await axios.get(`${API}/health`);
      if (healthRes.data) {
        setBackendStatus("online");
        if (healthRes.data.device) {
          setGpuDevice(healthRes.data.device.toUpperCase() === "MPS" ? "Apple Silicon MPS GPU" : healthRes.data.device);
        }
      }
      const samplesRes = await axios.get(`${API}/samples`);
      if (samplesRes.data?.samples) {
        setSamples(samplesRes.data.samples);
      }
    } catch (e) {
      setBackendStatus("offline");
    }
  };

  const [selectedSigma, setSelectedSigma] = useState(2);
  const [activeGridTile, setActiveGridTile] = useState("Original image");

  const processPipeline = async (sampleOverride = null, uploadedFileOverride = null) => {
    setError("");
    setBusy(true);
    setProgress(15);

    if (sampleOverride) {
      setFile(null);
    }

    const activeFile = sampleOverride ? null : (uploadedFileOverride || file);

    try {
      let response;
      if (activeFile) {
        const form = new FormData();
        form.append("file", activeFile);
        for (const p of [30, 55, 78, 92]) {
          await new Promise(r => setTimeout(r, 180));
          setProgress(p);
        }
        response = await axios.post(`${API}/upload?target_resolution_m=${targetResolution}&blur_sigma=${selectedSigma}`, form);
      } else {
        const targetSample = sampleOverride || (activeSampleId === "marked_custom" ? "marked_custom" : activeSampleId);
        for (const p of [35, 60, 82, 95]) {
          await new Promise(r => setTimeout(r, 160));
          setProgress(p);
        }
        response = await axios.post(`${API}/process`, {
          sample_id: targetSample,
          target_resolution_m: targetResolution,
          lat: circleCenter[0],
          lng: circleCenter[1]
        });
      }

      setProgress(100);
      setJob(response.data);
      
      if (appMode === "crop") setViewMode("agriculture");
      else if (appMode === "urban") setViewMode("edge");
      else if (appMode === "disaster") setViewMode("uncertainty");
      else setViewMode("output_sr");
    } catch (e) {
      setError("Backend server error. Ensure uvicorn is running at http://127.0.0.1:8000.");
    } finally {
      setBusy(false);
    }
  };

  const selectSample = (sample) => {
    setActiveSampleId(sample.id);
    setCenter([sample.lat, sample.lng]);
    setCircleCenter([sample.lat, sample.lng]);
    setMarkedLocationName(sample.name);
    setFile(null);
  };

  const handleMapClickLocation = async (coords, nameOverride = "") => {
    setActiveSampleId("marked_custom");
    setCircleCenter(coords);
    setCenter(coords);
    setFile(null);
    if (nameOverride) {
      setMarkedLocationName(nameOverride);
    } else {
      try {
        const res = await axios.get(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${coords[0]}&lon=${coords[1]}`);
        if (res.data && res.data.display_name) {
          const parts = res.data.display_name.split(',');
          const shortName = parts.slice(0, 3).join(',').trim();
          setMarkedLocationName(shortName);
        } else {
          setMarkedLocationName(`Location (${safeFixed(coords[0])}°, ${safeFixed(coords[1])}°)`);
        }
      } catch (e) {
        setMarkedLocationName(`Location (${safeFixed(coords[0])}°, ${safeFixed(coords[1])}°)`);
      }
    }
  };

  const handleLocationSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    const coordsMatch = searchQuery.match(/^(-?\d+(\.\d+)?),\s*(-?\d+(\.\d+)?)$/);
    if (coordsMatch) {
      const lat = parseFloat(coordsMatch[1]);
      const lng = parseFloat(coordsMatch[3]);
      await handleMapClickLocation([lat, lng]);
      setSearchQuery("");
      return;
    }

    setSearching(true);
    try {
      const res = await axios.get(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}`);
      if (res.data && res.data.length > 0) {
        const lat = parseFloat(res.data[0].lat);
        const lng = parseFloat(res.data[0].lon);
        const parts = res.data[0].display_name.split(',');
        const displayName = parts.slice(0, 3).join(',').trim();
        await handleMapClickLocation([lat, lng], displayName);
        setSearchQuery("");
      } else {
        setError("Location not found. Enter city name or coordinates (e.g. 17.38, 78.48).");
      }
    } catch (err) {
      setError("Geocoding lookup failed.");
    } finally {
      setSearching(false);
    }
  };

  const handleFileChange = (e) => {
    const uploaded = e.target.files?.[0] || null;
    if (uploaded) {
      setFile(uploaded);
      setJob(null);
      processPipeline(null, uploaded);
    }
  };

  const handleFileDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const dropped = e.dataTransfer.files[0];
      setFile(dropped);
      setJob(null);
      processPipeline(null, dropped);
    }
  };

  const downloadReportJson = () => {
    if (!job) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(job, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `GeoSpecSR_Validation_Report_${job.job_id}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const handleMouseMoveMagnifier = (e) => {
    if (!previewRef.current) return;
    const { left, top, width, height } = previewRef.current.getBoundingClientRect();
    const x = e.clientX - left;
    const y = e.clientY - top;
    setMagnifierPos({ x, y, width, height, show: true });
  };

  const getActiveSampleMeta = () => {
    if (file) {
      return {
        name: `Uploaded File: ${file.name}`,
        input_resolution: "10 m Sentinel-2 Blur Image",
        target_resolution: `<${targetResolution} m Clear Super-Resolved`,
        sensor: "Sentinel-2 MSI / Custom Input",
        description: `Uploaded 10m blurred image processed into <${targetResolution}m clear spatial resolution.`
      };
    }
    if (activeSampleId === "marked_custom") {
      return {
        name: markedLocationName || `Marked Location (${safeFixed(circleCenter?.[0])}°, ${safeFixed(circleCenter?.[1])}°)`,
        input_resolution: "10 m (Sentinel-2 L2A)",
        target_resolution: `${targetResolution} m (GeoSpecSR)`,
        sensor: "Sentinel-2 MSI",
        description: `User marked location at Lat ${safeFixed(circleCenter?.[0])}°, Lng ${safeFixed(circleCenter?.[1])}°.`
      };
    }
    return samples.find(s => s.id === activeSampleId) || {
      name: markedLocationName || "Hyderabad — Hitec City & Durgam Cheruvu",
      input_resolution: "10 m (Sentinel-2 L2A)",
      target_resolution: `${targetResolution} m (GeoSpecSR)`,
      sensor: "Sentinel-2 MSI",
      description: "Urban commercial sector, tech park building roofs, water shoreline & flyovers."
    };
  };

  const meta = getActiveSampleMeta();

  return (
    <div className="app dark-theme">
      {/* Top Navbar */}
      <header className="topbar">
        <div className="brand">
          <div className="logo-glow">
            <span className="logo-icon">🛰️</span>
          </div>
          <div>
            <div className="brand-title">
              GeoSpec<span className="accent">SR</span>
              <span className="ver-badge">10m Blur → &lt;4m Clear</span>
            </div>
            <div className="brand-sub">Generative Earth Observation Enhancement (10m Sentinel-2 Blur Input → &lt;4m Clear Output)</div>
          </div>
        </div>

        {/* Application Preset Modes Switcher */}
        <div className="app-mode-switcher">
          <button
            className={`mode-tab ${appMode === "crop" ? "active" : ""}`}
            onClick={() => { setAppMode("crop"); if (job) setViewMode("agriculture"); }}
          >
            🌾 Crop Monitoring
          </button>
          <button
            className={`mode-tab ${appMode === "urban" ? "active" : ""}`}
            onClick={() => { setAppMode("urban"); if (job) setViewMode("edge"); }}
          >
            🏙️ Urban Mapping
          </button>
          <button
            className={`mode-tab ${appMode === "disaster" ? "active" : ""}`}
            onClick={() => { setAppMode("disaster"); if (job) setViewMode("uncertainty"); }}
          >
            🚨 Disaster Response
          </button>
        </div>

        <div className="topbar-actions">
          <div className={`status-pill ${backendStatus}`}>
            <span className="pulse-dot" />
            <span>{backendStatus === "online" ? `PyTorch ${gpuDevice}` : "Backend Offline"}</span>
          </div>
          {job && (
            <button className="export-report-btn" onClick={downloadReportJson}>
              📥 Export Report
            </button>
          )}
          <button className="icon-btn" title="Reload Backend Info" onClick={fetchBackendInfo}>🔄</button>
        </div>
      </header>

      {/* Main Grid Studio */}
      <div className="workspace">
        {/* Left Sidebar Control Dock */}
        <aside className="sidebar">
          {/* Target Resolution Selector */}
          <section className="side-card highlight-card">
            <div className="card-header">
              <span className="card-icon">🎯</span>
              <h3>Target Clear Resolution</h3>
            </div>
            <div className="target-res-selector" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
              {[
                { res: 0.15, label: "<15 cm", desc: "Maxar/Airbus HD" },
                { res: 0.50, label: "<50 cm", desc: "Sub-Meter HD" },
                { res: 1.0, label: "<1.0 m", desc: "Fine Feature" },
                { res: 2.0, label: "<2.0 m", desc: "Ultra-Sharp" },
                { res: 2.5, label: "<2.5 m", desc: "High Precision" },
                { res: 4.0, label: "<4.0 m", desc: "Enhanced Clear" }
              ].map(r => (
                <button
                  key={r.res}
                  className={`target-res-btn ${targetResolution === r.res ? "active" : ""}`}
                  onClick={() => {
                    setTargetResolution(r.res);
                    if (job || file) processPipeline();
                  }}
                >
                  <span className="res-val">{r.label}</span>
                  <span className="res-desc">{r.desc}</span>
                </button>
              ))}
            </div>
          </section>

          {/* Blur Level / Degradation Sigma Selector */}
          <section className="side-card">
            <div className="card-header">
              <span className="card-icon">🌀</span>
              <h3>Input Blur Level (Sigma)</h3>
            </div>
            <div className="sigma-selector-row">
              {[1, 2, 3, 4, 5].map(s => (
                <button
                  key={s}
                  className={`sigma-btn ${selectedSigma === s ? "active" : ""}`}
                  onClick={() => {
                    setSelectedSigma(s);
                    if (job || file) processPipeline();
                  }}
                >
                  Sigma={s}
                </button>
              ))}
            </div>
            <small style={{ color: "#94a3b8", display: "block", marginTop: "6px" }}>
              Controls input degradation matrix ($\sigma=1\dots5$).
            </small>
          </section>

          {/* Drag & Drop 10m Blur Image Upload */}
          <section className="side-card upload-card-primary">
            <div className="card-header">
              <span className="card-icon">⬆️</span>
              <h3>Upload 10m Blur Satellite Image</h3>
            </div>
            <div
              className={`upload-dropzone ${dragOver ? "dragover" : ""} ${file ? "has-file" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleFileDrop}
            >
              <input
                type="file"
                id="file-input"
                accept="image/*,.tif,.tiff"
                onChange={handleFileChange}
              />
              <label htmlFor="file-input" className="dropzone-label">
                <div className="upload-glow-icon">📡</div>
                {file ? (
                  <div className="file-preview-name">
                    <strong>{file.name}</strong>
                    <small>{(file.size / 1024).toFixed(1)} KB • Transforming to &lt;{targetResolution}m Clear</small>
                  </div>
                ) : (
                  <div>
                    <strong>Drop 10m Blur Satellite Image</strong>
                    <small>Supports PNG, JPG, GeoTIFF images.<br/>Outputs &lt;4m clear imagery & Sigma 1..5 differentiation matrix.</small>
                  </div>
                )}
              </label>
            </div>
          </section>

          {/* Marked Location & Red Circle ROI */}
          <section className="side-card">
            <div className="card-header">
              <span className="card-icon">📍</span>
              <h3>Map Study Area ROI</h3>
            </div>
            <div className="roi-info-box">
              <div className="roi-coords">
                <b>Geospatial Center:</b>
                <span>{safeFixed(circleCenter?.[0])}°, {safeFixed(circleCenter?.[1])}°</span>
              </div>
              <div className="roi-radius">
                <label>Radius: <b>{(circleRadius / 1000).toFixed(1)} km</b></label>
                <input
                  type="range"
                  min="500"
                  max="3000"
                  step="100"
                  value={circleRadius}
                  onChange={e => setCircleRadius(Number(e.target.value))}
                />
              </div>
              <small>👉 Click map or search location to set ROI circle!</small>
            </div>
          </section>

          {/* Preset Satellite Datasets */}
          <section className="side-card">
            <div className="card-header">
              <span className="card-icon">🏙️</span>
              <h3>Preset Study Areas</h3>
            </div>
            <div className="sample-grid">
              {[
                { id: "hyderabad", name: "Hyderabad", tag: "Urban & Water", coords: [17.3850, 78.4867] },
                { id: "bengaluru", name: "Bengaluru", tag: "Tech Parks & Lakes", coords: [12.9716, 77.5946] },
                { id: "delhi", name: "Delhi", tag: "Grid & River Corridor", coords: [28.6139, 77.2090] },
                { id: "agriculture", name: "Punjab", tag: "Crop Parcels & Canals", coords: [30.9010, 75.8573] }
              ].map(s => (
                <div
                  key={s.id}
                  className={`sample-item ${activeSampleId === s.id && !file ? "active" : ""}`}
                  onClick={() => {
                    selectSample(s);
                    processPipeline(s.id);
                  }}
                >
                  <div className="sample-info">
                    <span className="sample-name">{s.name}</span>
                    <span className="sample-tag">{s.tag}</span>
                  </div>
                  <span className="arrow-indicator">→</span>
                </div>
              ))}
            </div>
          </section>

          {/* Execute Button */}
          <button className="execute-btn" disabled={busy} onClick={() => processPipeline()}>
            <span className="btn-shine" />
            {busy ? (
              <span className="busy-text">
                <span className="spinner" /> Transforming 10m Blur → &lt;{targetResolution}m Clear {progress}%
              </span>
            ) : (
              <span>✨ Run 10m Blur → &lt;{targetResolution}m Clear Pipeline</span>
            )}
          </button>

          {error && <div className="error-banner">⚠️ {error}</div>}
        </aside>

        {/* Center Interactive GIS Viewport */}
        <main className="map-area">
          {/* Global Location Search Bar */}
          <form className="map-search-bar" onSubmit={handleLocationSearch}>
            <input
              type="text"
              placeholder="Search location or enter Lat, Lng (e.g. Paris, Tokyo, 17.38, 78.48)..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
            />
            <button type="submit" disabled={searching}>
              {searching ? "..." : "🔎 Search"}
            </button>
          </form>

          <MapContainer center={center} zoom={14} zoomControl={false} className="leaflet-map">
            <MapController center={center} />
            <MapLocationEvents onSelectLocation={handleMapClickLocation} />

            {tileProvider === "satellite_labels" && (
              <>
                <TileLayer
                  attribution='&copy; Esri World Imagery'
                  url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                />
                <TileLayer
                  attribution='&copy; Esri Transportation'
                  url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}"
                />
                <TileLayer
                  attribution='&copy; Esri Places'
                  url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
                />
              </>
            )}

            {tileProvider === "osm" && (
              <TileLayer
                attribution='&copy; OpenStreetMap contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
            )}

            {tileProvider === "satellite_pure" && (
              <TileLayer
                attribution='&copy; Esri World Imagery'
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              />
            )}

            {/* Marked Red Circle ROI on Map */}
            <Circle
              center={circleCenter}
              radius={circleRadius}
              pathOptions={{
                color: "#ef4444",
                fillColor: "#ef4444",
                fillOpacity: 0.25,
                weight: 3,
                dashArray: "6 4"
              }}
            />

            <Marker position={circleCenter} icon={markerIcon}>
              <Popup>
                <div className="popup-card">
                  <b>📍 {markedLocationName || meta.name || "Marked Location"}</b>
                  <p>Lat: {safeFixed(circleCenter?.[0])}°, Lng: {safeFixed(circleCenter?.[1])}°</p>
                  <button className="popup-run-btn" onClick={() => { setActiveSampleId("marked_custom"); setFile(null); processPipeline("marked_custom"); }}>
                    ✨ Generate &lt;{targetResolution}m Clear Image
                  </button>
                </div>
              </Popup>
            </Marker>
          </MapContainer>

          {/* HUD Overlays */}
          <div className="hud-top-left">
            <div className="hud-card">
              <span className="hud-label">10M BLUR INPUT → &lt;{targetResolution}M CLEAR TARGET</span>
              <div className="hud-title">{markedLocationName || meta.name}</div>
              <div className="hud-sub">Lat: {safeFixed(circleCenter?.[0])}° • Lng: {safeFixed(circleCenter?.[1])}°</div>
            </div>
          </div>

          <div className="hud-top-right">
            <div className="map-tile-switcher">
              <button
                className={`tile-tab ${tileProvider === "satellite_labels" ? "active" : ""}`}
                onClick={() => setTileProvider("satellite_labels")}
                title="Satellite Imagery with City & Road Names"
              >
                🛰️ Satellite + Labels
              </button>
              <button
                className={`tile-tab ${tileProvider === "osm" ? "active" : ""}`}
                onClick={() => setTileProvider("osm")}
                title="OpenStreetMap Street View with Location Names"
              >
                🗺️ Street Map
              </button>
              <button
                className={`tile-tab ${tileProvider === "satellite_pure" ? "active" : ""}`}
                onClick={() => setTileProvider("satellite_pure")}
                title="Pure Satellite Imagery without Labels"
              >
                🌐 Pure Satellite
              </button>
            </div>
            <button className="hud-btn highlight" onClick={() => processPipeline()}>
              ✨ Transform to &lt;{targetResolution}m Clear
            </button>
            <button className="hud-btn" title="Inspect Fullscreen" onClick={() => setIsFullInspectorOpen(true)}>🔍 Inspect Studio</button>
          </div>

          <div className="hud-bottom-legend">
            <div className="legend-chip"><span className="leg-box red" /> Red Circle ROI</div>
            <div className="legend-chip"><span className="leg-box blue" /> Output: &lt;{targetResolution}m Clear Image</div>
          </div>
        </main>

        {/* Right Super-Resolution Analytics Studio */}
        <aside className="right-panel">
          <div className="studio-head">
            <div>
              <span className="studio-sub">SUPER-RESOLUTION STUDIO</span>
              <h2>&lt;{targetResolution}m Clear Output</h2>
            </div>
            <span className="badge-live">&lt;{targetResolution}m CLEAR</span>
          </div>

          {/* Model Output Card */}
          <div className="studio-results">
            {!job ? (
              <div className="placeholder-results">
                <div className="satellite-radar-icon">📡</div>
                <h4>Upload 10m blur image or click process</h4>
                <p>
                  Upload any <b>10m blurred satellite image</b> to automatically transform it into a <b>&lt;4m sharp clear image</b> with PSNR/SSIM metrics and blur differentiation grid (Sigma 1..5).
                </p>
              </div>
            ) : (
              <div className="active-results">
                {/* 3-Step Super-Resolution Architecture Pipeline Banner */}
                <div className="visual-flow-card">
                  <div className="visual-flow-header">
                    <span className="flow-badge">SUPER-RESOLUTION PIPELINE FLOW</span>
                  </div>
                  <div className="visual-flow-pipeline">
                    <div className="flow-step">
                      <span className="step-title">10 m Sentinel-2 Blur Input</span>
                      <div className="flow-img-box input">
                        <img src={`${API}${job.urls.input_lr}`} alt="10m Sentinel-2 Blur Input" />
                      </div>
                    </div>
                    <div className="flow-arrow">➔</div>
                    <div className="flow-step">
                      <span className="step-title">&lt;{targetResolution}m GeoSpec-SR Output</span>
                      <div className="flow-img-box sr">
                        <img src={`${API}${job.urls.output_sr}`} alt="<2m GeoSpec-SR Output" />
                      </div>
                    </div>
                    <div className="flow-arrow">➔</div>
                    <div className="flow-step">
                      <span className="step-title">Spatial Uncertainty Map</span>
                      <div className="flow-img-box uncertainty">
                        <img src={`${API}${job.urls.uncertainty}`} alt="Spatial Uncertainty Map" />
                      </div>
                    </div>
                  </div>
                </div>

                {/* View Mode Tabs */}
                <div className="view-mode-tabs">
                  {[
                    { key: "output_sr", label: `✨ <${targetResolution}m Clear SR` },
                    { key: "differentiation_grid", label: "📊 Blur Differentiation Grid" },
                    { key: "agriculture", label: "🌾 Agriculture (NDVI)" },
                    { key: "edge", label: "📐 SR Edge Map" },
                    { key: "uncertainty", label: "🛡️ Uncertainty Map" },
                    { key: "nir", label: "🔴 False Color NIR" },
                    { key: "input_lr", label: "📷 10m Blur Input" },
                    { key: "heatmap", label: "🔥 Error Heatmap" }
                  ].map(tab => (
                    (job.urls[tab.key] || tab.key === "differentiation_grid") && (
                      <button
                        key={tab.key}
                        className={`view-tab-btn ${viewMode === tab.key ? "active" : ""}`}
                        onClick={() => setViewMode(tab.key)}
                      >
                        {tab.label}
                      </button>
                    )
                  ))}
                </div>

                {/* Differentiation Grid View (Interactive 2x3 Grid with Cyan Banners) */}
                {viewMode === "differentiation_grid" ? (
                  <div className="differentiation-container">
                    <div className="diff-header-bar">
                      <div>
                        <strong>📊 Blur Differentiation Matrix (Sigma 1 to 5 vs Ground Truth / &lt;4m Clear)</strong>
                        <p>Comparing blur degradation levels against high-frequency restored spatial details.</p>
                      </div>
                    </div>

                    <div className="differentiation-grid-2x3">
                      {[
                        { title: "Original image", url: job.urls.output_sr, isOrig: true },
                        { title: "Blurry image, Sigma=1", url: job.urls.sigma_1 || job.urls.input_lr, isOrig: false },
                        { title: "Blurry image, Sigma=2", url: job.urls.sigma_2 || job.urls.input_lr, isOrig: false },
                        { title: "Blurry image, Sigma=3", url: job.urls.sigma_3 || job.urls.input_lr, isOrig: false },
                        { title: "Blurry image, Sigma=4", url: job.urls.sigma_4 || job.urls.input_lr, isOrig: false },
                        { title: "Blurry image, Sigma=5", url: job.urls.sigma_5 || job.urls.input_lr, isOrig: false }
                      ].map((item, i) => (
                        <div key={i} className={`diff-tile-card ${activeGridTile === item.title ? "selected" : ""}`} onClick={() => setActiveGridTile(item.title)}>
                          <div className="cyan-banner-header">
                            <span>{item.title}</span>
                          </div>
                          <div className="diff-tile-img-box">
                            <img src={`${API}${item.url}`} alt={item.title} />
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Comparative Sigma Metrics Table */}
                    {job.sigma_metrics && (
                      <div className="sigma-metrics-table-card">
                        <h4>📈 Blur Degradation vs Reconstruction Metrics</h4>
                        <table className="sigma-table">
                          <thead>
                            <tr>
                              <th>Blur Level</th>
                              <th>PSNR (dB)</th>
                              <th>SSIM</th>
                              <th>Spatial Ambiguity</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr className="highlight-row">
                              <td>✨ Clear SR (&lt;{targetResolution}m)</td>
                              <td><b>{job.metrics.psnr} dB</b></td>
                              <td><b>{job.metrics.ssim}</b></td>
                              <td>0m (Sharp Object Detail)</td>
                            </tr>
                            {job.sigma_metrics.map(sm => (
                              <tr key={sm.sigma}>
                                <td>Blurry image, Sigma={sm.sigma}</td>
                                <td>{sm.psnr} dB</td>
                                <td>{sm.ssim}</td>
                                <td>{sm.blur_description}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                ) : (
                  /* Primary Image Preview Container with Magnifier Lens */
                  <div
                    className="preview-viewport"
                    ref={previewRef}
                    onMouseMove={handleMouseMoveMagnifier}
                    onMouseLeave={() => setMagnifierPos(prev => ({ ...prev, show: false }))}
                  >
                    <img src={`${API}${job.urls[viewMode] || job.urls.output_sr}`} alt="SR Layer Preview" className="preview-img" />
                    
                    {/* Hover Pixel Magnifier Lens */}
                    {magnifierPos.show && (
                      <div
                        className="magnifier-lens"
                        style={{
                          left: `${magnifierPos.x - 50}px`,
                          top: `${magnifierPos.y - 50}px`,
                          backgroundImage: `url(${API}${job.urls[viewMode] || job.urls.output_sr})`,
                          backgroundPosition: `-${magnifierPos.x * 2 - 50}px -${magnifierPos.y * 2 - 50}px`,
                          backgroundSize: `${magnifierPos.width * 2}px ${magnifierPos.height * 2}px`
                        }}
                      />
                    )}

                    <div className="preview-tag">
                      {viewMode === "output_sr" && `GeoSpecSR <${targetResolution}m Clear Super-Resolved Image`}
                      {viewMode === "agriculture" && "Agricultural NDVI Vegetation Index & Crop Canopy"}
                      {viewMode === "edge" && "SR Structural Edge Detection (Small Buildings & Narrow Roads)"}
                      {viewMode === "uncertainty" && "Spatial Uncertainty Map (Blue = Directly Observed, Yellow/Red = Model Inferred)"}
                      {viewMode === "nir" && "False Color Near-Infrared (NIR) Band"}
                      {viewMode === "input_lr" && "10m Blur Satellite Image Input"}
                      {viewMode === "heatmap" && "Reconstruction Error Heatmap"}
                    </div>
                  </div>
                )}

                {/* Interactive Split Slider (10m Blur vs <4m Clear) */}
                <div className="split-comparison">
                  <div className="split-header">
                    <span>10m Blur Input vs &lt;{targetResolution}m Clear Output</span>
                    <small>Drag handle to compare</small>
                  </div>
                  <div className="split-box">
                    <img src={`${API}${job.urls.output_sr}`} alt="SR Clear" className="split-img base" />
                    <div className="split-overlay" style={{ width: `${sliderPos}%` }}>
                      <img src={`${API}${job.urls.input_lr}`} alt="Blur Input 10m" className="split-img overlay" />
                      <span className="split-tag left">10m Blur Input</span>
                    </div>
                    <span className="split-tag right">&lt;{targetResolution}m Clear SR</span>
                    <div className="slider-divider" style={{ left: `${sliderPos}%` }}>
                      <div className="slider-knob">↔</div>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={sliderPos}
                      onChange={e => setSliderPos(e.target.value)}
                      className="split-range-input"
                    />
                  </div>
                </div>

                {/* Uncertainty & Reliability Management Card */}
                {job.uncertainty && (
                  <div className="uncertainty-card">
                    <div className="uncert-header">
                      <span>🛡️ Uncertainty & Scientific Reliability</span>
                      <span className="conf-badge">{job.uncertainty.confidence_score_pct}% Confidence</span>
                    </div>
                    <div className="uncert-grid">
                      <div className="uncert-item">
                        <small>Directly Observed Data</small>
                        <b>{job.uncertainty.directly_observed_pct}%</b>
                      </div>
                      <div className="uncert-item">
                        <small>Model Inferred Detail</small>
                        <b>{job.uncertainty.model_inferred_pct}%</b>
                      </div>
                    </div>
                  </div>
                )}

                {/* RGB Color & Geometry Object Breakdown Card */}
                {job.object_breakdown && (
                  <div className="object-breakdown-card">
                    <div className="obj-title">🔍 Remote Sensing Feature Classification</div>
                    <div className="obj-grid">
                      <div className="obj-item">
                        <small>🏢 Buildings & Structures</small>
                        <div className="obj-bar-row">
                          <b>{job.object_breakdown.buildings_structures_pct}%</b>
                          <div className="obj-progress"><div className="obj-fill blue" style={{ width: `${job.object_breakdown.buildings_structures_pct}%` }} /></div>
                        </div>
                      </div>
                      <div className="obj-item">
                        <small>🛣️ Narrow Roads & Infrastructure</small>
                        <div className="obj-bar-row">
                          <b>{job.object_breakdown.road_networks_pct}%</b>
                          <div className="obj-progress"><div className="obj-fill orange" style={{ width: `${job.object_breakdown.road_networks_pct}%` }} /></div>
                        </div>
                      </div>
                      <div className="obj-item">
                        <small>🌾 Agricultural Canopy</small>
                        <div className="obj-bar-row">
                          <b>{job.object_breakdown.vegetation_canopy_pct}%</b>
                          <div className="obj-progress"><div className="obj-fill green" style={{ width: `${job.object_breakdown.vegetation_canopy_pct}%` }} /></div>
                        </div>
                      </div>
                      <div className="obj-item">
                        <small>🌊 Water Bodies & Canals</small>
                        <div className="obj-bar-row">
                          <b>{job.object_breakdown.water_bodies_pct}%</b>
                          <div className="obj-progress"><div className="obj-fill cyan" style={{ width: `${job.object_breakdown.water_bodies_pct}%` }} /></div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Metric Performance Dashboard */}
                <div className="metrics-dashboard">
                  <div className="metric-card">
                    <div className="metric-header">
                      <span className="m-title">PSNR</span>
                      <span className="m-gain">+{job.metrics.psnr_gain} dB</span>
                    </div>
                    <div className="m-val">{job.metrics.psnr} <small>dB</small></div>
                    <div className="m-bar"><div className="m-fill" style={{ width: `${Math.min(100, job.metrics.psnr * 3)}%` }} /></div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-header">
                      <span className="m-title">SSIM</span>
                      <span className="m-gain">+{job.metrics.ssim_gain}</span>
                    </div>
                    <div className="m-val">{job.metrics.ssim}</div>
                    <div className="m-bar"><div className="m-fill purple" style={{ width: `${Math.min(100, job.metrics.ssim * 100)}%` }} /></div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-header">
                      <span className="m-title">RMSE</span>
                      <span className="m-gain green">Validated</span>
                    </div>
                    <div className="m-val">{job.metrics.rmse}</div>
                    <div className="m-bar"><div className="m-fill green" style={{ width: `${Math.max(10, 100 - job.metrics.rmse)}%` }} /></div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </aside>
      </div>

      {/* Fullscreen Inspector Modal */}
      {isFullInspectorOpen && job && (
        <div className="modal-overlay" onClick={() => setIsFullInspectorOpen(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>🔍 GeoSpecSR Validation Studio — 10m Blur → &lt;{targetResolution}m Clear Image</h3>
              <button className="close-btn" onClick={() => setIsFullInspectorOpen(false)}>✕</button>
            </div>
            <div className="modal-grid">
              <div className="modal-img-card highlight">
                <span className="modal-tag sr">✨ &lt;{targetResolution}m CLEAR SUPER-RESOLVED IMAGE</span>
                <img src={`${API}${job.urls.output_sr}`} alt="SR Clear Product" />
              </div>
              <div className="modal-img-card">
                <span className="modal-tag uncert">📷 10m BLUR INPUT IMAGE</span>
                <img src={`${API}${job.urls.input_lr}`} alt="10m Blur Input" />
              </div>
              <div className="modal-img-card">
                <span className="modal-tag edge">📐 SR GEOMETRY EDGE MAP</span>
                <img src={`${API}${job.urls.edge}`} alt="Edge Map" />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
);
