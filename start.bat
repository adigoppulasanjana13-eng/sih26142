@echo off
echo 🚀 Starting GeoSpecSR — Hackathon Setup ^& Launch Script
echo --------------------------------------------------------

echo 📦 Installing Python backend dependencies...
cd backend
pip install -r requirements.txt

echo 📦 Installing React frontend dependencies...
cd ..\frontend
call npm install

echo ⚡ Building frontend production bundle...
call npm run build

echo --------------------------------------------------------
echo ✅ Launching GeoSpecSR at http://127.0.0.1:8000
echo --------------------------------------------------------
cd ..\backend
uvicorn main:app --host 127.0.0.1 --port 8000
