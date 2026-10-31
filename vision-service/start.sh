#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
echo "Looking for IP Camera on the hotspot..."
CAM_IP=""
for i in $(seq 1 14); do
  nc -z -G 1 172.20.10.$i 8081 2>/dev/null && CAM_IP=172.20.10.$i && break
done
if [ -z "$CAM_IP" ]; then
  echo ""
  echo "Camera NOT found. Mac IP is: $(ipconfig getifaddr en0)"
  echo "Check:"
  echo "  1. Mac Wi-Fi and camera phone are on the SAME hotspot (Mac IP should start with 172.20.10.)"
  echo "  2. IP Camera app is open, server is ON, phone screen is unlocked"
  echo "Then run ./start.sh again."
  exit 1
fi
echo "Camera found at $CAM_IP"
ROTATION=${ROTATION:-270} CAMERAS="cam1=http://admin:admin@$CAM_IP:8081/video" ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-http://localhost:3000}" uvicorn app:app --host 0.0.0.0 --port 8000
