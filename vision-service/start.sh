#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
MAC_IP=$(ipconfig getifaddr en0)
if [ -z "$MAC_IP" ]; then
  echo "Mac is not connected to Wi-Fi."
  exit 1
fi
PREFIX=${MAC_IP%.*}
if [ -z "$CAM_IP" ]; then
  echo "Mac IP is $MAC_IP - looking for IP Camera on $PREFIX.x ..."
  FOUND=$(for i in $(seq 1 254); do
    [ "$PREFIX.$i" = "$MAC_IP" ] && continue
    (nc -z -G 1 "$PREFIX.$i" 8081 2>/dev/null && echo "$PREFIX.$i") &
  done; wait)
  CAM_IP=$(echo "$FOUND" | head -1)
fi
if [ -z "$CAM_IP" ]; then
  echo ""
  echo "Camera NOT found on $PREFIX.x"
  echo "Check:"
  echo "  1. Camera phone is on the SAME Wi-Fi/hotspot as this Mac"
  echo "  2. IP Camera app is open, server is ON, phone screen is unlocked"
  echo "Or set it by hand:  CAM_IP=<ip shown in the app> ./start.sh"
  exit 1
fi
echo "Camera found at $CAM_IP"
ROTATION=${ROTATION:-270} CAMERAS="cam1=http://admin:admin@$CAM_IP:8081/video" ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-http://localhost:3000}" uvicorn app:app --host 0.0.0.0 --port 8000
