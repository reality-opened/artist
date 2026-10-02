#!/bin/sh
# Kill every camera server (wrist webcam, root-owned Gemini RGB-D, orphaned ffmpeg) and start them fresh.
#   ./restart_cameras.sh     asks for the sudo password once
cd "$(dirname "$0")" || exit 1
sudo -v || exit 1

./start_cameras.sh stop

# The Gemini SDK can take a few seconds to let go; a second server would steal the camera and segfault.
for _ in $(seq 1 20); do
    pgrep -f "camera_stream.py|ffmpeg .*USB Camera" >/dev/null || break
    sleep 0.5
done
if pgrep -f "camera_stream.py|ffmpeg .*USB Camera" >/dev/null; then
    echo "force-killing leftovers"
    sudo pkill -9 -f "camera_stream.py"
    pkill -9 -f "ffmpeg .*USB Camera"
    sleep 1
fi

./start_cameras.sh
