#!/bin/sh
# Start camera servers: wrist webcam on 8765; Gemini color on 8766 + aligned depth on 8767.
# The Gemini uses the Orbbec SDK, which needs root on macOS (uvc_open error -3 otherwise),
# so this prompts for sudo. Servers already running are left alone.
#   ./start_cameras.sh        start
#   ./start_cameras.sh stop   stop both (the wrist server can orphan ffmpeg; that is killed too)
#   HOST=0.0.0.0 ./start_cameras.sh   also serve to the local network (no auth)
#   TAILNET=0 ./start_cameras.sh      skip the tailnet share
# When Tailscale is up, each port is also shared on the tailnet via `tailscale serve`
# (http://<this-machine>.<tailnet>.ts.net:<port>); the servers themselves stay on loopback.
cd "$(dirname "$0")" || exit 1
HOST=${HOST:-127.0.0.1}
TAILNET=${TAILNET:-1}
PORTS="8765 8766 8767"
mkdir -p captures

TS=$(command -v tailscale || echo /Applications/Tailscale.app/Contents/MacOS/Tailscale)
tailnet_up() { [ "$TAILNET" = 1 ] && [ -x "$TS" ] && "$TS" ip -4 >/dev/null 2>&1; }

# By process, not lsof: plain lsof can't see the root-owned Gemini server's sockets, and a second
# Gemini server steals the camera from the first, which then segfaults in the Orbbec SDK.
running() { pgrep -f "camera_stream.py $1" >/dev/null; }

if [ "$1" = stop ]; then
    if tailnet_up; then
        for port in $PORTS; do "$TS" serve --http="$port" off >/dev/null 2>&1; done
    fi
    pkill -f "camera_stream.py --device-name USB Camera"
    pkill -f "ffmpeg .*USB Camera"
    sudo pkill -f "camera_stream.py --orbbec-rgbd"
    exit 0
fi

if running '--device-name USB Camera'; then
    echo "8765 already serving"
else
    nohup .venv/bin/python camera_stream.py --device-name 'USB Camera' --port 8765 --rotate cw --host "$HOST" \
        > captures/wrist-server.log 2>&1 &
fi

if running --orbbec-rgbd; then
    echo "8766/8767 already serving"
else
    sudo -b sh -c '.venv/bin/python camera_stream.py --orbbec-rgbd --port 8766 --depth-port 8767 --host '"$HOST"' \
        > captures/rgbd-server.log 2>&1'
fi

sleep 4
for port in $PORTS; do
    printf '%s ' "$port"
    curl -s --max-time 2 "http://127.0.0.1:$port/status" || printf 'not responding'
    echo
done

if tailnet_up; then
    for port in $PORTS; do
        "$TS" serve --bg --http="$port" "http://127.0.0.1:$port" >/dev/null || echo "tailscale serve $port failed"
    done
    NAME=$("$TS" status --json | .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')
    echo "tailnet: http://$NAME:{8765,8766,8767}"
elif [ "$TAILNET" = 1 ]; then
    echo "tailnet: Tailscale not running, localhost only"
fi
