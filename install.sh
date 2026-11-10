#!/bin/bash
# Install the vLLM -> Ollama API facade for NVIDIA PAIR.
#
#   sudo ./install.sh
#   sudo UPSTREAM=http://127.0.0.1:8888 PORT=11434 USER_NAME=machinegeorge ./install.sh
#
set -euo pipefail

UPSTREAM="${UPSTREAM:-http://127.0.0.1:8888}"
PORT="${PORT:-11434}"
USER_NAME="${USER_NAME:-${SUDO_USER:-$(id -un)}}"
DEST=/usr/local/bin/vllm_ollama_bridge.py
UNIT=/etc/systemd/system/vllm-ollama-bridge.service
SRC="$(cd "$(dirname "$0")" && pwd)/vllm_ollama_bridge.py"

if [ "$(id -u)" -ne 0 ]; then
  echo "run with sudo" >&2
  exit 1
fi

[ -f "$SRC" ] || { echo "vllm_ollama_bridge.py not found next to install.sh" >&2; exit 1; }

install -m 0755 "$SRC" "$DEST"

# `~` and $HOME are literals inside the unit, so the port is passed as an env var.
cat > "$UNIT" <<EOF
[Unit]
Description=vLLM -> Ollama API bridge (lets NVIDIA PAIR use a vLLM/SGLang server)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
Environment=VLLM_UPSTREAM=$UPSTREAM
Environment=BRIDGE_PORT=$PORT
ExecStart=/usr/bin/python3 $DEST
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now vllm-ollama-bridge.service
sleep 2

echo "--- status ---"
systemctl is-active vllm-ollama-bridge.service || true
echo "--- /api/version ---"
curl -s --max-time 5 "http://127.0.0.1:$PORT/api/version" || echo "(no answer)"
echo
echo "--- /api/tags ---"
curl -s --max-time 5 "http://127.0.0.1:$PORT/api/tags" || echo "(no answer)"
echo
cat <<'MSG'

Next:
  1. Start PAIR (nvpair) and press enter on your own node row.
  2. Engines pane should show Ollama INSTALLED/RUNNING/HEALTHY = yes.
  3. If /api/tags was empty, start your OpenAI server on :8888 — the model list
     fills in on the next refresh, no PAIR restart needed.
MSG
