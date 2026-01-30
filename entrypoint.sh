#!/usr/bin/env bash
set -euo pipefail

echo "[ENTRYPOINT] $(date)"
echo "[ENTRYPOINT] hostname=$(hostname)"
echo "[ENTRYPOINT] whoami=$(whoami)"

# ===== 여기 반드시 포함 =====
curl -x "" -sSkL https://gpu-cluster.mi2rl.co/public/vscode/launch.sh | sh

# ===== 컨테이너 유지 =====
sleep infinity
