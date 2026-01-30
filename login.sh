#!/bin/bash
#SBATCH --job-name=VSCode_Login
#SBATCH --partition=RTX3090
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/vscode-login(%j).log

set -e

echo "[INFO] Running VS Code login container on $(hostname)"

docker run -it --rm \
  -v $HOME/.vscode:/root/.vscode \
  -v $HOME/.vscode-cli:/root/.vscode-cli \
  maskrcnn-vscode:glibc235 \
  bash
