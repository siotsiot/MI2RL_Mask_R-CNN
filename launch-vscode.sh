#!/bin/bash
#SBATCH --job-name=SCH_LSS_VSCode
#SBATCH --partition=RTX3090
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x(%j).log

set -euo pipefail

cd /mnt/nas100/forGPU/SCH_project/seungsu0806/PyTorch-Simple-MaskRCNN_train

mkdir -p logs
mkdir -p ${HOME}/.vscode-server
mkdir -p ${HOME}/.vscode-cli
mkdir -p ${HOME}/.vscode-tunnels

IMAGE_NAME="maskrcnn-vscode:glibc235"
IMAGE_TAR="/mnt/nas100/forGPU/SCH_project/seungsu0806/images/maskrcnn-vscode.tar"

if ! docker image inspect ${IMAGE_NAME} >/dev/null 2>&1; then
    docker load < ${IMAGE_TAR}
fi

docker rm -f maskrcnn-vscode-dev 2>/dev/null || true

docker compose up vscode
