#!/bin/bash
set -e

echo ">>> 🛑 STOPPING OLD CONTAINERS..."
# Stop and remove any existing vLLM containers
sudo docker stop vllm-server 2>/dev/null || true
sudo docker rm vllm-server 2>/dev/null || true

echo ">>> 🧹 CLEANING DISK SPACE..."
# 1. Remove conflicting runtime if present
sudo apt-get remove -y runc 2>/dev/null || true

# 2. EMERGENCY: Delete the cache from the main disk to free up the 100% usage
# If we don't do this, the symlink below will fail.
sudo rm -rf /root/.cache/huggingface

# 3. Create the folder in the 700GB RAM Disk (/dev/shm)
mkdir -p /dev/shm/huggingface

# 4. Create the "Magic Link"
# This tricks the system into saving the model in RAM instead of the full disk
ln -s /dev/shm/huggingface /root/.cache/huggingface

echo ">>> 🔑 AUTHENTICATING..."
# Set your token
export HF_TOKEN="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"

# Authenticate with Hugging Face inside the container
sudo docker run --rm \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --env HF_TOKEN=$HF_TOKEN \
  vllm/vllm-tpu:latest \
  huggingface-cli login --token $HF_TOKEN 

echo ">>> 🚀 DEPLOYING Gemma 27B (v6e)..."
# Launch vLLM with v6e settings (TP=8)
sudo docker run -d \
  --name vllm-server \
  --privileged \
  --net=host \
  --shm-size=16g \
  --restart always \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-tpu:latest \
  vllm serve google/gemma-3-27b-it \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 16384 \
  --trust-remote-code

echo ">>> ✅ DONE! Monitor download with: sudo docker logs -f vllm-server"