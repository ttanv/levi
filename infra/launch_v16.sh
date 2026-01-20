#!/bin/bash

# ==========================================================
# CONFIGURATION - Multi-Host TPU v4-16 vLLM Deployment
# ==========================================================
TPU_NAME="algoforge-glm"
ZONE="us-central2-b"
MODEL="zai-org/GLM-4.5-Air"
HF_TOKEN="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"
# ==========================================================

echo ">>> Deploying vLLM to TPU v4-16 cluster: $TPU_NAME"

# ==========================================================
# Get coordinator IP (worker 0's IP)
# ==========================================================
echo ">>> Step 1: Getting coordinator IP..."
COORDINATOR_IP=$(gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 --command "
curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-network-endpoints' -H 'Metadata-Flavor: Google' | sed 's/unknown:unknown://g' | cut -d',' -f1
" 2>/dev/null | tail -1)
echo ">>> Coordinator IP: $COORDINATOR_IP"

# ==========================================================
# Stop existing containers on both workers
# ==========================================================
echo ">>> Step 2: Cleaning up..."
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command "
sudo docker stop vllm-server 2>/dev/null || true
sudo docker rm vllm-server 2>/dev/null || true
sudo rm -f /tmp/libtpu_lockfile 2>/dev/null || true
echo 'Cleanup done'
" 2>&1 | grep -v "^SSH:" | grep -v "^Using ssh" || true

# ==========================================================
# Launch vLLM on ALL workers with JAX distributed config
# ==========================================================
echo ">>> Step 3: Launching vLLM containers on all workers..."

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command "
# Get worker info from metadata
WORKER_ID=\$(curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env' -H 'Metadata-Flavor: Google' | grep -oP \"WORKER_ID: '\\K[0-9]+\")
WORKER_ENDPOINTS=\$(curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-network-endpoints' -H 'Metadata-Flavor: Google')
WORKER_IPS=\$(echo \"\$WORKER_ENDPOINTS\" | sed 's/unknown:unknown://g')
COORDINATOR_IP=\$(echo \"\$WORKER_IPS\" | cut -d',' -f1)
NUM_WORKERS=\$(echo \"\$WORKER_IPS\" | tr ',' '\n' | wc -l)

echo \"Worker \$WORKER_ID starting with coordinator \$COORDINATOR_IP (total workers: \$NUM_WORKERS)\"

# Run vLLM container with JAX distributed environment
sudo docker run -d \\
  --name vllm-server \\
  --privileged \\
  --net=host \\
  --shm-size=16g \\
  -v /tmp:/tmp \\
  -e HF_TOKEN=$HF_TOKEN \\
  -e JAX_COORDINATOR_ADDRESS=\$COORDINATOR_IP:8476 \\
  -e JAX_NUM_PROCESSES=\$NUM_WORKERS \\
  -e JAX_PROCESS_ID=\$WORKER_ID \\
  -e TPU_WORKER_ID=\$WORKER_ID \\
  -e TPU_WORKER_HOSTNAMES=\$WORKER_IPS \\
  vllm/vllm-tpu:latest \\
  vllm serve $MODEL \\
  --host 0.0.0.0 \\
  --port 8000 \\
  --tensor-parallel-size 4 \\
  --pipeline-parallel-size 2 \\
  --max-model-len 16384 \\
  --trust-remote-code

echo \"Worker \$WORKER_ID: Container started\"
"

echo ">>> Step 4: Waiting for containers to initialize..."
sleep 15

# ==========================================================
# Check container status on both workers
# ==========================================================
echo ">>> Step 5: Checking container status..."
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command "
WORKER_ID=\$(curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env' -H 'Metadata-Flavor: Google' | grep -oP \"WORKER_ID: '\\K[0-9]+\")
echo \"=== Worker \$WORKER_ID ===\"
sudo docker ps -a --filter name=vllm-server --format 'Status: {{.Status}}'
echo 'Last 10 lines of log:'
sudo docker logs vllm-server 2>&1 | tail -10
echo ''
"

echo "=========================================================="
echo ">>> LAUNCHED! Both workers should be coordinating."
echo ""
echo ">>> Monitor worker 0 logs:"
echo "gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 --command 'sudo docker logs -f vllm-server'"
echo ""
echo ">>> Monitor worker 1 logs:"
echo "gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=1 --command 'sudo docker logs -f vllm-server'"
echo "=========================================================="
