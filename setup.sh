#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$ROOT/workspace"
REPO_URL="https://github.com/tejeshbhalla/unsloth-broken-env.git"

echo "=== Setting up workspace ==="
mkdir -p "$WORKSPACE"
if [ -d "$WORKSPACE/unsloth-broken-env" ]; then
    echo "Removing old unsloth-broken-env..."
    rm -rf "$WORKSPACE/unsloth-broken-env"
fi
echo "Cloning broken unsloth..."
git clone "$REPO_URL" "$WORKSPACE/unsloth-broken-env"
echo "=== Workspace ready ==="

# Remove git history so agent can't use git diff/log to cheat
rm -rf "$WORKSPACE/unsloth-broken-env/.git"

# Create pristine backup for resetting between runs
cp -r "$WORKSPACE/unsloth-broken-env" "$WORKSPACE/unsloth-broken-env-pristine"
echo "=== Pristine backup created ==="

echo ""
echo "=== Setting up workspace venv ==="
python3 -m venv "$ROOT/.venv-workspace"
source "$ROOT/.venv-workspace/bin/activate"

pip install --upgrade pip
pip install torch
pip install "transformers==4.56.2"
pip install --no-deps "trl==0.22.2"
pip install "datasets==4.3.0"
pip install accelerate peft bitsandbytes
pip install sentencepiece protobuf
pip install "huggingface_hub>=0.34.0" hf_transfer
pip install triton xformers torchvision
pip install git+https://github.com/unslothai/unsloth-zoo.git
pip install --no-deps -e "$WORKSPACE/unsloth-broken-env/"

deactivate
echo "=== Workspace venv ready at .venv-workspace ==="

echo ""
echo "=== Setting up agent venv ==="
python3 -m venv "$ROOT/.venv-agent"
source "$ROOT/.venv-agent/bin/activate"

echo "=== Setting up docker container ==="
cd "$ROOT"
docker build -t rl-sandbox .

# Stop and remove old container if it exists
docker stop rl-agent-sandbox 2>/dev/null || true
docker rm rl-agent-sandbox 2>/dev/null || true

# Run the container with workspace mounted as a volume
docker run -d \
  --name rl-agent-sandbox \
  --network none \
  -v "$WORKSPACE:/workspace:z" \
  --memory 512m \
  --cpus 1 \
  rl-sandbox
echo "=== Docker container ready ==="

pip install --upgrade pip
pip install litellm

deactivate
echo "=== Agent venv ready at .venv-agent ==="

echo ""
echo "Done! Usage:"
echo "  Agent:  source .venv-agent/bin/activate && python agent.py"
echo "  Train:  .venv-workspace/bin/python train.py"
