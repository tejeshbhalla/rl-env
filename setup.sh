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

#install requirements
pip install -r "$ROOT/requirements.txt"

# ── Sandbox setup (auto-detect) ────────────────────────────
echo "=== Setting up sandbox ==="
cd "$ROOT"

if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    echo "Docker detected — using Docker sandbox"
    docker build -t rl-sandbox .
    docker stop rl-agent-sandbox 2>/dev/null || true
    docker rm rl-agent-sandbox 2>/dev/null || true
    docker run -d \
      --name rl-agent-sandbox \
      --network none \
      -v "$WORKSPACE:/workspace:z" \
      --memory 512m \
      --cpus 1 \
      rl-sandbox
    SANDBOX_MODE="docker"

elif command -v bwrap &>/dev/null; then
    echo "bubblewrap detected — using bwrap sandbox"
    SANDBOX_MODE="bwrap"

else
    echo "ERROR: Neither Docker nor bubblewrap found."
    echo "Install one of: docker, bubblewrap (apt install bubblewrap)"
    exit 1
fi

# Write sandbox mode into settings.json
python3 -c "
import json
with open('$ROOT/settings.json', 'r') as f:
    cfg = json.load(f)
cfg['sandbox'] = '$SANDBOX_MODE'
with open('$ROOT/settings.json', 'w') as f:
    json.dump(cfg, f, indent=4)
"
echo "=== Sandbox mode: $SANDBOX_MODE ==="

pip install --upgrade pip
pip install litellm

deactivate
echo "=== Agent venv ready at .venv-agent ==="

echo ""
echo "Done! Usage:"
echo "  Agent:  source .venv-agent/bin/activate && python agent.py"
echo "  Train:  .venv-workspace/bin/python train.py"
