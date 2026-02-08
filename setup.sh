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

pip install --upgrade pip
pip install litellm

deactivate
echo "=== Agent venv ready at .venv-agent ==="

echo ""
echo "Done! Usage:"
echo "  Agent:  source .venv-agent/bin/activate && python agent.py"
echo "  Train:  .venv-workspace/bin/python train.py"
