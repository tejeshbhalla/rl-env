### Requirements 
- GPU VM Cuda
- Python 

### 1. Clone the repo

git clone <your-repo-url> rl-env
cd rl-env### 2. Add your API key

Create `settings.json` (or edit the existing one):

{
    "model_info": {
        "api_key": "YOUR_API_KEY_HERE",
        "model_name": "openrouter/anthropic/claude-sonnet-4.5"
    },
    "sandbox": "basic"
}### 3. Run setup

chmod +x setup.sh
./setup.sh


This will:
- Clone the bugged Unsloth repo into `workspace/unsloth-broken-env/`
- Set up `.venv-workspace` (torch, transformers, triton, etc.)
- Set up `.venv-agent` (litellm, numpy, scipy)
- Auto-detect Docker or fall back to basic subprocess sandbox

### 4. Run the agent

source .venv-agent/bin/activate
python agent.py
