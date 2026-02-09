"""
Validator for the script fixes 

Validates in phases if bugs are fixed.
 1. Validates hash of the bugged file before and after the fix.
 2. Validates kernel bugs with matching buggy kernels with actual pytorch implementation.
 3. Validates if the training script runs correctly and steady loss decrease and no gradient norm explosion 

Things not added:
 1. Validating if the fixes are correct and not just a random change. Hard to validate exactly even with matching lines since agent can add a proxy file to fix which is totally fine , thought of embedding similarity but wont work good for code evals 
 2. Kernel bugs are validated by matching tolerance with actual pytorch implementation, if fixed passes that test then it is good.
 3. Training loss and gradient norm explosion are validated by checking the training script output, validates for training loss and gradient norm explosion comparing to stable unsloth train run output 
     (we do correlation for  loss curve to measure the moment of difference in both curves , for the gradient norm compare avg gradient norm of both runs, also compare final loss of both runs and finally make a continuous score for validating agent performance)

"""

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
VENV_PYTHON = str(ROOT / ".venv-workspace" / "bin" / "python")
VENV_PIP = str(ROOT / ".venv-workspace" / "bin" / "pip")
UNSLOTH_PKG = str(WORKSPACE / "unsloth-broken-env")
TRAIN_STATS = str(WORKSPACE / "runs" / "train_stats.json")
REAL_TRAIN_FILE = str(ROOT / "real_train.json")


BUGGED_FILES = [
    'unsloth/kernels/rope_embedding.py', #rope ctx twisted incorrectly causes pos embeddings to be incorrect cos,sin flipped
    'unsloth/kernels/swiglu.py', #incorrect backward math ,
    'unsloth/kernels/rms_layernorm.py', #incorrect precision in forward pass it uses the precision of the weight which can be bf16 or fp16 but in backward uses fp32 (more of a bait bug good to solve but makes llm look at bait bugs) ,
    "unsloth/models/llama.py", #incorrect rope precision , residual skip connection bugged,noise injection in residual,some other bait bugs in llama.py

]


def create_file_hash()->dict[str,str]:
    """Create initial hash mapping for all bugged files before fixes are applied."""
    hash_map = {}
    workspace_path = os.path.join(os.path.dirname(__file__), 'workspace', 'unsloth-broken-env')
    
    for bugged_file in BUGGED_FILES:
        file_path = os.path.join(workspace_path, bugged_file)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                hash_map[bugged_file] = file_hash
        else:
            hash_map[bugged_file] = None
    
    return hash_map


def hash_score(initial_hash, final_hash)->int:
    """Count the number of hash changes in the bugged files."""
    return sum(1 for key in initial_hash if initial_hash[key] != final_hash[key])/len(BUGGED_FILES)


def validate_train():
    """Run train, compare stats to golden reference, return score 0.0-1.0"""

    # reinstall unsloth from workspace so agent's changes are picked up
    install = subprocess.run(
        [VENV_PIP, "install", "--no-deps", "--force-reinstall", "-e", UNSLOTH_PKG],
        capture_output=True, text=True, timeout=120
    )
    if install.returncode != 0:
        print(f"Install failed:\n{install.stderr}")
        return 0.0

    # run training
    result = subprocess.run(
        [VENV_PYTHON, str(ROOT / "train.py")],
        cwd=str(WORKSPACE),
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        print(f"Train crashed:\n{result.stderr[-500:]}")
        return 0.0

    # load agent stats and golden stats
    if not os.path.exists(TRAIN_STATS):
        print("train_stats.json not found")
        return 0.0

    with open(TRAIN_STATS) as f:
        agent_stats = json.load(f)
    with open(REAL_TRAIN_FILE) as f:
        golden_stats = json.load(f)

    agent_losses = [e["loss"] for e in agent_stats if "loss" in e]
    agent_norms = [e["grad_norm"] for e in agent_stats if "grad_norm" in e]
    golden_losses = [e["loss"] for e in golden_stats if "loss" in e]
    golden_norms = [e["grad_norm"] for e in golden_stats if "grad_norm" in e]

    #have to check length of agent_losses and golden_losses
    if len(agent_losses) != len(golden_losses):
        return 0.0
    

    # score A: final loss ratio (weight 0.5)
    # log ratio: 1.0 at exact match, drops toward 0 as it diverges
    ratio = agent_losses[-1] / golden_losses[-1]
    if ratio > 1.2:
        return 0.0
    if ratio<0.5:
        return 0.0
    score_a = max(0.0, 1.0 - abs(math.log(ratio)))

    corr = np.corrcoef(agent_losses, golden_losses)[0, 1]
    if corr < 0.5:
        return 0.0
    score_b = max(0.0, corr)

    avg_agent_norm = sum(agent_norms) / len(agent_norms)
    avg_golden_norm = sum(golden_norms) / len(golden_norms)
    norm_ratio = avg_agent_norm / avg_golden_norm
    score_c = max(0.0, 1.0 - abs(math.log(norm_ratio)))

    total = 0.5 * score_a + 0.3 * score_b + 0.2 * score_c

    return total