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
import textwrap
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


def validate_rope()->float:
    """Validate the rope implementation by comparing the output of the rope implementation with the actual pytorch implementation."""
    script = """
    import torch,json
    torch.manual_seed(3407)
    batch,seq_len,num_heads,head_dim = 2,16,4,64
    half = head_dim // 2
    Q = torch.randn(batch,num_heads,seq_len,head_dim,device="cuda",dtype=torch.bfloat16)
    K = torch.randn(batch,num_heads,seq_len,head_dim,device="cuda",dtype=torch.bfloat16)
    freqs = torch.randn(seq_len,half,device='cuda')

    cos_t = freqs.cos().bfloat16()
    sin_t = freqs.sin().bfloat16()
    go_q = torch.randn_like(Q)
    go_k = torch.randn_like(K)
    
    cos_ref = cos_t.unsqueeze(0).unsqueeze(0)
    sin_ref = sin_t.unsqueeze(0).unsqueeze(0)

    Q_ref = Q.clone().requires_grad_(True)
    K_ref = K.clone().requires_grad_(True)
    
    Q1,Q2 = Q_ref[...,:half],Q_ref[...,half:]
    Q_rotated = torch.cat([Q1*cos_ref - Q2*sin_ref,Q1*sin_ref + Q2*cos_ref],dim=-1)
    K1,K2 = K_ref[...,:half],K_ref[...,half:]
    K_rotated = torch.cat([K1*cos_ref - K2*sin_ref,K1*sin_ref + K2*cos_ref],dim=-1)

    loss = (Q_rotated*go_q).sum() + (K_rotated*go_k).sum()
    loss.backward()
    
    from unsloth.kernels.rope_embedding import fast_rope_embedding

    Q_tri = Q.clone().requires_grad_(True)
    K_tri = K.clone().requires_grad_(True)
    Q_rotated_tri,K_rotated_tri = fast_rope_embedding(Q_tri,K_tri,cos_t,sin_t)
    loss_tri = (Q_rotated_tri*go_q).sum() + (K_rotated_tri*go_k).sum()
    loss_tri.backward()
    
    r = {}
    r["fwd_q"] = bool(torch.allclose(Q_rotated_tri, Q_rotated, rtol=1e-2, atol=1e-2))
    r["fwd_k"] = bool(torch.allclose(K_rotated_tri, K_rotated, rtol=1e-2, atol=1e-2))
    r["bwd_q"] = bool(torch.allclose(Q_tri.grad, Q_ref.grad, rtol=1e-2, atol=1e-2))
    r["bwd_k"] = bool(torch.allclose(K_tri.grad, K_ref.grad, rtol=1e-2, atol=1e-2))
    print(json.dumps(r))
    """

    result = subprocess.run(
        [VENV_PYTHON, "-c", textwrap.dedent(script)],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        print(f"Rope validation failed:\n{result.stderr}")
        return 0.0
    try:
        r = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        print(f"Rope validation failed:\n{result.stdout}")
        return 0.0
    score = 0
    if r["fwd_q"]:
        score += 0.15
    if r["fwd_k"]:
        score += 0.15
    if r["bwd_q"]:
        score += 0.35
    if r["bwd_k"]:
        score += 0.35
    return score

def validate_swiglu()->float:
    """Validate the swiglu implementation by comparing the output of the swiglu implementation with the actual pytorch implementation."""
    script = """
    import torch,json
    torch.manual_seed(3407)
    batch,seq_len,hd = 2,16,128

    e = torch.randn(batch,seq_len,hd,device="cuda",dtype=torch.bfloat16)
    g = torch.randn(batch,seq_len,hd,device="cuda",dtype=torch.bfloat16)
    grad_o = torch.randn(batch,seq_len,hd,device="cuda",dtype=torch.bfloat16)

    e_ref = e.clone().requires_grad_(True)
    g_ref = g.clone().requires_grad_(True)
    h_ref = torch.nn.functional.silu(e_ref)*g_ref
    h_ref.backward(grad_o)

    from unsloth.kernels.swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel
    h_tri = swiglu_fg_kernel(e.clone(), g.clone())
    grad_o_tri = grad_o.clone().reshape(-1,hd)
    e_bwd = e.clone().reshape(-1,hd)
    g_bwd = g.clone().reshape(-1,hd)
    swiglu_DWf_DW_dfg_kernel(grad_o_tri, e_bwd, g_bwd)

    r = {}
    r["fwd"] = bool(torch.allclose(h_tri, h_ref.detach(), rtol=1e-2, atol=1e-2))
    r["bwd_dg"] = bool(torch.allclose(
        e_bwd.reshape(batch, seq_len, hd),
        g_ref.grad,
        rtol=1e-2, atol=1e-2
    ))
    r["bwd_de"] = bool(torch.allclose(
        g_bwd.reshape(batch, seq_len, hd),
        e_ref.grad,
        rtol=1e-2, atol=1e-2
    ))
    print(json.dumps(r))
    """

    result = subprocess.run(
        [VENV_PYTHON, "-c", textwrap.dedent(script)],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        print(f"SwiGLU validation crashed: {result.stderr[-300:]}")
        return 0.0
    try:
        r = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        print(f"SwiGLU validation bad output: {result.stdout[:200]}")
        return 0.0
    score = 0.0
    if r.get("fwd"):
        score += 0.2
    if r.get("bwd_dg"):
        score += 0.2
    if r.get("bwd_de"):
        score += 0.6
    return score