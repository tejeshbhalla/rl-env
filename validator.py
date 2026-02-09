"""
Validator for the script fixes 

Validates in phases if bugs are fixed.
 1. Validates hash of the bugged file before and after the fix.
 2. Validates kernel bugs with matching buggy kernels with actual pytorch implementation.
 3. Validates if the training script runs correctly and steady loss decrease and no gradient norm explosion 

Things not added:
 1. Validating if the fixes are correct and not just a random change. Hard to validate exactly even with matching lines since agent can add a proxy file to fix which is totally fine , thought of embedding similarity but wont work good for code evals 
 2. Kernel bugs are validated by matching tolerance with actual pytorch implementation, if fixed passes that test then it is good.
 3. Training loss and gradient norm explosion are validated by checking the training script output, validates for training loss and gradient norm explosion.

"""

import hashlib
import os

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