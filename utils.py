import os 
import json
#setup creating run dir and cleaning up old runs
def setup_run_dir():
    """Setup creating run dir and cleaning up old runs."""
    run_dir = os.path.join(os.path.dirname(__file__), 'runs')
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    #clean up old runs
    for file in os.listdir(run_dir):
        if file.endswith('.json'):
            os.remove(os.path.join(run_dir, file))
    return run_dir

def create_run_json(initial_hash):
    """Create run json file in runs/run.json."""
    with open('runs/run.json', 'w') as f:
        json.dump({"initial_hash": initial_hash, "final_hash": None}, f)

def update_run_json(final_hash):
    """Update run json file in runs/run.json."""
    with open('runs/run.json', 'r') as f:
        data = json.load(f)
    data["final_hash"] = final_hash
    with open('runs/run.json', 'w') as f:
        json.dump(data, f)