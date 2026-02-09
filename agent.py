"""Agent for simulating env for solving training bugs ."""
import json
from litellm import completion
from settings import settings
from tools import TOOLS, execute_tool
from validator import create_file_hash, hash_score
from utils import setup_run_dir, create_run_json, update_run_json
# System prompt for the agent
SYSTEM_PROMPT = """\
You are a debugging agent. Your only job is to find and fix bugs in a modified Unsloth package so that a training script runs correctly.

## Setup

- The Unsloth source code you must fix is inside: unsloth-broken-env/
- The training script is train.py. You cannot modify train.py.
- All shell commands via the run tool execute with cwd already set to the workspace. Use relative paths only (e.g. "cat unsloth-broken-env/unsloth/models/llama.py", not absolute paths).

## Tools (you have exactly two)

1. run(cmd): Execute a shell command in the workspace. Use this to read files, search code, and apply edits (e.g. cat, grep, sed, python -c). Path traversal (..) and absolute paths are blocked.
2. run_train(): Reinstalls the unsloth package from unsloth-broken-env/ into the training venv, then runs train.py. Returns stdout, stderr, and exit code. Call this to validate your fixes.
3. final_response(response): Wrap up the final response and return it once you are done debugging and all the fixes are applied.
There are no other tools. Do not attempt to call any tool other than "run" and "run_train".

## Workflow

1. Call run_train() first to see the current error or loss behavior.
2. Use run(cmd) to read source files, search for bugs (grep, cat, etc.), and understand the code.
3. Use run(cmd) with sed or python -c to edit files in unsloth-broken-env/.
4. Call run_train() again to check if your fix worked.
5. Repeat until training completes with stable, decreasing loss.

## Rules

- Only modify files inside unsloth-broken-env/. Do not create new files.
- Do not hardcode loss values, bypass training, fake logs, or override metrics.
- Do not modify the dataset or train.py.
- Fixes must be principled: correct the actual bug (e.g. wrong loss formula, bad normalization, incorrect masking, missing scaling).
- Add a brief inline comment at each fix explaining why it is correct.

## Success Criteria

Training (run_train) must:
- Complete without errors (exit code 0).
- Show training loss that trends downward over steps without diverging.
- Training should be stable and not diverge with no significant loss spikes and exploding gradients.
- Call final_response(response) once you are done debugging and all the fixes are applied.
"""

# Maximum iterations before stopping
MAX_ITERATIONS = 100


def run_agent(task):
    """
    Run the agent to solve a coding task.
    
    Args:
        task: Description of the issue to solve
        
    Returns:
        Final response from the agent
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task}
    ]
    
    for iteration in range(MAX_ITERATIONS):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{MAX_ITERATIONS}")
        print(f"{'='*60}\n")
        
        # Call LLM
        response = completion(
            model=settings.model_info.model_name,
            messages=messages,
            tools=TOOLS,
            api_key=settings.model_info.api_key
        )
        
        message = response.choices[0].message
        messages.append(message.to_dict())
    
        # Print assistant message
        if message.content:
            print(f"Agent: {message.content}\n")
        
        # Execute tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:

                if tool_call.function.name == "final_response":
                    return json.loads(tool_call.function.arguments)["response"]
                else:
                    func_name = tool_call.function.name
                    raw_args = tool_call.function.arguments
                    args = json.loads(raw_args) if raw_args else {}
                    
                    print(f"→ Calling {func_name}({args})")
                    
                    try:
                        result = execute_tool(func_name, **args)
                        print(f"✓ Result: {result[:200]}..." if len(str(result)) > 200 else f"✓ Result: {result}")
                    except Exception as e:
                        result = f"Error: {str(e)}"
                        print(f"✗ {result}")
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
        
        # If no tool calls and no completion, something went wrong
        if not (hasattr(message, 'tool_calls') and message.tool_calls):
            if not message.content or "TASK_COMPLETE" not in message.content:
                print("Agent didn't make tool calls or complete. Continuing...")
                break
    
    print(f"\n⚠ Reached max iterations ({MAX_ITERATIONS})")
    return "Max iterations reached without completion"


if __name__ == "__main__":
    # Example usage
    task = """
    fix the training instability issue in the training pipeline.
    """
    
    initial_hash = create_file_hash()#create initial hash of the bugged files
    run_dir = setup_run_dir()#setup run directory
    create_run_json(initial_hash)#create run json file
    result = run_agent(task)#run the agent
    final_hash = create_file_hash()#create final hash of the bugged files
    update_run_json(final_hash)#update run json file
    hash_score = hash_score(initial_hash, final_hash)#calculate hash score
    print(f"Hash score: {hash_score}")