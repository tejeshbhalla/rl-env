"""Tools for the rl environment workspace."""
import subprocess
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
VENV_PYTHON = str(ROOT / ".venv-workspace" / "bin" / "python")
VENV_PIP = str(ROOT / ".venv-workspace" / "bin" / "pip")
UNSLOTH_PKG = str(WORKSPACE / "unsloth-broken-env")


def run(cmd):
    """Run a shell command in the workspace."""
    if ".." in cmd:
        return "Error: Path traversal is not allowed"
    if "cd " in cmd and cmd.split("cd ")[1].strip().startswith("/"):
        return "Error: Cannot navigate outside workspace"
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=300,
            env={**subprocess.os.environ, "HOME": str(WORKSPACE)}
        )
        output = []
        if result.stdout:
            output.append(result.stdout)
        if result.stderr:
            output.append(result.stderr)
        output.append(f"Exit code: {result.returncode}")
        return "\n".join(output)
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 300 seconds"
    except Exception as e:
        return f"Error: {str(e)}"


def run_train():
    """Reinstall workspace unsloth then run train.py and return output."""
    output = []
    try:
        # Reinstall unsloth from workspace so latest fixes are picked up
        install = subprocess.run(
            [VENV_PIP, "install", "--no-deps", "--force-reinstall", "-e", UNSLOTH_PKG],
            capture_output=True, text=True, timeout=120
        )
        if install.returncode != 0:
            output.append(f"Unsloth reinstall failed:\n{install.stderr}")
            return "\n".join(output)
        output.append("Reinstalled unsloth from workspace.")

        # Run training
        result = subprocess.run(
            [VENV_PYTHON, str(ROOT / "train.py")],
            cwd=WORKSPACE,
            capture_output=True, text=True, timeout=600
        )
        if result.stdout:
            output.append(result.stdout)
        if result.stderr:
            output.append(result.stderr)
        output.append(f"Exit code: {result.returncode}")
        return "\n".join(output)
    except subprocess.TimeoutExpired:
        return "Error: Timed out"
    except Exception as e:
        return f"Error: {str(e)}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run",
            "description": "Run a shell command in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Shell command to execute"
                    }
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_train",
            "description": "Run python train.py and return the output with training loss logs.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

TOOL_MAP = {
    "run": run,
    "run_train": run_train
}


def execute_tool(name, **kwargs):
    """Execute a tool by name with given arguments."""
    if name not in TOOL_MAP:
        raise ValueError(f"Unknown tool: {name}")
    return TOOL_MAP[name](**kwargs)
