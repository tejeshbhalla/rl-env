"""Tools for the rl environment workspace."""
import subprocess
import re
import os
from pathlib import Path
from settings import settings

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
VENV_PYTHON = str(ROOT / ".venv-workspace" / "bin" / "python")
VENV_PIP = str(ROOT / ".venv-workspace" / "bin" / "pip")
UNSLOTH_PKG = str(WORKSPACE / "unsloth-broken-env")

SANDBOX_MODE = settings.sandbox
CONTAINER_NAME = "rl-agent-sandbox"

BLOCKED = [
    r"\.\.",
    r"(^|[\s;|&])cd\s+/",
    r"/home",
    r"/root",
    r"/etc",
    r"/proc",
    r"/sys",
    r"train\.py",
    r"agent\.py",
    r"validator\.py",
    r"settings\.(json|py)",
    r"tools\.py",
    r"real_train",
    r"\.env",
    r"symlink|ln\s+-s",
    r"mount\b",
    r"chroot\b",
    r"curl\b|wget\b",
]
BLOCKED_RE = re.compile("|".join(BLOCKED), re.IGNORECASE)


def _run_docker(cmd):
    result = subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "bash", "-c", f"cd /workspace && {cmd}"],
        capture_output=True, text=True, timeout=300,
    )
    return result


def _run_basic(cmd):
    clean_env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(WORKSPACE),
        "LANG": "C.UTF-8",
        "TERM": "xterm",
    }
    result = subprocess.run(
        ["bash", "-c", cmd],
        cwd=str(WORKSPACE),
        capture_output=True, text=True, timeout=300,
        env=clean_env,
    )
    return result


RUNNERS = {
    "docker": _run_docker,
    "basic": _run_basic,
}


def run(cmd):
    """Run a shell command locked to the workspace directory."""
    if BLOCKED_RE.search(cmd):
        return "Error: command blocked by sandbox policy"

    runner = RUNNERS[SANDBOX_MODE]
    try:
        result = runner(cmd)
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
    },
    {
        "type": "function",
        "function": {
            "name": "final_response",
            "description": "Wrap up the final response and return it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "The final response to the user"
                    }
                },
                "required": ["response"]
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
