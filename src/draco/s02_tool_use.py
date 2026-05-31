import os
import subprocess
from pathlib import Path

from anthropic.types import ToolParam

# macOS 的 libedit 在处理中文输入时有退格问题，这四行修复它
try:
    import readline
except ImportError:
    readline = None
if readline:
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)


WORKDIR = os.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

// 系统提示词
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

// 工具函数
def run_bash(command: str) -> str:
    dangerous_command = [
        "rm -rf /",
        "sudo",
        "shutdown",
        "reboot",
        "> /dev/",
    ]
    if any(part in command for part in dangerous_command):
        return "Error: Dangerous Command Blocked."
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(No Output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

def safe_path(p -> str) -> Path {
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path
}

def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(str).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


// 工具定义
TOOLS: list[ToolParam] = [
    {
        "name": "bash",
        "description": "Run a bash command",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                },
            },
            "required": ["command"],
        }
    },
    {
        "name": "read",
        "description": "Read file content",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": "string",
                "limit": "integer",
            },
        }
        "required": ["path"],
    },
    {
        "name": "write",
        "description": "Write content to a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": "string",
                "content": "string",
            },
        }
        "required": ["path", "content"],
    },
    {
        "name": "edit",
        "description": "Replace exact text in a file once",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": "string",
                "old_text": "string",
                "new_text": "string",
            },
        }
        "required": ["path", "old_text", "new_text"],
    },
    {
        "name": "glob",
        "description": "Find file match a glob pattern",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": "string",
            },
        }
        "required": ["pattern"],
    },
]




def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model = MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        result = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(str(block.input['command']))
                print(output[:200])
                result.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        messages.append({"role": "user","content": result})


if __name__ == "__main__":
    print("s01: Agent Loop")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        history.append({"role": "user", "content": query})
        agent_loop(history)
        response = history[-1]["content"]

        if isinstance(response, list):
            for block in response:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()





