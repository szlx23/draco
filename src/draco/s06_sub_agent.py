import os
import subprocess
from pathlib import Path

from anthropic.types import ToolUseBlock

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

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# 系统提示词
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


# 工具函数
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


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str):
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str):
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text))
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str):
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(No Match)"
    except Exception as e:
        return f"Error: {e}"


TODO_LIST = []
TODO_ICON = {
    "pending": "⭕",
    "working": "🔄",
    "done": "✅",
}


def run_todo_write(todos: list):
    global TODO_LIST
    TODO_LIST = []

    for todo in todos:
        TODO_LIST.append(TODO_ICON[todo["status"]] + " " + todo["content"])


def spawn_subagent(description: str) -> str:
    messages: list = [{
        "role": "user",
        "content": description,
    }, ]

    sub_tools: list = [
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
                    "path": {
                        "type": "string",
                    },
                    "limit": {
                        "type": "integer",
                    }
                },
            },
            "required": ["path"],
        },
        {
            "name": "write",
            "description": "Write content to a file",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                    },
                    "content": {
                        "type": "string",
                    },
                },
            },
            "required": ["path", "content"],
        },
        {
            "name": "edit",
            "description": "Replace exact text in a file once",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                    },
                    "old_text": {
                        "type": "string",
                    },
                    "new_text": {
                        "type": "string",
                    },
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
        {
            "name": "glob",
            "description": "Find file match a glob pattern",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                    },
                },
            },
            "required": ["pattern"],
        },
    ]

    iter_round = 0

    while iter_round < 30:

        iter_round += 1

        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=sub_tools, max_tokens=8000
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            trigger_hook("Stop", messages)
            break

        result = []
        for block in response.content:
            if isinstance(block, ToolUseBlock):
                print(f"\033[33m[sub] $ {block.name}\033[0m")
                blocked = trigger_hook("PreToolUse", block)
                if blocked:
                    result.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": str(blocked)})
                    continue
                output = TOOL_HANDLERS.get(block.name)(**block.input)
                if block.name == "todo_write":
                    not_todo_round = 0
                trigger_hook("PostToolUse", block, output)
                print(f"[sub]{(output or "")[:200]}")
                result.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        messages.append({"role": "user", "content": result})

    return extract_text(messages[-1]["content"])

def extract_text(content) -> str:
    if not isinstance(content, list):
        return str(content)
    return "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")

# 工具定义
TOOLS: list = [
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
                "path": {
                    "type": "string",
                },
                "limit": {
                    "type": "integer",
                }
            },
        },
        "required": ["path"],
    },
    {
        "name": "write",
        "description": "Write content to a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                },
                "content": {
                    "type": "string",
                },
            },
        },
        "required": ["path", "content"],
    },
    {
        "name": "edit",
        "description": "Replace exact text in a file once",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                },
                "old_text": {
                    "type": "string",
                },
                "new_text": {
                    "type": "string",
                },
            },
        },
        "required": ["path", "old_text", "new_text"],
    },
    {
        "name": "glob",
        "description": "Find file match a glob pattern",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                },
            },
        },
        "required": ["pattern"],
    },
    {
        "name": "todo_write",
        "description": "Write todo list to a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["pending", "working", "done"],
                            },
                            "content": {
                                "type": "string",
                            }
                        }
                    }
                }
            }
        }
    },
    {
        "name": "task",
        "description": "Spawn a subagent to do a task",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                }
            }
        }
    },
]

# 工具分发
TOOL_HANDLERS = {
    "bash": run_bash,
    "read": run_read,
    "write": run_write,
    "edit": run_edit,
    "glob": run_glob,
    "todo_write": run_todo_write,
    "task": spawn_subagent,
}

# hook
HOOKS = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": []
}


def register_hook(event: str, callback):
    HOOKS[event].append(callback)


def trigger_hook(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


def prompt_injection_hook(query: str):
    query.join(f" Current working director is {WORKDIR}")
    return None


register_hook("UserPromptSubmit", prompt_injection_hook)

# 工具权限拦截
DENY_LIST: list[str] = ["sudo", "rm -rf", "reboot", "shutdown", "mkfs", "> /dev/sda", "dd if"]


def check_deny_list(command: str) -> str | None:
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked: {command} is in the deny list"
    return None


from typing import Callable, TypedDict


class PermissionRule(TypedDict):
    tools: list[str]
    rule: Callable[[dict], bool]
    message: str


PERMISSION_RULES: list[PermissionRule] = [
    {
        "tools": ["write", "edit"],
        "rule": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
        "message": "Writing outside workspace",
    },
    {
        "tools": ["bash"],
        "rule": lambda args: any(pattern in args.get("command", "") for pattern in ["rm", "> /etc/", "chmod 777"]),
        "message": "Potentially destructive command",
    }
]


def check_permission(block: ToolUseBlock):
    if block.name == "bash":
        if any(pattern in block.input.get("command", "") for pattern in DENY_LIST):
            return "Permission denied by deny list"
    if block.name in ("write", "edit"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            choice = input("    Allow? [y/N] ").strip().lower()
            if choice not in ["y", "yes"]:
                return "Permission denied by user"
    return None


register_hook("PreToolUse", check_permission)


def log_hook(block):
    print(f"[HOOK] {block.name}(...)")


register_hook("PreToolUse", log_hook)


def large_output_hook(block, output):
    if len(str(output)) > 100000:
        print(f"[HOOK] Large output from {block.name}")


register_hook("PostToolUse", large_output_hook)


def summary_hook(messages: list) -> str | None:
    tool_count = sum(
        1
        for m in messages
        for b in (
            m.get("content")
            if isinstance(m.get("content"), list)
            else []
        )
        if isinstance(b, dict)
        and b.get("type") == "tool_result"
    )
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None


register_hook("Stop", summary_hook)


def agent_loop(messages: list):
    not_todo_round = 0

    while True:

        if not_todo_round > 3 and messages:
            messages.append({
                "role": "user",
                "content": "<Reminder>Update your todo list.<Reminder>",
            })

        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            trigger_hook("Stop", messages)
            return

        result = []
        for block in response.content:
            if isinstance(block, ToolUseBlock):
                print(f"\033[33m$ {block.name}\033[0m")
                blocked = trigger_hook("PreToolUse", block)
                if blocked:
                    result.append({"type": "tool_result", "tool_use_id": block.id,
                                   "content": str(blocked)})
                    continue
                output = TOOL_HANDLERS.get(block.name)(**block.input)
                if block.name == "todo_write":
                    not_todo_round = 0
                trigger_hook("PostToolUse", block, output)
                print((output or "")[:200])
                result.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        messages.append({"role": "user", "content": result})


if __name__ == "__main__":
    print("s01: Agent Loop")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
            trigger_hook("UserPromptSubmit", query)
        except (EOFError, KeyboardInterrupt):
            break

        if query == "q":
            break

        history.append({"role": "user", "content": query})
        agent_loop(history)
        response = history[-1]["content"]

        if isinstance(response, list):
            for block in response:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()
