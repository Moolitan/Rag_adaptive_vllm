"""
Colored Logging Helpers
"""
import os
import json
import time
from typing import List, Dict

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# =============================
# Colored Logging Helpers
# =============================
class C:
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    PURPLE = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    END = "\033[0m"

# =============================
# File logging (per-run)
# =============================
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log")
ensure_dir(LOG_DIR)
_LOG_FILE = os.path.join(LOG_DIR, time.strftime("%Y%m%d_%H%M%S") + ".log")

def _append_log(line: str):
    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Best-effort file logging; avoid breaking stdout logging.
        pass

def log(tag: str, msg: str, color: str = C.BLUE):
    line = f"[{tag}] {msg}"
    colored = f"{color}{line}{C.END}"
    print(colored)
    _append_log(line)

def safe_preview(text: str, n: int = 160) -> str:
    if not text:
        return ""
    t = text.replace("\n", " ").strip()
    return t[:n] + ("..." if len(t) > n else "")

def now_ts() -> int:
    return int(time.time())

def write_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_jsonl_tail(path: str, k: int) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()[-k:]
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out
