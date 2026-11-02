"""DeepSeek-based proposer for the Meta-Harness outer loop.

Drop-in replacement for the Claude Code proposer (claude_wrapper.py): an
agentic tool-use loop over the DeepSeek chat completions API. The proposer
keeps the paper's contract — it accesses prior candidates' source code,
scores, and execution traces through the filesystem, then writes new
candidate memory systems into `agents/` plus a `pending_eval.json`.

Ablation switch: set MH_PROPOSER_NO_HISTORY=1 to restrict the proposer to
the current frontier system's source + score only (Markovian proposer, no
search history / execution traces). Enforced at the tool layer, not just
in the prompt.

Env vars:
    DEEPSEEK_API_KEY        (required)
    MH_PROPOSER_MODEL       default "deepseek-chat"
    MH_PROPOSER_NO_HISTORY  "1" enables the history ablation
    MH_PROPOSER_MAX_STEPS   default "40"
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from litellm import completion as litellm_completion

EVOLVE_DIR = Path(__file__).parent
REPO_SUBROOT = EVOLVE_DIR.parent  # reference_examples/
SKILL_PATH = EVOLVE_DIR / ".claude/skills/meta-harness/SKILL.md"

API_BASE = "https://api.deepseek.com/v1"
MAX_FILE_CHARS = 60_000
MAX_CMD_OUTPUT_CHARS = 8_000

# Paths blocked in EVERY mode (huge and useless to the proposer).
_ALWAYS_BLOCKED = ["proposer_sessions", "claude_sessions", "__pycache__"]

# Paths the no-history ablation blocks at the tool layer.
_HISTORY_PATTERNS = [
    "evolution_summary",
    "frontier_val",
    "pending_eval",
    "/reports/",
    "log.jsonl",
    "memory.json",
]

_TMP_ROOT = Path("/tmp").resolve()  # macOS: /tmp -> /private/tmp


def _load_skill_text() -> str:
    text = SKILL_PATH.read_text()
    # Strip YAML frontmatter
    return re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = EVOLVE_DIR / p
    return p.resolve()


def _is_within(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


class ProposerAgent:
    def __init__(self, logs_dir: Path, pending_eval: Path, no_history: bool):
        self.logs_dir = Path(logs_dir)
        self.pending_eval = Path(pending_eval)
        self.no_history = no_history
        self.frontier_file = self.logs_dir / "frontier_val.json"
        self.session_dir = (
            self.logs_dir
            / "proposer_sessions"
            / datetime.now(timezone.utc).strftime("iter_%Y%m%dT%H%M%SZ")
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._trace = self.session_dir / "messages.jsonl"
        self.steps = 0

    # ── logging ─────────────────────────────────────────────
    def _log(self, record: dict):
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with open(self._trace, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # ── tool implementations ────────────────────────────────
    def _blocked_by_ablation(self, p: Path) -> str | None:
        if not self.no_history:
            return None
        s = str(p)
        for pat in _HISTORY_PATTERNS:
            if pat in s:
                return f"BLOCKED (no-history ablation): '{pat}' paths are unavailable."
        # In no-history mode, only the frontier system + shipped baselines are
        # readable under agents/.
        if _is_within(p, EVOLVE_DIR / "agents"):
            allow = {"no_memory.py", "fewshot_memory.py", "fewshot_all.py", "__init__.py"}
            frontier = self._frontier_systems()
            if p.name not in allow and p.stem not in frontier:
                return (
                    "BLOCKED (no-history ablation): only the current frontier "
                    f"system {sorted(frontier) or '(none yet)'} and shipped baselines "
                    "are readable."
                )
        return None

    def _frontier_systems(self) -> set[str]:
        try:
            data = json.loads(self.frontier_file.read_text())
        except Exception:
            return set()
        systems = set()
        for entry in data.get("_pareto", []):
            if isinstance(entry, dict) and entry.get("system"):
                systems.add(entry["system"])
        for key, value in data.items():
            if isinstance(value, dict) and value.get("best_system"):
                systems.add(value["best_system"])
        return systems

    def tool_read_file(self, path: str) -> str:
        p = _resolve(path)
        if any(b in str(p) for b in _ALWAYS_BLOCKED):
            return "BLOCKED: proposer session traces are not useful input."
        blocked = self._blocked_by_ablation(p)
        if blocked:
            return blocked
        if not (_is_within(p, REPO_SUBROOT) or _is_within(p, _TMP_ROOT)):
            return "ERROR: path outside allowed roots."
        if not p.exists() or not p.is_file():
            return f"ERROR: no such file: {p}"
        text = p.read_text(errors="replace")
        if len(text) > MAX_FILE_CHARS:
            text = text[:MAX_FILE_CHARS] + "\n... [TRUNCATED]"
        return text

    def tool_list_files(self, pattern: str = "**/*.py") -> str:
        if self.no_history and ("log" in pattern or "report" in pattern):
            return "BLOCKED (no-history ablation)."
        matches = sorted(
            str(p.relative_to(REPO_SUBROOT))
            for p in REPO_SUBROOT.glob(pattern)
            if p.is_file() and ".venv" not in p.parts
        )
        matches = [
            m
            for m in matches
            if not any(b in m for b in _ALWAYS_BLOCKED)
        ]
        if self.no_history:
            matches = [
                m
                for m in matches
                if not any(pat in m for pat in _HISTORY_PATTERNS)
            ]
        return "\n".join(matches[:300]) or "(no matches)"

    def tool_grep(self, pattern: str, path: str = ".") -> str:
        p = _resolve(path)
        blocked = self._blocked_by_ablation(p)
        if blocked:
            return blocked
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"ERROR: bad regex: {e}"
        hits = []
        files = [p] if p.is_file() else sorted(p.rglob("*"))
        for f in files:
            if not f.is_file() or any(b in str(f) for b in _ALWAYS_BLOCKED):
                continue
            if ".venv" in f.parts or f.suffix not in (
                ".py", ".md", ".json", ".jsonl", ".yaml", ".txt",
            ):
                continue
            if self.no_history and self._blocked_by_ablation(f):
                continue
            try:
                for i, line in enumerate(f.read_text(errors="replace").splitlines()):
                    if rx.search(line):
                        hits.append(f"{f.relative_to(REPO_SUBROOT)}:{i + 1}: {line[:200]}")
            except OSError:
                continue
            if len(hits) >= 50:
                break
        return "\n".join(hits[:50]) or "(no matches)"

    def tool_write_file(self, path: str, content: str) -> str:
        p = _resolve(path)
        ok = (
            _is_within(p, EVOLVE_DIR / "agents")
            or p == self.pending_eval.resolve()
            or _is_within(p, self.logs_dir.resolve() / "reports")
            or _is_within(p, _TMP_ROOT)
        )
        if not ok:
            return "ERROR: writes allowed only to agents/, pending_eval.json, the run reports dir, or /tmp."
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} chars to {p}"

    def tool_run_command(self, command: str) -> str:
        cmd = command.strip()
        if not re.match(r"^(uv run python|PYTHONPATH=\S* uv run python)", cmd):
            return "ERROR: only 'uv run python ...' commands are allowed."
        env = dict(os.environ, PYTHONPATH=str(REPO_SUBROOT))
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=str(EVOLVE_DIR), env=env,
                capture_output=True, text=True, timeout=180,
            )
            out = (result.stdout or "") + (result.stderr or "")
            if len(out) > MAX_CMD_OUTPUT_CHARS:
                out = out[:MAX_CMD_OUTPUT_CHARS] + "\n... [TRUNCATED]"
            return f"exit={result.returncode}\n{out}"
        except subprocess.TimeoutExpired:
            return "ERROR: command timed out after 180s."

    # ── main loop ───────────────────────────────────────────
    def run(self, task_prompt: str, iteration: int) -> bool:
        model = os.environ.get("MH_PROPOSER_MODEL", "deepseek-chat")
        max_steps = int(os.environ.get("MH_PROPOSER_MAX_STEPS", "60"))
        system = self._system_prompt()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task_prompt},
        ]
        self._log({"type": "system", "content": system})
        self._log({"type": "user", "content": task_prompt})

        tools = self._tool_schemas()
        t0 = time.time()
        total_in = total_out = 0

        for step in range(max_steps):
            self.steps = step + 1
            response = litellm_completion(
                model=f"openai/{model}",
                base_url=API_BASE,
                api_key=os.environ["DEEPSEEK_API_KEY"],
                messages=messages,
                tools=tools,
                temperature=1.0,
                max_tokens=8192,
                timeout=600.0,
            )
            usage = getattr(response, "usage", None)
            total_in += getattr(usage, "prompt_tokens", 0) or 0
            total_out += getattr(usage, "completion_tokens", 0) or 0

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            self._log({
                "type": "assistant",
                "content": getattr(msg, "content", None),
                "tool_calls": [
                    {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in (msg.tool_calls or [])
                ],
            })

            if not msg.tool_calls:
                # Nudge the model to keep acting until candidates exist.
                if self.pending_eval.exists():
                    break
                messages.append({
                    "role": "user",
                    "content": (
                        "Continue working. You have not written pending_eval.json "
                        "yet. Use the tools to read state, write candidate files, "
                        "validate them, and finally write pending_eval.json."
                    ),
                })
                continue

            for tc in msg.tool_calls:
                result = self._dispatch(tc.function.name, tc.function.arguments)
                self._log({
                    "type": "tool_result",
                    "name": tc.function.name,
                    "result": result[:2000],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            if self.pending_eval.exists():
                # Candidates registered: allow a couple of wrap-up steps at most.
                steps_after_done = getattr(self, "_steps_after_done", 0) + 1
                self._steps_after_done = steps_after_done
                if steps_after_done >= 3:
                    break

            remaining = max_steps - step - 1
            if not self.pending_eval.exists() and remaining == 14:
                messages.append({
                    "role": "user",
                    "content": (
                        f"STEP BUDGET WARNING: only {remaining} tool steps remain. "
                        "STOP all analysis and prototyping NOW. Within your next "
                        "steps you MUST: (1) write your 3 candidate files to "
                        "agents/<name>.py (keep each mechanism simple — a clean, "
                        "correct implementation beats an elaborate one), "
                        "(2) validate each with the import check, "
                        "(3) write pending_eval.json. Do not start anything new."
                    ),
                })

        wall = time.time() - t0
        self._log({
            "type": "session_end",
            "steps": self.steps,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "wall_seconds": round(wall, 1),
            "pending_eval_written": self.pending_eval.exists(),
        })
        print(
            f"  deepseek proposer: {self.steps} steps, "
            f"{total_in}in/{total_out}out tokens, {wall:.0f}s, "
            f"trace: {self._trace}"
        )
        return self.pending_eval.exists()

    def _dispatch(self, name: str, arguments: str) -> str:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            return "ERROR: tool arguments were not valid JSON."
        try:
            if name == "read_file":
                return self.tool_read_file(args["path"])
            if name == "list_files":
                return self.tool_list_files(args.get("pattern", "**/*.py"))
            if name == "grep":
                return self.tool_grep(args["pattern"], args.get("path", "."))
            if name == "write_file":
                return self.tool_write_file(args["path"], args["content"])
            if name == "run_command":
                return self.tool_run_command(args["command"])
            if name == "finish":
                return "OK. Session over."
            return f"ERROR: unknown tool '{name}'."
        except Exception as e:  # keep the loop alive on tool errors
            return f"ERROR: {type(e).__name__}: {e}"

    def _tool_schemas(self) -> list[dict]:
        def fn(name, desc, props, required):
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    },
                },
            }

        return [
            fn("read_file", "Read a file (path relative to the text_classification directory or absolute).",
               {"path": {"type": "string"}}, ["path"]),
            fn("list_files", "List files matching a glob pattern under reference_examples/ (e.g. 'text_classification/agents/*.py').",
               {"pattern": {"type": "string"}}, []),
            fn("grep", "Search file contents with a regex. 'path' may be a file or directory.",
               {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
            fn("write_file", "Write a file. Allowed targets: agents/*.py candidate files, the pending_eval.json path from the task prompt, the run reports dir, or /tmp.",
               {"path": {"type": "string"}, "content": {"type": "string"}},
               ["path", "content"]),
            fn("run_command", "Run a shell command restricted to 'uv run python ...' (for import validation and /tmp prototyping).",
               {"command": {"type": "string"}}, ["command"]),
            fn("finish", "Signal that all candidates are written and validated.", {}, []),
        ]

    def _system_prompt(self) -> str:
        skill = _load_skill_text()
        header = (
            "You are the proposer agent in a Meta-Harness evolution loop for "
            "memory systems for online text classification. You work through "
            "tools: read_file / list_files / grep / write_file / run_command. "
            "Do all analysis yourself, step by step, then write exactly 3 new "
            "candidate memory systems.\n\n"
            "TIME BUDGET: you have at most 60 tool steps. Spend at most 20 on "
            "reading + prototyping, then move on to writing candidates. "
            "Prototype scripts go to /tmp (writes there are allowed). A clean, "
            "correct, simple mechanism beats an elaborate one.\n\n"
            f"Your working directory is: {EVOLVE_DIR}\n"
            f"The reference_examples root is: {REPO_SUBROOT}\n"
            f"pending_eval.json MUST be written to: {self.pending_eval}\n"
            f"Post-eval reports (if any) go to: {self.logs_dir / 'reports'}/\n\n"
            "Candidate files go to agents/<snake_case_name>.py and must define "
            "a MemorySystem subclass. Validate each candidate with:\n"
            "  uv run python -c \"from text_classification.agents.<name> import *; print('OK')\"\n"
            "(PYTHONPATH is set for you.)\n\n"
            "The underlying LLM for memory systems is a non-reasoning chat "
            "model (DeepSeek) with a 4096-token output cap. Prompts must ask "
            "for strict JSON with a 'final_answer' field; use "
            "extract_json_field(response, 'final_answer') for parsing.\n\n"
        )
        if self.no_history:
            header += (
                "## ABLATION: NO SEARCH HISTORY\n"
                "You do NOT have access to evolution_summary.jsonl, frontier "
                "files, past reports, execution traces, or other past "
                "candidates' code. You may read ONLY the shipped baselines "
                "(agents/no_memory.py, agents/fewshot_memory.py, "
                "agents/fewshot_all.py), the current frontier system's file "
                "if one exists, config.yaml, and the interface files "
                "memory_system.py / llm.py. Base your 3 candidates on the "
                "single best current system plus your own priors.\n\n"
            )
        return header + "## Skill prior (follow its workflow and constraints)\n\n" + skill


def run(task_prompt: str, iteration: int, logs_dir, pending_eval) -> bool:
    """Entry point used by meta_harness.py. Returns True if pending_eval.json exists."""
    no_history = os.environ.get("MH_PROPOSER_NO_HISTORY", "0") == "1"
    agent = ProposerAgent(
        logs_dir=Path(logs_dir),
        pending_eval=Path(pending_eval),
        no_history=no_history,
    )
    return agent.run(task_prompt=task_prompt, iteration=iteration)


if __name__ == "__main__":
    # Manual smoke run: python deepseek_proposer.py "<task prompt>" <logs_dir> <pending_eval_path>
    import sys

    run(
        task_prompt=sys.argv[1],
        iteration=0,
        logs_dir=Path(sys.argv[2]),
        pending_eval=Path(sys.argv[3]),
    )
