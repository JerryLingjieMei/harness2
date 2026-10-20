# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ClaudeSubstrate: Claude Code CLI (claude) as a solver substrate.

Carries the c1..c8 surface (CLAUDE_SURFACE), the one install mode (claude_root: CLAUDE.md,
AGENTS.md, .claude, scripts staged into the workspace), and trajectory renderer.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

from .. import config
from ..core import runmark as RM
from ..core.tree import SurfaceSpec
from .base import Install, Launcher, RunRequest

CLAUDE_SURFACE = SurfaceSpec(
    name="claude",
    always_on=("CLAUDE.md", "AGENTS.md"),
    always_on_max=24_000,
    always_on_cap_reason="CLAUDE.md is loaded on every turn",
    skill_dir=".claude/skills",
    skill_name_rule="overrides_dir",
    trigger_hint=(
        'Fix it by adding a trigger line naming `{name}` to CLAUDE.md, e.g. "When '
        "<the situation this skill handles>, read .claude/skills/{name}/SKILL.md "
        'and follow it."'
    ),
    autotrigger_file="CLAUDE.md",
    trigger_line="- When {desc}, read .claude/skills/{name}/SKILL.md and follow it.\n",
    subagent_requires_mode=False,
    plugin_dir=None,
    hooks_file=".claude/hooks.json",
    hook_events=frozenset(
        {
            "PreToolUse",
            "PostToolUse",
            "SessionStart",
            "UserPromptSubmit",
            "Stop",
        }
    ),
    compile_skill_scripts=True,
    reject_empty_always_on=False,
    stubs=(
        (
            "scripts/README.md",
            "<!-- c7: deterministic helpers, run as `python3 scripts/<name>.py`. -->\n",
        ),
        (
            "sub_agents/README.md",
            "<!-- c5: delegated agent roles, each spawned in its own context. -->\n",
        ),
        (
            ".claude/skills/README.md",
            "<!-- c6: reusable workflows. Add one as .claude/skills/<name>/SKILL.md -->\n",
        ),
    ),
)

CLAUDE_ROOT_ENTRIES = ("CLAUDE.md", "AGENTS.md", ".claude", "scripts")


class ClaudeSubstrate:
    """claude-cli behind the one Substrate interface."""

    name = "claude"

    def __init__(self, launcher: Launcher | None = None):
        self._launcher = launcher

    def surface(self) -> SurfaceSpec:
        return CLAUDE_SURFACE

    def materialise(self, tree_dir: Path, spec, ctx: Mapping[str, str]) -> Install:
        tree_dir = Path(tree_dir)
        staged = Path(tempfile.mkdtemp(prefix="adapt-claude-root-"))
        for entry in CLAUDE_ROOT_ENTRIES:
            src = tree_dir / entry
            if not src.exists():
                continue
            if src.is_dir():
                shutil.copytree(src, staged / entry)
            else:
                shutil.copy2(src, staged / entry)
        return Install(
            staging_dir=staged,
            fingerprint=RM.fingerprint(tree_dir),
            extras={
                "install_mode": "claude_root",
                **{k: v for k, v in ctx.items() if not k.startswith("file:")},
            },
        )

    def exec_argv(
        self, prompt: str, workspace: Path, model: str | None = None, *, reasoning_effort: str = "high"
    ) -> list[str]:
        cmd = [
            shutil.which("claude") or "claude",
            "--dangerously-skip-permissions",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--effort",
            reasoning_effort,
        ]
        if model:
            cmd.extend(["--model", model])
        return cmd

    def execute(
        self, task_ref: str, install: Install, run_ids: list[str], timeout: int
    ) -> list[Path]:
        if self._launcher is None:
            raise RuntimeError("ClaudeSubstrate was constructed without a launcher body.")
        return self._launcher(
            RunRequest(task_ref=task_ref, run_ids=tuple(run_ids), timeout=int(timeout)), install
        )

    def read_trajectory(
        self, stream_path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 2000
    ) -> str:
        return render_claude_trajectory(
            Path(stream_path), obs_trunc=obs_trunc, max_chars=max_chars, args_trunc=args_trunc
        )

    def preflight(self) -> None:
        if not shutil.which("claude"):
            raise RuntimeError("claude CLI is not installed or not in PATH")


_TRUNC_FMT = "…(+{n} chars — render truncation)"


def _clip(s, n: int) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\x00", "")
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + _TRUNC_FMT.format(n=len(s) - n)


def render_claude_trajectory(
    path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 2000
) -> str:
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"[READ ERROR] {e}"

    lines: list[str] = []
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            res = data.get("result") or data.get("text") or ""
            if res:
                lines.append(f"[ASSISTANT] {_clip(str(res), obs_trunc)}")
            for it in data.get("iterations") or []:
                if isinstance(it, dict):
                    tool_calls = it.get("tool_calls") or []
                    for tc in tool_calls:
                        name = tc.get("name", "unknown")
                        lines.append(f"[ACTION] {name}: {_clip(str(tc.get('args', '')), args_trunc)}")
    except json.JSONDecodeError:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            lines.append(f"[LOG] {_clip(line, obs_trunc)}")

    out = "\n".join(lines).strip()
    return _clip(out, max_chars)
