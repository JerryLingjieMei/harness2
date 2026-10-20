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

"""AgySubstrate: Antigravity CLI (agy) as a solver substrate.

Carries the c1..c8 surface (AGY_SURFACE), the one install mode (agy_root: AGENTS.md,
.agents, scripts, and .gemini staged into the workspace), the `agy` flag contract,
and the transcript trajectory renderer.
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
from ..core import tree as HT
from ..core.tree import SurfaceSpec
from .base import Install, Launcher, RunRequest

AGY_SURFACE = SurfaceSpec(
    name="agy",
    always_on=("AGENTS.md",),
    always_on_max=24_000,
    always_on_cap_reason="AGENTS.md is injected into turns and competes with output token budget",
    skill_dir=".agents/skills",
    skill_name_rule="overrides_dir",
    trigger_hint=(
        'Fix it by adding a trigger line naming `{name}` to AGENTS.md, e.g. "When '
        "<the situation this skill handles>, read .agents/skills/{name}/SKILL.md "
        'and follow it."'
    ),
    autotrigger_file="AGENTS.md",
    trigger_line="- When {desc}, read .agents/skills/{name}/SKILL.md and follow it.\n",
    subagent_requires_mode=False,
    plugin_dir=None,
    hooks_file=".gemini/hooks.json",
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
            ".agents/skills/README.md",
            "<!-- c6: reusable workflows. Add one as .agents/skills/<name>/SKILL.md -->\n",
        ),
    ),
)

AGY_ROOT_ENTRIES = ("AGENTS.md", "GEMINI.md", ".agents", "scripts", ".gemini")


class AgySubstrate:
    """agy-cli behind the one Substrate interface."""

    name = "agy"

    def __init__(self, launcher: Launcher | None = None):
        self._launcher = launcher

    def surface(self) -> SurfaceSpec:
        return AGY_SURFACE

    def materialise(self, tree_dir: Path, spec, ctx: Mapping[str, str]) -> Install:
        tree_dir = Path(tree_dir)
        staged = Path(tempfile.mkdtemp(prefix="adapt-agy-root-"))
        for entry in AGY_ROOT_ENTRIES:
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
                "install_mode": "agy_root",
                **{k: v for k, v in ctx.items() if not k.startswith("file:")},
            },
        )

    def exec_argv(
        self, prompt: str, workspace: Path, model: str | None = None, *, reasoning_effort: str = "high"
    ) -> list[str]:
        cmd = [
            shutil.which("agy") or "agy",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--dangerously-skip-permissions",
            "--effort",
            reasoning_effort,
            "--add-dir",
            str(workspace),
        ]
        if model:
            cmd.extend(["--model", model])
        return cmd

    def execute(
        self, task_ref: str, install: Install, run_ids: list[str], timeout: int
    ) -> list[Path]:
        if self._launcher is None:
            raise RuntimeError("AgySubstrate was constructed without a launcher body.")
        return self._launcher(
            RunRequest(task_ref=task_ref, run_ids=tuple(run_ids), timeout=int(timeout)), install
        )

    def read_trajectory(
        self, stream_path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 2000
    ) -> str:
        return render_agy_trajectory(
            Path(stream_path), obs_trunc=obs_trunc, max_chars=max_chars, args_trunc=args_trunc
        )

    def preflight(self) -> None:
        if not shutil.which("agy"):
            raise RuntimeError("agy CLI is not installed or not in PATH")


_TRUNC_FMT = "…(+{n} chars — render truncation)"


def _clip(s, n: int) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\x00", "")
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + _TRUNC_FMT.format(n=len(s) - n)


def render_agy_trajectory(
    path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 2000
) -> str:
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"[READ ERROR] {e}"

    lines: list[str] = []
    # Try parsing as JSON first
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            conv_id = data.get("conversation_id")
            # If transcript file exists, try reading steps
            transcript_file = None
            if conv_id:
                app_data = Path.home() / ".gemini" / "antigravity-cli" / "brain" / conv_id / ".system_generated" / "logs" / "transcript.jsonl"
                if app_data.is_file():
                    transcript_file = app_data
            if transcript_file and transcript_file.is_file():
                for tline in transcript_file.read_text(errors="replace").splitlines():
                    if not tline.strip():
                        continue
                    try:
                        step = json.loads(tline)
                    except json.JSONDecodeError:
                        continue
                    stype = step.get("type", "")
                    content = step.get("content", "")
                    thinking = step.get("thinking", "")
                    tool_calls = step.get("tool_calls") or []
                    if thinking:
                        lines.append(f"[THOUGHT] {_clip(thinking, obs_trunc)}")
                    if content and stype == "USER_INPUT":
                        lines.append(f"[USER] {_clip(content, obs_trunc)}")
                    elif content:
                        lines.append(f"[ASSISTANT] {_clip(content, obs_trunc)}")
                    for tc in tool_calls:
                        name = tc.get("name") if isinstance(tc, dict) else str(tc)
                        args = tc.get("args") if isinstance(tc, dict) else ""
                        lines.append(f"[ACTION] {name}: {_clip(json.dumps(args) if isinstance(args, dict) else str(args), args_trunc)}")
            else:
                resp = data.get("response", "")
                if resp:
                    lines.append(f"[ASSISTANT] {_clip(resp, obs_trunc)}")
    except json.JSONDecodeError:
        # Fallback: line-by-line JSONL or plain text
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    if "response" in obj:
                        lines.append(f"[ASSISTANT] {_clip(obj['response'], obs_trunc)}")
                    elif "content" in obj:
                        lines.append(f"[OBSERVATION] {_clip(obj['content'], obs_trunc)}")
                    continue
                except Exception:
                    pass
            lines.append(f"[LOG] {_clip(line, obs_trunc)}")

    out = "\n".join(lines).strip()
    return _clip(out, max_chars)
