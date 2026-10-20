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

"""Regressions for newly added substrates (Agy, Claude) and system tool discovery."""

import json
import tempfile
import unittest
from pathlib import Path

from harness2 import config
from harness2.core import tree
from harness2.core.spec import BenchSpec, resolve_channel_model
from harness2.demo import DemoAdapter
from harness2.substrate.agy import AGY_SURFACE, AgySubstrate, render_agy_trajectory
from harness2.substrate.claude import CLAUDE_SURFACE, ClaudeSubstrate, render_claude_trajectory


class SubstratesTests(unittest.TestCase):
    def test_channel_resolution_agy(self):
        ch = resolve_channel_model("agy", "gemini-2.5-pro")
        self.assertEqual(ch.kind, "agy")
        self.assertEqual(ch.model, "gemini-2.5-pro")

    def test_agy_surface_and_materialise(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            harness_dir = tmp / "harness"
            harness_dir.mkdir()
            (harness_dir / "AGENTS.md").write_text("# Test instructions\n")
            (harness_dir / "scripts").mkdir()
            (harness_dir / "scripts" / "test.py").write_text("print('test')\n")

            sub = AgySubstrate()
            self.assertEqual(sub.surface().name, "agy")
            spec = BenchSpec(
                name="test",
                runs_root=tmp / "runs",
                install_mode="agy_root",
                prompts=DemoAdapter(tmp).spec.prompts,
                channels=DemoAdapter(tmp).spec.channels,
            )
            install = sub.materialise(harness_dir, spec, {})
            self.assertTrue((install.staging_dir / "AGENTS.md").is_file())
            self.assertTrue((install.staging_dir / "scripts" / "test.py").is_file())

            # Test argv generation
            argv = sub.exec_argv("solve this", tmp, model="gemini-2.5-pro", reasoning_effort="high")
            self.assertIn("-p", argv)
            self.assertIn("solve this", argv)
            self.assertIn("--output-format", argv)
            self.assertIn("json", argv)
            self.assertIn("--model", argv)
            self.assertIn("gemini-2.5-pro", argv)

    def test_claude_surface_and_materialise(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            harness_dir = tmp / "harness"
            harness_dir.mkdir()
            (harness_dir / "CLAUDE.md").write_text("# Test instructions\n")

            sub = ClaudeSubstrate()
            self.assertEqual(sub.surface().name, "claude")
            spec = BenchSpec(
                name="test",
                runs_root=tmp / "runs",
                install_mode="claude_root",
                prompts=DemoAdapter(tmp).spec.prompts,
                channels=DemoAdapter(tmp).spec.channels,
            )
            install = sub.materialise(harness_dir, spec, {})
            self.assertTrue((install.staging_dir / "CLAUDE.md").is_file())

            argv = sub.exec_argv("run task", tmp)
            self.assertIn("-p", argv)
            self.assertIn("run task", argv)

    def test_agy_trajectory_parsing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            f = Path(tmp_dir) / "output.json"
            f.write_text(json.dumps({
                "conversation_id": "test-123",
                "status": "SUCCESS",
                "response": "Final computed result: 42.00\n",
            }))
            rendered = render_agy_trajectory(f, obs_trunc=1000, max_chars=5000)
            self.assertIn("[ASSISTANT] Final computed result: 42.00", rendered)

    def test_demo_adapter_with_agy_substrate(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            demo = DemoAdapter(root, substrate="agy")
            self.assertEqual(demo.substrate.name, "agy")
            self.assertTrue((demo.base_harness() / "AGENTS.md").is_file())
            self.assertTrue((demo.base_harness() / "scripts" / "summarize.py").is_file())


if __name__ == "__main__":
    unittest.main()
