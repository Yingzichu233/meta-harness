from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import meta_harness
from benchmark import run_dir as benchmark_run_dir
from text_classification.agents.fewshot_memory import FewShotMemory, _seed_for_input
from text_classification.data.api import _with_context


def _args(**overrides):
    values = {
        "model": "deepseek-chat",  # matches the DeepSeek entry in config.yaml
        "run_name": "test-run",
        "fresh": False,
        "skip_baseline": True,
        "iterations": 0,
        "propose_timeout": 1,
        "test": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class DataContractTests(unittest.TestCase):
    def test_with_context_preserves_metadata(self) -> None:
        examples = _with_context(
            [
                {
                    "input": "question",
                    "target": "answer",
                    "context": "instructions",
                    "raw_question": "question",
                    "source_id": "example-1",
                }
            ]
        )

        self.assertEqual(examples[0]["input"], "instructions\n\nquestion")
        self.assertEqual(examples[0]["raw_question"], "question")
        self.assertEqual(examples[0]["source_id"], "example-1")

    def test_fewshot_predict_uses_stable_input_seed(self) -> None:
        input_text = "same input"
        expected = int.from_bytes(
            hashlib.sha256(input_text.encode()).digest()[:8], "big"
        )
        memory = FewShotMemory(lambda _: '{"final_answer": "ok"}')
        memory._format_examples_section = Mock(return_value="")

        memory.predict(input_text)

        self.assertEqual(_seed_for_input(input_text), expected)
        memory._format_examples_section.assert_called_once_with(seed=expected)


class HeldOutIsolationTests(unittest.TestCase):
    def test_default_evolution_never_runs_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            with (
                patch.object(meta_harness, "EVOLVE_DIR", Path(tmp)),
                patch.object(meta_harness, "AGENTS_DIR", Path(tmp) / "agents"),
                patch.object(
                    meta_harness,
                    "run_benchmark",
                    side_effect=lambda args: calls.append(args),
                ),
                redirect_stdout(io.StringIO()),
            ):
                meta_harness.run_evolve(_args())

        self.assertFalse(any("--test" in call for call in calls))

    def test_finalization_prints_test_results_and_blocks_evolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "logs" / "test-run"
            run_dir.mkdir(parents=True)
            (run_dir / "frontier_val.json").write_text(
                json.dumps({"_pareto": [{"system": "candidate", "val_accuracy": 75.0}]})
            )
            calls = []

            def fake_benchmark(args):
                calls.append(args)
                if "--memory" in args:
                    system = args[args.index("--memory") + 1]
                    for dataset in ("USPTO", "Symptom2Disease", "LawBench"):
                        result_dir = benchmark_run_dir(
                            meta_harness.RESULTS_DIR,
                            dataset,
                            system,
                            "deepseek-chat",
                            42,
                        )
                        result_dir.mkdir(parents=True, exist_ok=True)
                        (result_dir / "test.json").write_text('{"accuracy": 1.0}')
                stdout = (
                    "held-out test table" if args == ["--results", "--test"] else ""
                )
                return subprocess.CompletedProcess(args, 0, stdout, "")

            output = io.StringIO()
            with (
                patch.object(meta_harness, "EVOLVE_DIR", root),
                patch.object(meta_harness, "AGENTS_DIR", root / "agents"),
                patch.object(meta_harness, "run_benchmark", fake_benchmark),
                redirect_stdout(output),
            ):
                meta_harness.run_evolve(_args(test=True))
                with self.assertRaises(SystemExit):
                    meta_harness.run_evolve(_args())

            state = json.loads((run_dir / "finalized.json").read_text())

        self.assertEqual(state["status"], "complete")
        self.assertIn(["--memory", "candidate", "--test"], calls)
        self.assertIn(["--results", "--test"], calls)
        self.assertIn("held-out test table", output.getvalue())

    def test_benchmark_test_results_are_run_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(meta_harness, "EVOLVE_DIR", root),
                patch.object(meta_harness, "LOGS_DIR", root / "logs" / "run"),
                patch.object(
                    meta_harness, "RESULTS_DIR", root / "logs" / "run" / "results"
                ),
                patch.object(
                    meta_harness,
                    "run_cmd",
                    return_value=subprocess.CompletedProcess([], 0, "", ""),
                ) as run_cmd,
            ):
                meta_harness.run_benchmark(["--results", "--test"])

            command = run_cmd.call_args.args[0]

        results_flag = command.index("--results-dir")
        self.assertEqual(command[results_flag + 1], str(root / "logs/run/results"))
        self.assertEqual(command[-2:], ["--results", "--test"])

    def test_incomplete_finalization_allows_evolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "logs" / "test-run"
            run_dir.mkdir(parents=True)
            (run_dir / "frontier_val.json").write_text('{"_pareto": []}')

            def corrupt_benchmark(args):
                if "--memory" in args:
                    system = args[args.index("--memory") + 1]
                    for dataset in ("USPTO", "Symptom2Disease", "LawBench"):
                        result_dir = benchmark_run_dir(
                            meta_harness.RESULTS_DIR,
                            dataset,
                            system,
                            "deepseek-chat",
                            42,
                        )
                        result_dir.mkdir(parents=True, exist_ok=True)
                        (result_dir / "test.json").write_text("{}")
                return subprocess.CompletedProcess(args, 0, "", "")

            with (
                patch.object(meta_harness, "EVOLVE_DIR", root),
                patch.object(meta_harness, "AGENTS_DIR", root / "agents"),
                patch.object(meta_harness, "run_benchmark", corrupt_benchmark),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    meta_harness.run_evolve(_args(test=True))
                meta_harness.run_evolve(_args())

            state = json.loads((run_dir / "finalized.json").read_text())

        self.assertEqual(state["status"], "in_progress")


if __name__ == "__main__":
    unittest.main()
