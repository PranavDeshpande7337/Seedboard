"""test_run_experiments.py - tests for the resumable runner. Stub model only; no Ollama calls.
Run with:  python test_run_experiments.py"""

import os
import tempfile
import unittest
from unittest import mock

import orchestrator as orch
import run_experiments as rx
from config import CONDITIONS, N_AGENTS, N_ROUNDS, PAYLOADS

INFO = {"ollama_version": "dry-run", "model_digest": "dry-run"}
CELLS = len(PAYLOADS) * len(CONDITIONS)
CALLS_PER_RUN = N_AGENTS * N_ROUNDS


def flaky(should_fail, exc):
    """Stub that raises exc on the calls where should_fail(call_number) is true."""
    calls = {"n": 0}

    def fn(prompt, seed, truth=None):
        calls["n"] += 1
        if should_fail(calls["n"]):
            raise exc
        return orch.fake_model(prompt, seed, truth)
    return fn, calls


def go(runs, d, fn=orch.fake_model, say=None):
    return rx.run_all(runs, d, fn, "fake-model", INFO, dry_run=True, say=say or (lambda *a: None))


class Plan(unittest.TestCase):
    def test_every_cell_gets_every_run_index_in_interleaved_order(self):
        p = rx.plan(3)
        self.assertEqual(len(p), 3 * CELLS)
        self.assertEqual({i for _, _, i in p[:CELLS]}, {0})                 # run index 0 for all cells first
        self.assertEqual(len({(a, b) for a, b, _ in p}), CELLS)
        self.assertEqual(len(set(p)), len(p))


class Resume(unittest.TestCase):
    def test_second_call_skips_everything_and_calls_no_model(self):
        d = tempfile.mkdtemp()
        first = go(1, d)
        self.assertEqual((len(first["done"]), len(first["skipped"])), (CELLS, 0))
        fn, calls = flaky(lambda n: False, None)
        second = go(1, d, fn)
        self.assertEqual((len(second["done"]), len(second["skipped"]), calls["n"]), (0, CELLS, 0))

    def test_only_the_missing_run_is_redone(self):
        d = tempfile.mkdtemp()
        go(1, d)
        os.remove(os.path.join(d, "hack-B-00.jsonl"))
        self.assertEqual(go(1, d)["done"], ["hack-B-00"])

    def test_leftover_tmp_file_is_redone_and_removed(self):
        d = tempfile.mkdtemp()
        go(1, d)
        os.remove(os.path.join(d, "trick-A-00.jsonl"))
        open(os.path.join(d, "trick-A-00.jsonl.tmp"), "w").write("partial\n")
        messages = []
        self.assertEqual(go(1, d, say=messages.append)["done"], ["trick-A-00"])
        self.assertTrue(any("unfinished" in m for m in messages))
        self.assertFalse([f for f in os.listdir(d) if f.endswith(".tmp")])

    def test_truncated_file_counts_as_unfinished(self):
        d = tempfile.mkdtemp()
        go(1, d)
        path = os.path.join(d, "hack-C-00.jsonl")
        lines = open(path, encoding="utf-8").read().splitlines()[:5]
        open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        self.assertEqual(go(1, d)["done"], ["hack-C-00"])
        self.assertEqual(len(open(path, encoding="utf-8").read().splitlines()), CALLS_PER_RUN)

    def test_staging_adds_only_the_new_run_indices(self):
        d = tempfile.mkdtemp()
        go(1, d)
        second = go(2, d)
        self.assertEqual(len(second["skipped"]), CELLS)
        self.assertEqual(sorted(second["done"]), sorted(f"{p}-{c}-01" for p in PAYLOADS for c in CONDITIONS))

    def test_runs_from_different_settings_are_never_mixed(self):
        d = tempfile.mkdtemp()
        go(1, d)
        with mock.patch.object(orch, "TEMPERATURE", 0.9):
            with self.assertRaises(RuntimeError) as ctx:
                go(1, d)
        self.assertIn("different settings", str(ctx.exception))
        self.assertEqual(len([f for f in os.listdir(d) if f.endswith(".jsonl")]), CELLS)  # nothing overwritten


class Failures(unittest.TestCase):
    def test_a_failed_run_is_not_finished_and_the_rest_continue(self):
        d = tempfile.mkdtemp()
        fn, _ = flaky(lambda n: n == 1, RuntimeError("boom"))
        result = go(1, d, fn)
        self.assertEqual(result["failed"], ["trick-A-00"])
        self.assertEqual(len(result["done"]), CELLS - 1)
        self.assertFalse(os.path.exists(os.path.join(d, "trick-A-00.jsonl")))
        self.assertEqual(go(1, d)["done"], ["trick-A-00"])                 # picked up next time

    def test_two_failures_in_a_row_stop_the_program(self):
        d = tempfile.mkdtemp()
        fn, _ = flaky(lambda n: True, RuntimeError("Ollama down"))
        result = go(1, d, fn)
        self.assertEqual((len(result["failed"]), result["done"]), (2, []))

    def test_ctrl_c_stops_cleanly_and_keeps_finished_runs(self):
        d = tempfile.mkdtemp()
        fn, _ = flaky(lambda n: n == CALLS_PER_RUN + 6, KeyboardInterrupt())  # inside the second run
        result = go(1, d, fn)
        self.assertTrue(result["interrupted"])
        self.assertEqual(result["done"], ["trick-A-00"])
        self.assertTrue(os.path.exists(os.path.join(d, "trick-A-00.jsonl")))
        self.assertEqual(len(go(1, d)["done"]), CELLS - 1)                  # resumes the rest
        self.assertFalse([f for f in os.listdir(d) if f.endswith(".tmp")])


class Output(unittest.TestCase):
    def test_describe_and_progress_lines(self):
        d = tempfile.mkdtemp()
        messages = []
        go(1, d, say=messages.append)
        run_lines = [m for m in messages if m.startswith("[")]
        self.assertEqual(len(run_lines), CELLS)
        self.assertIn("seed agent posted the payload", run_lines[0])        # trick-A is seeded
        self.assertNotIn("seed agent posted", run_lines[1])                 # trick-B is not
        self.assertTrue(messages[-1].startswith("Finished runs per cell"))


if __name__ == "__main__":
    unittest.main()