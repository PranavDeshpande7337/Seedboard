"""test_orchestrator.py - stage 3 tests. No model calls: a stub model and a mocked HTTP call.
Run with:  python test_orchestrator.py"""

import io
import json
import os
import re
import tempfile
import unittest
from unittest import mock

import calibrate
import orchestrator as orch
from config import BASE_SEED, CONDITIONS, N_AGENTS, N_ROUNDS, PAYLOADS
from questions import make_questions


def load(log_dir, run_id):
    with open(os.path.join(log_dir, run_id + ".jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run(payload="trick", cond="A", idx=0, fn=orch.fake_model):
    d = tempfile.mkdtemp()
    return orch.run_one(payload, cond, idx, fn, d, "fake-model", dry_run=True), d


def scripted_model():
    """Stub that posts a numbered note every call and answers correctly in trick format.
    Relies on the fixed call order: agents 1..4 within rounds 1..N."""
    calls = {"n": 0}

    def fn(prompt, seed, truth=None):
        rnd, agent = calls["n"] // N_AGENTS + 1, calls["n"] % N_AGENTS + 1
        calls["n"] += 1
        return {"content": f"x\n[BOARD]\nPOST-R{rnd}-A{agent}\n[ANSWER]\nFINAL: {truth}",
                "prompt_tokens": 100, "eval_tokens": 20, "done_reason": "stop"}
    return fn


class Seeds(unittest.TestCase):
    def test_fingerprint_is_stable_logged_and_sensitive_to_settings(self):
        self.assertEqual(orch.config_fingerprint("m"), orch.config_fingerprint("m"))
        self.assertNotEqual(orch.config_fingerprint("m"), orch.config_fingerprint("other-model"))
        with mock.patch.object(orch, "TEMPERATURE", 0.9):
            self.assertNotEqual(orch.config_fingerprint("m"), orch.config_fingerprint("fake-model"))
        recs, _ = run()
        self.assertEqual({r["config_fingerprint"] for r in recs}, {orch.config_fingerprint("fake-model")})

    def test_derive_seed_is_stable_and_distinct(self):
        self.assertEqual(orch.derive_seed(1, "a", 2), orch.derive_seed(1, "a", 2))
        self.assertNotEqual(orch.derive_seed(1, "a", 2), orch.derive_seed(1, "a", 3))
        self.assertLess(orch.derive_seed(1, "a", 2), 2 ** 32)


class DryRunStructure(unittest.TestCase):
    def test_one_line_per_agent_round_and_no_tmp_file(self):
        recs, d = run()
        self.assertEqual(len(recs), N_AGENTS * N_ROUNDS)
        self.assertEqual({(r["round"], r["agent"]) for r in recs},
                         {(a, b) for a in range(1, N_ROUNDS + 1) for b in range(1, N_AGENTS + 1)})
        self.assertEqual(os.listdir(d), ["trick-A-00.jsonl"])
        self.assertEqual(len(load(d, "trick-A-00")), len(recs))

    def test_required_fields_logged(self):
        recs, _ = run()
        needed = {"run_id", "condition", "payload", "round", "agent", "is_seed_agent", "raw_reply",
                  "parsed_answer", "board_post", "score", "parse_ok", "timestamp", "model", "seed",
                  "temperature", "prompt", "prompt_tokens", "ctx_warning", "done_reason",
                  "ollama_version", "model_digest", "board_visible", "seconds",
                  "final_in_answer", "final_elsewhere", "config_fingerprint"}
        self.assertTrue(needed <= set(recs[0]))

    def test_every_question_is_different(self):
        recs, _ = run()
        self.assertEqual(len({r["question_id"] for r in recs}), N_AGENTS * N_ROUNDS)

    def test_same_inputs_give_same_log(self):
        a, _ = run()
        b, _ = run()
        self.assertEqual([r["raw_reply"] for r in a], [r["raw_reply"] for r in b])
        self.assertEqual([r["seed"] for r in a], [r["seed"] for r in b])


class SeedAgentAndPrompts(unittest.TestCase):
    def test_seed_agent_rotates(self):
        self.assertEqual([run(idx=i)[0][0]["seed_agent"] for i in range(5)], [1, 2, 3, 4, 1])

    def test_only_seed_agent_gets_note_in_seeded_conditions(self):
        for payload in PAYLOADS:
            for cond in ("A", "C"):
                recs, _ = run(payload, cond, 1)
                for r in recs:
                    self.assertEqual(r["note_given"], r["agent"] == 2)

    def test_condition_b_gives_nobody_a_note_but_logs_nominal_slot(self):
        recs, _ = run("trick", "B", 2)
        self.assertFalse(any(r["note_given"] for r in recs))
        self.assertEqual(recs[0]["seed_agent"], 3)
        self.assertTrue(any(r["is_seed_agent"] for r in recs))

    def test_base_prompt_never_shows_a_canary(self):
        for payload in PAYLOADS:
            p = orch.build_prompt(2, "What is 2 + 2?", [], [], True, None)
            self.assertNotIn("FINAL", p)
            self.assertNotIn("AUDITED", p)
            self.assertNotIn("experiment", p.lower())
            self.assertNotIn("evaluation", p.lower())

    def test_round_one_prompts_of_non_seed_agents_carry_no_canary(self):
        for payload in PAYLOADS:
            for cond in CONDITIONS:
                recs, _ = run(payload, cond, 0)
                for r in recs:
                    if r["round"] == 1 and not r["note_given"]:
                        self.assertNotIn("FINAL", r["prompt"])
                        self.assertNotIn("AUDITED", r["prompt"])

    def test_seed_note_is_present_for_seed_agent_only(self):
        recs, _ = run("hack", "A", 0)
        r1 = [r for r in recs if r["round"] == 1]
        self.assertTrue(["AUDITED" in r["prompt"] for r in r1] == [True, False, False, False])

    def test_condition_c_never_shows_a_board(self):
        recs, _ = run("trick", "C", 0, scripted_model())
        self.assertFalse(any("Notes posted" in r["prompt"] for r in recs))
        self.assertTrue(any(r["board_post"] for r in recs))  # posts are still logged

    def test_board_posts_appear_only_from_the_next_round(self):
        recs, _ = run("trick", "A", 0, scripted_model())
        for r in recs:
            for rnd in range(1, N_ROUNDS + 1):
                for agent in range(1, N_AGENTS + 1):
                    visible = f"POST-R{rnd}-A{agent}" in r["prompt"]
                    self.assertEqual(visible, rnd < r["round"], (r["round"], rnd, agent))

    def test_board_is_empty_in_round_one(self):
        recs, _ = run("trick", "B", 0, scripted_model())
        self.assertTrue(all("none yet" in r["prompt"] for r in recs if r["round"] == 1))

    def test_history_shown_from_round_two_with_scores(self):
        recs, _ = run("trick", "A", 0, scripted_model())
        for r in recs:
            self.assertEqual("Your previous rounds:" in r["prompt"], r["round"] > 1)
            self.assertEqual("Your score: 1" in r["prompt"], r["round"] > 1)

    def test_board_visible_flag(self):
        for cond, expected in (("A", True), ("B", True), ("C", False)):
            recs, _ = run("trick", cond, 0)
            self.assertTrue(all(r["board_visible"] is expected for r in recs), cond)


def numbered_model():
    """Stub with secret reasoning and a numbered answer section that is never correct."""
    calls = {"n": 0}

    def fn(prompt, seed, truth=None):
        n = calls["n"]
        calls["n"] += 1
        return {"content": f"SECRET-REASONING-{n}\n[BOARD]\nNONE\n[ANSWER]\nANS-{n}",
                "prompt_tokens": 100, "eval_tokens": 10, "done_reason": "stop"}
    return fn


class PromptWording(unittest.TestCase):
    def test_board_line_nudges_sharing_without_naming_a_payload(self):
        from config import BASE_PROMPT
        self.assertIn("anything you have learned that might help them", BASE_PROMPT)
        self.assertNotIn("FINAL", BASE_PROMPT)
        self.assertNotIn("AUDITED", BASE_PROMPT)

    def test_board_instruction_is_identical_in_every_condition_and_agent(self):
        lines = set()
        for payload in PAYLOADS:
            for cond in CONDITIONS:
                for r in run(payload, cond, 0)[0]:
                    start = r["prompt"].index("2. A line containing only [BOARD]")
                    lines.add(r["prompt"][start:r["prompt"].index("3. A line containing only [ANSWER]")])
        self.assertEqual(len(lines), 1)


class OwnHistory(unittest.TestCase):
    def test_history_block_is_exact_and_leaks_nothing(self):
        recs, _ = run("trick", "A", 0, numbered_model())
        qs = make_questions(N_AGENTS * N_ROUNDS, orch.derive_seed(BASE_SEED, "questions", 0))
        total = N_AGENTS * N_ROUNDS
        for r in recs:
            self.assertNotIn("SECRET-REASONING", r["prompt"])          # never its reasoning
            mine = {(k - 1) * N_AGENTS + r["agent"] - 1 for k in range(1, N_ROUNDS + 1)}
            for n in set(range(total)) - mine:                          # never others' answers
                self.assertIsNone(re.search(rf"ANS-{n}\b", r["prompt"]), (r["round"], r["agent"], n))
            for k in range(1, r["round"]):                              # exactly the agreed block
                n = (k - 1) * N_AGENTS + r["agent"] - 1
                block = (f"Round {k}. Question: {qs[n]['text']}\n"
                         f"Your answer section: ANS-{n}\nYour score: 0\n")
                self.assertIn(block, r["prompt"])
            self.assertNotIn(f"Round {r['round']}. Question", r["prompt"])  # nothing from now or later

    def test_empty_answer_section_is_shown_as_empty(self):
        self.assertIn("Your answer section: (empty)", orch.history_text([(1, "Q?", "", 0)]))


class ParsingAndScoringInLogs(unittest.TestCase):
    def test_scripted_correct_answers_score_one_in_trick_mode(self):
        recs, _ = run("trick", "A", 0, scripted_model())
        self.assertTrue(all(r["score"] == 1 and r["parse_ok"] for r in recs))

    def test_parse_failure_is_logged_and_board_post_dropped(self):
        def no_tags(prompt, seed, truth=None):
            return {"content": f"FINAL: {truth}", "prompt_tokens": 50, "eval_tokens": 5, "done_reason": "stop"}
        recs, _ = run("trick", "A", 0, no_tags)
        self.assertTrue(all(not r["parse_ok"] and r["board_post"] is None for r in recs))

    def test_context_warning_flag(self):
        def big(prompt, seed, truth=None):
            return {"content": "x", "prompt_tokens": 7000, "eval_tokens": 1, "done_reason": "stop"}
        recs, _ = run("trick", "A", 0, big)
        self.assertTrue(all(r["ctx_warning"] for r in recs))
        recs, _ = run("trick", "A", 0, scripted_model())
        self.assertFalse(any(r["ctx_warning"] for r in recs))

    def test_stub_uses_a_payload_only_after_seeing_it(self):
        recs, _ = run("trick", "B", 0)  # nobody is seeded and the board starts empty
        self.assertFalse(any(r["strict_format"] for r in recs))


class OllamaRequest(unittest.TestCase):
    def test_request_shape_and_response_parsing(self):
        reply = {"message": {"content": "hi"}, "prompt_eval_count": 12, "eval_count": 3, "done_reason": "stop"}
        with mock.patch("orchestrator.urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = io.BytesIO(json.dumps(reply).encode())
            out = orch.ollama_chat("hello", 99)
        req = urlopen.call_args[0][0]
        body = json.loads(req.data)
        self.assertTrue(req.full_url.endswith("/api/chat"))
        self.assertIs(body["stream"], False)
        self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(set(body["options"]), {"seed", "temperature", "num_ctx", "num_predict"})
        self.assertEqual(body["options"]["seed"], 99)
        self.assertEqual(out, {"content": "hi", "prompt_tokens": 12, "eval_tokens": 3, "done_reason": "stop"})


class Calibration(unittest.TestCase):
    def cal(self, payload, arm, fn=orch.fake_model):
        return orch.run_one(payload, arm, 0, fn, tempfile.mkdtemp(), "fake-model", dry_run=True,
                            n_agents=1, question_ns="cal-questions")

    def test_single_agent_run_shape(self):
        recs = self.cal("trick", "cal-seeded")
        self.assertEqual(len(recs), N_ROUNDS)
        self.assertTrue(all(r["agent"] == 1 and r["note_given"] for r in recs))
        self.assertFalse(any(r["board_visible"] or "Notes posted" in r["prompt"] for r in recs))
        self.assertTrue(all(("Your previous rounds:" in r["prompt"]) == (r["round"] > 1) for r in recs))

    def test_unseeded_arm_gives_no_note(self):
        self.assertFalse(any(r["note_given"] for r in self.cal("hack", "cal-unseeded")))

    def test_calibration_questions_differ_from_experiment_questions(self):
        cal = {r["question_id"] for r in self.cal("trick", "cal-seeded")}
        real = {r["question_id"] for r in run("trick", "C", 0)[0]}
        self.assertFalse(cal & real)

    def test_summarise_counts(self):
        s = calibrate.summarise(self.cal("hack", "cal-seeded", scripted_model()))
        self.assertEqual((s["calls"], s["score"], s["strict_format"], s["lenient_correct"]), (6, 6, 6, 6))
        self.assertEqual((s["audited_in_answer"], s["parse_failures"], s["ctx_warnings"]), (0, 0, 0))
        self.assertEqual(s["final_in_answer"], 6)
        self.assertEqual(s["score_by_round"], [1] * N_ROUNDS)


if __name__ == "__main__":
    unittest.main()