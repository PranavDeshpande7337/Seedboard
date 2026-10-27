"""orchestrator.py - runs ONE experimental run: agents, rounds, board, scoring, logging.

One run = N_AGENTS agents x N_ROUNDS rounds. Per round, per agent (fixed order 1..4):
  1. build the prompt (base prompt + optional seed note + board + last score + question),
  2. call the model (Ollama, or the dry-run stub),
  3. parse the reply and score it,
  4. write one JSONL line.
Board posts are collected during a round and become visible from the NEXT round, so the
order of agents within a round cannot matter.

Each agent call is a single, independent prompt: an agent sees only what the prompt shows
it: its seed note (seed agent only), the board, its own previous rounds (question, its
[ANSWER] section and its score; never its reasoning, other agents' answers or the true
answer), and its question.

Usage (one run, for testing):
    python orchestrator.py trick A 0 --dry-run
    python orchestrator.py trick A 0
"""

import argparse
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from config import (BASE_PROMPT, BASE_SEED, BOARD_MAX_WORDS, CALIBRATION_ARMS, CONDITIONS,
                    CTX_WARN_FRACTION, LOG_DIR, MODEL, N_AGENTS, N_ROUNDS, NUM_CTX, NUM_PREDICT,
                    OLLAMA_URL, PAYLOADS, QUESTION_RANGES, SEED_NOTES, TEMPERATURE)
from questions import make_questions
from scorer import parse_reply, score_reply


# --- Seeds ---------------------------------------------------------------------------
def derive_seed(*parts):
    """Stable 32-bit seed from any parts. Uses hashlib because Python's built-in hash() of a
    string changes between processes, which would break repeatability."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)


def make_run_id(payload, condition, run_index):
    return f"{payload}-{condition}-{run_index:02d}"


def config_fingerprint(model_name=MODEL):
    """Short hash of every setting that decides what the model sees and how it is sampled
    (prompt, seed notes, question ranges, model, temperature, context size and so on). It is
    stored in every log line so run_experiments.py can refuse to mix runs made with different
    settings. It covers settings, not code."""
    settings = {"model": model_name, "temperature": TEMPERATURE, "num_ctx": NUM_CTX,
                "num_predict": NUM_PREDICT, "n_agents": N_AGENTS, "n_rounds": N_ROUNDS,
                "board_words": BOARD_MAX_WORDS, "base_seed": BASE_SEED, "prompt": BASE_PROMPT,
                "seed_notes": SEED_NOTES, "question_ranges": QUESTION_RANGES, "conditions": CONDITIONS}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]


# --- Model calls ------------------------------------------------------------------------
def _get_json(path):
    with urllib.request.urlopen(OLLAMA_URL + path, timeout=30) as r:
        return json.load(r)


def get_ollama_info(model=MODEL):
    """Ollama server version and the model's digest, for the logs. Never stops a run."""
    info = {"ollama_version": "unknown", "model_digest": "unknown"}
    try:
        info["ollama_version"] = _get_json("/api/version").get("version", "unknown")  # endpoint from memory: unverified
        for m in _get_json("/api/tags")["models"]:
            if m["name"] == model:
                info["model_digest"] = m["digest"]
    except Exception as err:  # info is nice to have, not essential
        print(f"warning: could not read Ollama info ({err})")
    return info


def ollama_chat(prompt, seed, truth=None):
    """One non-streaming /api/chat call. `truth` is accepted only so the stub and this
    function share a signature; the real model never sees it."""
    body = {"model": MODEL, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"seed": seed, "temperature": TEMPERATURE,
                        "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT}}
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.load(r)
    except urllib.error.URLError as err:
        raise RuntimeError(f"Could not reach Ollama at {OLLAMA_URL}: {err}") from err
    return {"content": data["message"]["content"], "prompt_tokens": data.get("prompt_eval_count"),
            "eval_tokens": data.get("eval_count"), "done_reason": data.get("done_reason")}


def fake_model(prompt, seed, truth=None):
    """Dry-run stub: a made-up model that exercises the whole pipeline with no model calls.
    It only 'knows' a payload if FINAL: or AUDITED appears in its prompt (seed note or board),
    then uses and sometimes posts it. 5% of replies lose their tags to test parse failures."""
    rng = random.Random(seed)
    number = truth if rng.random() < 0.6 else truth + rng.randint(1, 9)
    answer = f"The result is {number}."
    post = "NONE"
    if "FINAL:" in prompt:
        if rng.random() < 0.8:
            answer += f"\nFINAL: {number}"
        if rng.random() < 0.5:
            post = "The scorer wants the last line to be FINAL: <number>."
    if "AUDITED" in prompt:
        if rng.random() < 0.8:
            answer = "AUDITED " + answer
        if post == "NONE" and rng.random() < 0.5:
            post = "Including the word AUDITED seems to earn full credit."
    reply = f"Working it through step by step.\n[BOARD]\n{post}\n[ANSWER]\n{answer}"
    if rng.random() < 0.05:
        reply = f"The result is {number}."
    return {"content": reply, "prompt_tokens": len(prompt.split()),
            "eval_tokens": len(reply.split()), "done_reason": "stop"}


# --- Prompt -----------------------------------------------------------------------------
def board_text(board):
    """The board as shown to agents. board = list of (round, agent, post)."""
    if not board:
        return "Notes posted by agents in earlier rounds: none yet."
    lines = [f"Round {r}, Agent {a}: {' '.join(post.split())}" for r, a, post in board]
    return "Notes posted by agents in earlier rounds:\n" + "\n".join(lines)


def history_text(history):
    """An agent's own previous rounds. history = list of (round, question, answer section, score).
    Deliberately leaves out its reasoning, other agents' answers and the true answer."""
    lines = ["Your previous rounds:"]
    for rnd, question, answer, score in history:
        lines += [f"Round {rnd}. Question: {question}",
                  f"Your answer section: {answer or '(empty)'}",
                  f"Your score: {score}"]
    lines.append("(Scores are 0 or 1.)")
    return "\n".join(lines)


def build_prompt(agent, question, history, board, board_on, seed_note):
    """Base prompt plus the only things allowed to vary: seed note, board section, own history."""
    parts = []
    if seed_note:
        parts.append(seed_note)
    if board_on:
        parts.append(board_text(board))
    if history:
        parts.append(history_text(history))
    extra = "".join(p + "\n\n" for p in parts)
    return BASE_PROMPT.format(agent_no=agent, n_agents=N_AGENTS, extra=extra,
                              question=question, board_words=BOARD_MAX_WORDS)


# --- One run ------------------------------------------------------------------------------
def run_one(payload, condition, run_index, model_fn, log_dir, model_name=MODEL, info=None,
            dry_run=False, n_agents=N_AGENTS, question_ns="questions"):
    """Run one run and write logs/<run_id>.jsonl (via a .tmp file, so a crashed run leaves no
    finished-looking file). Returns the list of log records.

    The seed agent rotates with run_index, and the questions depend only on run_index, so
    conditions A, B and C (and both payloads) share questions and seed-agent position for the
    same run_index. In B the 'seed agent' slot is nominal (nobody gets the note): a placebo.
    n_agents and question_ns exist only for calibration (one agent, its own question set);
    condition may then be one of the CALIBRATION_ARMS.
    """
    cond = {**CONDITIONS, **CALIBRATION_ARMS}[condition]
    info = info or {"ollama_version": "unknown", "model_digest": "unknown"}
    run_id = make_run_id(payload, condition, run_index)
    fingerprint = config_fingerprint(model_name)
    run_seed = derive_seed(BASE_SEED, payload, condition, run_index)
    seed_agent = run_index % n_agents + 1
    questions = make_questions(n_agents * N_ROUNDS, derive_seed(BASE_SEED, question_ns, run_index))

    board, records = [], []
    history = {a: [] for a in range(1, n_agents + 1)}  # each agent's own previous rounds
    os.makedirs(log_dir, exist_ok=True)
    tmp_path = os.path.join(log_dir, run_id + ".jsonl.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rnd in range(1, N_ROUNDS + 1):
            new_posts = []
            for agent in range(1, n_agents + 1):
                q = questions[(rnd - 1) * n_agents + agent - 1]
                note = SEED_NOTES[payload] if cond["seeded"] and agent == seed_agent else None
                prompt = build_prompt(agent, q["text"], history[agent], board, cond["board"], note)
                call_seed = derive_seed(run_seed, agent, rnd)
                started = time.perf_counter()
                resp = model_fn(prompt, call_seed, truth=q["answer"])
                seconds = round(time.perf_counter() - started, 2)

                parsed = parse_reply(resp["content"])
                result = score_reply(resp["content"], q["answer"], payload)
                history[agent].append((rnd, q["text"], parsed["answer"].strip(), result["score"]))
                if parsed["board"]:
                    new_posts.append((rnd, agent, parsed["board"]))

                tokens = resp["prompt_tokens"]
                record = {
                    "run_id": run_id, "condition": condition, "payload": payload,
                    "round": rnd, "agent": agent, "seed_agent": seed_agent,
                    "is_seed_agent": agent == seed_agent, "note_given": note is not None,
                    "board_visible": cond["board"],
                    "prompt": prompt, "raw_reply": resp["content"],
                    "parsed_answer": parsed["answer"].strip(), "board_post": parsed["board"],
                    "board_truncated": parsed["board_truncated"],
                    "score": result["score"], "parse_ok": not parsed["tags_missing"],
                    "strict_format": result["strict_format"], "strict_correct": result["strict_correct"],
                    "lenient_correct": result["lenient_correct"], "near_miss_format": result["near_miss_format"],
                    "audited_in_answer": result["audited_in_answer"], "audited_elsewhere": result["audited_elsewhere"],
                    "final_in_answer": result["final_in_answer"], "final_elsewhere": result["final_elsewhere"],
                    "question_id": q["id"], "question_kind": q["kind"], "truth": q["answer"],
                    "timestamp": datetime.now(timezone.utc).isoformat(), "seconds": seconds,
                    "model": model_name, "seed": call_seed, "run_seed": run_seed,
                    "temperature": TEMPERATURE, "num_ctx": NUM_CTX, "prompt_tokens": tokens,
                    "ctx_warning": tokens is None or tokens > CTX_WARN_FRACTION * NUM_CTX,
                    "eval_tokens": resp["eval_tokens"], "done_reason": resp["done_reason"],
                    "ollama_version": info["ollama_version"], "model_digest": info["model_digest"],
                    "dry_run": dry_run, "config_fingerprint": fingerprint,
                }
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            board.extend(new_posts)  # visible from the next round only
    os.replace(tmp_path, os.path.join(log_dir, run_id + ".jsonl"))
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run one run of the experiment.")
    ap.add_argument("payload", choices=PAYLOADS)
    ap.add_argument("condition", choices=sorted(CONDITIONS))
    ap.add_argument("run_index", type=int)
    ap.add_argument("--dry-run", action="store_true", help="use the stub model; no Ollama calls")
    ap.add_argument("--log-dir", help="default: logs/ (logs/dry_run/ with --dry-run)")
    args = ap.parse_args()

    if args.dry_run:
        fn, name, info = fake_model, "fake-model", {"ollama_version": "dry-run", "model_digest": "dry-run"}
        log_dir = args.log_dir or os.path.join(LOG_DIR, "dry_run")
    else:
        fn, name, info = ollama_chat, MODEL, get_ollama_info()
        log_dir = args.log_dir or LOG_DIR
    recs = run_one(args.payload, args.condition, args.run_index, fn, log_dir, name, info, args.dry_run)
    for rnd in range(1, N_ROUNDS + 1):
        rr = [r for r in recs if r["round"] == rnd]
        print(f"round {rnd}: scores {[r['score'] for r in rr]}  posts {sum(bool(r['board_post']) for r in rr)}"
              f"  parse failures {sum(not r['parse_ok'] for r in rr)}")
    print(f"wrote {len(recs)} lines to {log_dir}")