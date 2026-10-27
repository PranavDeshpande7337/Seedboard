"""run_experiments.py - stages 5 and 6: run the pilot or the full experiment, resumably.

Every cell (2 payloads x 3 conditions) gets the same number of runs, one run at a time.
Runs are ordered by run index first, then cell, so stopping at any point leaves the cells
evenly covered, and (because the seed agent rotates with run index) the seed-agent positions
balanced. Each run is its own log file, logs/<stage>/<run_id>.jsonl.

Resuming: run the same command again. A run is finished only if its log file is complete
(24 lines) and was made with the current settings (config_fingerprint). Anything else (no
file, a leftover .tmp from a stop, a truncated file) is run again. A complete file made with
DIFFERENT settings stops the program instead of being mixed in or overwritten: use a new
--log-dir, or delete that folder if you really want to start again.

Staging: `full --runs 8` first, then `full --runs 12` later; the first 8 runs per cell are
skipped the second time.

Usage:
    python run_experiments.py pilot --dry-run      (stub model; checks the machinery)
    python run_experiments.py pilot                (3 runs per cell = 18 runs)
    python run_experiments.py full [--runs N]      (default RUNS_PER_CELL per cell)
"""

import argparse
import json
import os
import time

from config import (CANARIES, CONDITIONS, LOG_DIR, MODEL, N_AGENTS, N_ROUNDS, PAYLOADS,
                    PILOT_RUNS_PER_CELL, RUNS_PER_CELL)
from orchestrator import (config_fingerprint, fake_model, get_ollama_info, make_run_id,
                          ollama_chat, run_one)


def plan(runs):
    """All (payload, condition, run_index) to run: run index outermost, then the six cells."""
    return [(payload, cond, i) for i in range(runs) for payload in PAYLOADS for cond in sorted(CONDITIONS)]


def is_finished(path, fingerprint):
    """True if path is a complete log made with the current settings. A missing, partial or
    truncated file is not finished. A complete file from different settings raises an error."""
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    if len(lines) != N_AGENTS * N_ROUNDS:
        return False
    if json.loads(lines[0]).get("config_fingerprint") != fingerprint:
        raise RuntimeError(f"{path} was made with different settings (model, prompt, temperature, "
                           f"seed notes, question ranges...) from the current config. Use a new "
                           f"--log-dir, or delete that folder to start again.")
    return True


def describe(records):
    """One-line summary of a finished run, for the progress output."""
    text = (f"score {sum(r['score'] for r in records)}/{len(records)} | "
            f"board posts {sum(bool(r['board_post']) for r in records)}")
    if any(r["note_given"] for r in records):  # seeded run: did the seed agent post the payload?
        canary = CANARIES[records[0]["payload"]]
        posted = sum(1 for r in records if r["note_given"] and r["board_post"] and canary in r["board_post"])
        text += f" | seed agent posted the payload {posted}x"
    return text + f" | parse failures {sum(not r['parse_ok'] for r in records)}"


def fmt(seconds):
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def progress(runs, log_dir, fingerprint):
    """Finished runs per (payload, condition) cell."""
    counts = {}
    for payload, cond, i in plan(runs):
        path = os.path.join(log_dir, make_run_id(payload, cond, i) + ".jsonl")
        counts[(payload, cond)] = counts.get((payload, cond), 0) + int(is_finished(path, fingerprint))
    return counts


def run_all(runs, log_dir, model_fn, model_name, info, dry_run=False, say=print):
    """Run everything in plan(runs) that is not finished. Returns a dict describing what happened."""
    fingerprint = config_fingerprint(model_name)
    todo, skipped = [], []
    for payload, cond, i in plan(runs):
        run_id = make_run_id(payload, cond, i)
        finished = is_finished(os.path.join(log_dir, run_id + ".jsonl"), fingerprint)  # may raise
        (skipped if finished else todo).append(run_id if finished else (payload, cond, i))
    stale = [f for f in os.listdir(log_dir) if f.endswith(".tmp")] if os.path.isdir(log_dir) else []
    say(f"{len(skipped) + len(todo)} runs planned ({runs} per cell): {len(skipped)} already finished, "
        f"{len(todo)} to do. Model {model_name}, logs in {log_dir}")
    if stale:
        say(f"{len(stale)} unfinished run file(s) from an earlier stop will be redone.")

    done, failed, interrupted, in_a_row, started = [], [], False, 0, time.time()
    for n, (payload, cond, i) in enumerate(todo, 1):
        run_id, t0 = make_run_id(payload, cond, i), time.time()
        try:
            records = run_one(payload, cond, i, model_fn, log_dir, model_name, info, dry_run)
        except KeyboardInterrupt:
            interrupted = True
            say("Stopped. Finished runs are saved; run the same command to continue.")
            break
        except Exception as err:  # one bad run should not lose the others
            failed.append(run_id)
            in_a_row += 1
            say(f"[{n}/{len(todo)}] {run_id} FAILED: {err}")
            if in_a_row >= 2:
                say("Two failures in a row, stopping (is Ollama running and the model installed?).")
                break
            continue
        in_a_row = 0
        done.append(run_id)
        eta = (time.time() - started) / len(done) * (len(todo) - n)
        say(f"[{n}/{len(todo)}] {run_id}  {fmt(time.time() - t0)} | {describe(records)} | about {fmt(eta)} left")

    counts = progress(runs, log_dir, fingerprint)
    say("Finished runs per cell: " + ", ".join(f"{p} {c} {k}/{runs}" for (p, c), k in counts.items()))
    return {"done": done, "skipped": skipped, "failed": failed, "interrupted": interrupted, "progress": counts}


def main():
    ap = argparse.ArgumentParser(description="Run the pilot or the full experiment (resumable).")
    ap.add_argument("stage", choices=["pilot", "full"])
    ap.add_argument("--runs", type=int, help="runs per cell (default: pilot 3, full RUNS_PER_CELL)")
    ap.add_argument("--dry-run", action="store_true", help="use the stub model; no Ollama calls")
    ap.add_argument("--log-dir", help="default: logs/<stage> (logs/dry_run/<stage> with --dry-run)")
    args = ap.parse_args()
    runs = args.runs or (PILOT_RUNS_PER_CELL if args.stage == "pilot" else RUNS_PER_CELL)

    if args.dry_run:
        fn, name, info = fake_model, "fake-model", {"ollama_version": "dry-run", "model_digest": "dry-run"}
        log_dir = args.log_dir or os.path.join(LOG_DIR, "dry_run", args.stage)
    else:
        fn, name, info = ollama_chat, MODEL, get_ollama_info()
        log_dir = args.log_dir or os.path.join(LOG_DIR, args.stage)
        if info["model_digest"] == "unknown":
            print(f"warning: could not find {MODEL} on the Ollama server; check Ollama is running.")
    try:
        result = run_all(runs, log_dir, fn, name, info, args.dry_run)
    except RuntimeError as err:
        raise SystemExit(f"error: {err}")
    raise SystemExit(0 if all(k == runs for k in result["progress"].values()) else 1)


if __name__ == "__main__":
    main()