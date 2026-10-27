"""calibrate.py - stage 4: calibration, before anything else is trusted.

One agent, no board, N_ROUNDS rounds per run, in the same prompt shape as the real
experiment (same build_prompt, including the agent's own score history). Four arms: each
payload's scorer, with and without that payload's seed note. Read the output to check:
  * unseeded vs seeded success within a payload: does the payload help?
  * lenient_correct: raw arithmetic accuracy, i.e. is the question pool hard enough (hack
    mode needs room to gain) without being hopeless?
  * strict_format / final_in_answer / audited_in_answer when unseeded: reinvention, with
    feedback (final_in_answer counts any use of FINAL:, even when the exact format fails).
Counts are printed as k/n. With this few calls treat them as a rough check, not a result.

Each attempt writes to its own folder (logs/calibration/<timestamp>/), so tuning the
question ranges and re-running never overwrites or mixes with earlier attempts.

Usage:  python calibrate.py --dry-run      (stub model, no Ollama calls)
        python calibrate.py                (real model; --runs N changes runs per arm)
"""

import argparse
import csv
import os
from datetime import datetime, timezone

from config import (CALIBRATION_ARMS, CALIBRATION_RUNS, LOG_DIR, MODEL, N_ROUNDS, PAYLOADS,
                    QUESTION_RANGES, RESULTS_DIR)
from orchestrator import fake_model, get_ollama_info, ollama_chat, run_one

COUNTED = ["score", "strict_format", "strict_correct", "lenient_correct", "near_miss_format",
           "final_in_answer", "final_elsewhere", "audited_in_answer", "audited_elsewhere"]


def summarise(records):
    """Raw counts for one arm."""
    row = {"calls": len(records)}
    for key in COUNTED:
        row[key] = sum(bool(r[key]) for r in records)
    row["parse_failures"] = sum(not r["parse_ok"] for r in records)
    row["ctx_warnings"] = sum(bool(r["ctx_warning"]) for r in records)
    row["mean_seconds"] = round(sum(r["seconds"] for r in records) / max(len(records), 1), 2)
    row["score_by_round"] = [sum(r["score"] for r in records if r["round"] == rnd)
                             for rnd in range(1, N_ROUNDS + 1)]
    return row


def print_row(row, runs):
    n = row["calls"]
    print(f"\n{row['payload']} / {row['arm']}  ({n} calls, {row['mean_seconds']} s per call on average)")
    print("  " + " | ".join(f"{k} {row[k]}/{n}" for k in COUNTED))
    print(f"  parse failures {row['parse_failures']}/{n} | ctx warnings {row['ctx_warnings']}/{n}")
    print(f"  score by round (out of {runs}): {row['score_by_round']}")


def main():
    ap = argparse.ArgumentParser(description="Stage 4 calibration.")
    ap.add_argument("--dry-run", action="store_true", help="use the stub model; no Ollama calls")
    ap.add_argument("--runs", type=int, default=CALIBRATION_RUNS, help="runs per arm (each run = one question per round)")
    args = ap.parse_args()

    if args.dry_run:
        fn, name, stamp = fake_model, "fake-model", "dry_run"
        info = {"ollama_version": "dry-run", "model_digest": "dry-run"}
    else:
        fn, name, info = ollama_chat, MODEL, get_ollama_info()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOG_DIR, "calibration", stamp)
    print(f"model {name}; question ranges {QUESTION_RANGES}; logs in {log_dir}")

    rows = []
    for payload in PAYLOADS:
        for arm in CALIBRATION_ARMS:
            records = []
            for i in range(args.runs):
                print(f"{payload} {arm}: run {i + 1}/{args.runs}", flush=True)
                records += run_one(payload, arm, i, fn, log_dir, name, info, args.dry_run,
                                   n_agents=1, question_ns="cal-questions")
            rows.append({"payload": payload, "arm": arm, **summarise(records)})
            print_row(rows[-1], args.runs)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"calibration_{stamp}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "score_by_round": ";".join(map(str, row["score_by_round"]))})
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()