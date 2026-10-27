"""questions.py - programmatic three-step arithmetic word questions.

Every question has a known integer answer, so no external dataset is needed.
Three question types ("kinds") are used; each needs three arithmetic steps.
All randomness comes from a seeded random.Random, so the same seed always
gives the same questions.

The number ranges live in config.QUESTION_RANGES so difficulty can be tuned
during calibration without touching this file.
"""

import random

from config import QUESTION_RANGES

# One formula per kind: the single source of truth for each answer.
FORMULAS = {
    "stock": lambda a, b, c, d: a * b - c + d,     # boxes x items, minus sold, plus delivered
    "share": lambda a, b, c, d: (a * b - c) // d,  # pot, minus spend, shared equally
    "train": lambda a, b, c, d: a * b - (c - d),   # seats, minus net passengers on board
}


def _stock(rng, r):
    a, b, d = rng.randint(*r["a"]), rng.randint(*r["b"]), rng.randint(*r["d"])
    c = rng.randint(a * b // 4, a * b // 2)
    text = (f"A shop opens with {a} boxes of pencils, and each box holds {b} pencils. "
            f"During the day it sells {c} pencils, and then a delivery brings {d} more pencils. "
            f"How many pencils does the shop have at the end of the day?")
    return [a, b, c, d], text


def _share(rng, r):
    a, b, d = rng.randint(*r["a"]), rng.randint(*r["b"]), rng.randint(*r["d"])
    pot = a * b
    each = rng.randint(max(3, pot // (4 * d)), pot // (2 * d))  # each person's final share
    c = pot - d * each                                         # makes the division exact
    text = (f"{a} friends each put {b} stickers into a shared pile. They use {c} stickers for a "
            f"poster and divide what is left equally among {d} of the friends. "
            f"How many stickers does each of those {d} friends get?")
    return [a, b, c, d], text


def _train(rng, r):
    a, b = rng.randint(*r["a"]), rng.randint(*r["b"])
    seats = a * b
    c = rng.randint(seats // 3, 2 * seats // 3)  # board at the first station
    d = rng.randint(10, c // 3)                  # get off at the second station
    text = (f"A train has {a} carriages with {b} seats in each. At the first station {c} "
            f"passengers board, and at the second station {d} of them get off. "
            f"How many seats are empty once the train leaves the second station?")
    return [a, b, c, d], text


BUILDERS = {"stock": _stock, "share": _share, "train": _train}


def make_questions(n, seed):
    """Return n distinct questions as dicts: id, kind, nums, text, answer."""
    rng = random.Random(seed)
    questions, seen = [], set()
    while len(questions) < n:
        kind = rng.choice(sorted(BUILDERS))
        nums, text = BUILDERS[kind](rng, QUESTION_RANGES[kind])
        if text in seen:
            continue
        seen.add(text)
        questions.append({"id": f"{seed}-{len(questions)}", "kind": kind, "nums": nums,
                          "text": text, "answer": FORMULAS[kind](*nums)})
    return questions