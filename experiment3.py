"""
Experiment 3 -- sentence completion. No planted facts anywhere.

Experiments 1 and 2 both score facts INSERTED into a carrier. That has two problems: the
inserted sentence is a visible non-sequitur, so its encoding strength may be unrepresentative;
and the number of probes is capped by how many facts you are willing to plant (twelve).

Here nothing is inserted. Documents are unbroken corpus prose. Probes are real sentences taken
from A itself: give the model the opening words, score the true continuation.

    metric = log P(true continuation | prefix, cache)

compared across the same three conditions as before. There is no forced choice, so chance is
NOT zero and the absolute number means nothing -- the mismatched control is the only anchor,
which makes it load-bearing rather than merely important.

Probes are stratified by depth in A, so the primacy result from experiment 1 gets retested at
roughly twenty times the resolution.

    TARGET_TOKENS=4000 SEEDS=8 PROBES=200 python experiment3.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment import Ctx, Doc, _carrier_from_corpus, load_corpus, prefill, transplant  # noqa: E402
from gpu_run import load_real  # noqa: E402

SEEDS = list(range(int(os.environ.get("SEEDS", 8))))
TARGET_TOKENS = int(os.environ.get("TARGET_TOKENS", 4000))
PROBES = int(os.environ.get("PROBES", 200))
PREFIX_WORDS = 8          # words given to the model; the rest is scored
MIN_SENT_WORDS = 18       # a probe needs enough continuation to measure


def clean_docs(seed: int, target_tokens: int, n_docs: int = 3):
    """A, B, A2 as unbroken corpus prose. Disjoint paragraphs, nothing planted."""
    rng = random.Random(seed)
    corpus = load_corpus()
    used: set[int] = set()
    out = []
    for d in range(n_docs):
        want = int((600 if d == 1 else target_tokens) * 0.75)   # B stays short
        out.append(Doc(text=" ".join(_carrier_from_corpus(rng, want, used, corpus))))
    return out


def sentences_with_depth(text: str):
    """Split into sentences and record where each sits in the document, 0.0 to 1.0."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    out, pos, total = [], 0, len(text)
    for s in parts:
        s = s.strip()
        if len(s.split()) >= MIN_SENT_WORDS:
            out.append((s, min(pos / max(total, 1), 1.0)))
        pos += len(s) + 1
    return out


def pick(rng: random.Random, sents, k: int):
    """Stratified by depth: equal numbers from each of k/10 bands, so depth is covered evenly."""
    if len(sents) <= k:
        return sents
    bands, per = 10, max(1, k // 10)
    by = [[] for _ in range(bands)]
    for s, d in sents:
        by[min(int(d * bands), bands - 1)].append((s, d))
    out = []
    for b in by:
        rng.shuffle(b)
        out += b[:per]
    rng.shuffle(out)
    return out[:k]


def score(ctx: Ctx, cache, sentence: str):
    """Sum log P over the continuation, given the first PREFIX_WORDS words as prefix."""
    from experiment import COMPONENTS, restore, snapshot
    words = sentence.split()
    prefix, cont = " ".join(words[:PREFIX_WORDS]), " " + " ".join(words[PREFIX_WORDS:])
    work = restore(ctx.cache_cls, ctx.config, snapshot(cache), keep=COMPONENTS)
    p_ids, c_ids = ctx.tok.encode(prefix), ctx.tok.encode(cont)
    if not c_ids:
        return None, 0
    seq = torch.tensor([p_ids + c_ids], device=ctx.device)
    with torch.no_grad():
        logits = ctx.model(input_ids=seq, past_key_values=work, use_cache=True).logits
    lp = torch.log_softmax(logits.float(), dim=-1)[0]
    start = len(p_ids) - 1
    return sum(lp[start + i, t].item() for i, t in enumerate(c_ids)), len(c_ids)


def boot(xs, n=20000, seed=5):
    r = random.Random(seed)
    k = len(xs)
    m = sorted(sum(r.choice(xs) for _ in range(k)) / k for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def main():
    ctx = load_real()
    rows, wm_all, seed_means, depths_all = [], [], [], []

    for seed in SEEDS:
        a, b, a2 = clean_docs(seed, TARGET_TOKENS)
        rng = random.Random(1000 + seed)
        probes_a = pick(rng, sentences_with_depth(a.text), PROBES)
        probes_b = pick(rng, sentences_with_depth(b.text), max(20, PROBES // 5))

        caches = {}
        for cond in ("cold", "warm", "mismatched"):
            if cond == "cold":
                caches[cond] = prefill(ctx, b.text, cache=None)
            else:
                src = prefill(ctx, (a if cond == "warm" else a2).text)
                stt = transplant(ctx, src)
                del src
                torch.cuda.empty_cache()
                caches[cond] = prefill(ctx, b.text, cache=stt)

        per_cond = {}
        for cond, cache in caches.items():
            for label, probes in (("A", probes_a), ("B", probes_b)):
                vals = []
                for i, (sent, dep) in enumerate(probes):
                    lp, ntok = score(ctx, cache, sent)
                    if lp is None:
                        continue
                    vals.append(lp / ntok)          # per-token, so length does not dominate
                    rows.append(dict(seed=seed, condition=cond, facts=label, item=i,
                                     depth=round(dep, 4), ntok=ntok,
                                     logprob_per_token=lp / ntok))
                per_cond[(cond, label)] = vals

        wm = [w - m for w, m in zip(per_cond[("warm", "A")], per_cond[("mismatched", "A")])]
        wm_all += wm
        seed_means.append(st.mean(wm))
        depths_all += [d for _, d in probes_a][:len(wm)]
        bd = st.mean(per_cond[("warm", "B")]) - st.mean(per_cond[("cold", "B")])
        print(f"seed {seed}: n={len(wm):3d} probes | warm-mism = {st.mean(wm):+.5f} "
              f"nats/token | B damage = {bd:+.5f}", flush=True)
        for c in caches.values():
            del c
        torch.cuda.empty_cache()

    lo, hi = boot(wm_all)
    t = st.mean(seed_means) / (st.stdev(seed_means) / math.sqrt(len(seed_means)))
    print(f"\n=== |A| = {TARGET_TOKENS} tokens, {len(SEEDS)} seeds, "
          f"{len(wm_all)} probes ===")
    print(f"warm - mismatched : {st.mean(wm_all):+.5f} nats/token   "
          f"95% CI [{lo:+.5f}, {hi:+.5f}]   t({len(seed_means)-1}) = {t:.2f}")
    print(f"                    {sum(x > 0 for x in wm_all)}/{len(wm_all)} probes positive, "
          f"{sum(x > 0 for x in seed_means)}/{len(seed_means)} seeds")

    print("\ndepth profile (warm - mismatched by position in A):")
    for a_, b_ in ((0, 1/3), (1/3, 2/3), (2/3, 1.01)):
        sel = [x for x, d in zip(wm_all, depths_all) if a_ <= d < b_]
        if sel:
            print(f"  depth {a_:.2f}-{b_:.2f}  n={len(sel):4d}  mean = {st.mean(sel):+.5f}")

    with open(f"exp3_per_item_{TARGET_TOKENS}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json.dump(dict(tokens=TARGET_TOKENS, mean=st.mean(wm_all), lo=lo, hi=hi, t=t,
                   n=len(wm_all), seeds=len(SEEDS), seed_means=seed_means),
              open(f"exp3_summary_{TARGET_TOKENS}.json", "w"), indent=2)
    print(f"\nwrote exp3_per_item_{TARGET_TOKENS}.csv ({len(rows)} rows)")
    print("VERDICT:", "transfer detected (CI excludes 0)" if lo > 0 else
          "NOT significant -- CI includes 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
