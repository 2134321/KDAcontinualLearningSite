"""
Multi-seed replication with paired statistics.

Per seed, three documents cut from one shuffle so stems and values are disjoint:
    A   the source document -- its state is what we transplant
    B   the current document -- prefilled in every condition
    A2  an unrelated document -- the mismatched-S control

Three conditions, all scored on A's facts:
    cold        prefill B                                  no state at all
    warm        prefill B on top of S from A               the treatment
    mismatched  prefill B on top of S from A2              generic-state control

The headline is the PAIRED per-item difference warm - mismatched. Both conditions carry a
transplanted state of the same size and kind; only the source document differs. Anything that
survives that comparison is document-specific.

cold is reported too, but on its own it is not interpretable: the two candidate values are not
equally likely a priori (measured cold accuracy 0.83, not 0.50), so only differences cancel
that per-item prior.
"""

from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402

from experiment import make_docs, prefill, transplant, logprob  # noqa: E402
from gpu_run import load_real  # noqa: E402

SEEDS = list(range(int(os.environ.get("SEEDS", 24))))

# None keeps the short 20-sentence carrier (the published result). An integer switches document
# A and A2 to real post-cutoff prose of about that many tokens; B stays short either way.
# One-shot prefill only -- segmenting A would put ~12 bf16 chunk boundaries in the long condition
# and none in the short one, confounding length with segmentation.
TARGET_TOKENS = int(os.environ["TARGET_TOKENS"]) if os.environ.get("TARGET_TOKENS") else None


def margins(ctx, cache, facts):
    return [logprob(ctx, cache, f.stem, f.correct) - logprob(ctx, cache, f.stem, f.wrong)
            for f in facts]


def boot_ci(xs, n=10000, seed=0):
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(rng.choice(xs) for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ctx = load_real()
    wm, wc, mc, depths = [], [], [], []
    # raw per-item margins, keyed (condition, which facts) -- the 6 bars
    raw = {(c, f): [] for c in ("cold", "warm", "mismatched") for f in ("A", "B")}

    rows = []   # every individual measurement, written to per_item.csv

    for seed in SEEDS:
        a, b, a2 = make_docs(tok=ctx.tok, seed=seed, n_docs=3,
                             target_tokens=TARGET_TOKENS, short_docs=(1,))

        # ONE CONDITION AT A TIME. Holding all three caches at once costs 3x the KV, which is
        # fine at 600 tokens and 43 GB at 100K. Each source cache is dropped as soon as its
        # 40 MiB state has been lifted out of it.
        def build(cond):
            if cond == "cold":
                return prefill(ctx, b.text, cache=None)
            src = prefill(ctx, (a if cond == "warm" else a2).text)
            state = transplant(ctx, src)
            del src
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return prefill(ctx, b.text, cache=state)

        for cond in ("cold", "warm", "mismatched"):
            cache = build(cond)
            for label, doc in (("A", a), ("B", b)):
                ms = margins(ctx, cache, doc.facts)
                raw[(cond, label)] += ms
                for i, (f, m) in enumerate(zip(doc.facts, ms)):
                    rows.append(dict(seed=seed, condition=cond, facts=label, item=i,
                                     depth=round(f.depth, 4), stem=f.stem,
                                     correct=f.correct.strip(), wrong=f.wrong.strip(),
                                     margin=m))
            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        k = len(a.facts)
        m_cold, m_warm, m_mism = (raw[(c, "A")][-k:] for c in ("cold", "warm", "mismatched"))
        wm += [w - m for w, m in zip(m_warm, m_mism)]
        wc += [w - c for w, c in zip(m_warm, m_cold)]
        mc += [m - c for m, c in zip(m_mism, m_cold)]
        depths += [f.depth for f in a.facts]

        acc = lambda ms: sum(x > 0 for x in ms) / len(ms)  # noqa: E731
        print(f"seed {seed}: acc cold={acc(m_cold):.2f} warm={acc(m_warm):.2f} "
              f"mism={acc(m_mism):.2f} | mean warm-mism = {sum(wm[-k:])/k:+.4f}", flush=True)

    print("\n=== the six bars: mean margin (nats), 95% bootstrap CI ===")
    bars = {}
    for (cond, label), xs in raw.items():
        lo, hi = boot_ci(xs)
        acc = sum(x > 0 for x in xs) / len(xs)
        bars[f"{cond}|{label}"] = {"mean": sum(xs) / len(xs), "lo": lo, "hi": hi,
                                   "acc": acc, "n": len(xs)}
        print(f"  {cond:11s} on {label}-facts  mean = {sum(xs)/len(xs):+8.4f}  "
              f"CI [{lo:+.4f}, {hi:+.4f}]  acc = {acc:.2f}")
    import json
    with open("bars.json", "w") as f:
        json.dump(bars, f, indent=2)
    with open("per_item.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote bars.json and per_item.csv ({len(rows)} rows)")

    n = len(wm)
    print(f"\n=== {len(SEEDS)} seeds x 12 facts = {n} paired observations ===")
    for name, xs in (("warm - mismatched  (HEADLINE)", wm),
                     ("warm - cold", wc),
                     ("mismatched - cold", mc)):
        lo, hi = boot_ci(xs)
        pos = sum(x > 0 for x in xs)
        print(f"{name:32s} mean = {sum(xs)/n:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
              f"{pos}/{n} positive")

    print("\ndepth profile of warm - mismatched:")
    for lo_d, hi_d in ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)):
        sel = [x for x, d in zip(wm, depths) if lo_d <= d < hi_d]
        if sel:
            print(f"  depth {lo_d:.2f}-{hi_d:.2f}  n={len(sel):3d}  "
                  f"mean = {sum(sel)/len(sel):+.4f}")

    lo, hi = boot_ci(wm)
    print("\nVERDICT:", "document-specific transfer (CI excludes 0)" if lo > 0 else
          "NOT significant -- CI includes 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
