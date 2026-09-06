"""
Alpha sweep at experiment 1's configuration (600-token synthetic carrier, 24 seeds).

THE QUESTION. transplant() carries S across unchanged. This asks whether a uniform scale on
S does better. Two hypotheses pull in opposite directions and this length separates them:

  alpha > 1 amplifies whatever document-specific signal S holds. Experiment 1's length is the
            cleanest place to test that, because the cost of carrying a state is ~0 here
            (mismatched - cold = -0.003 nats, CI [-0.092, +0.087]) -- nothing is fighting it.

  alpha < 1 sheds the cost of carrying a state. That cost only appears at length
            (-0.262 nats at 50K), so this arm is expected to do nothing here. Running it
            anyway is what makes the 50K comparison interpretable later.

HOW TO READ IT. As alpha -> 0 the state vanishes and warm - cold -> 0 by construction. That is
not a result, it is the volume knob. The question is whether any alpha > 0 beats alpha = 0, and
whether the alpha that maximises warm - cold also maximises warm - mismatched. If raising alpha
lifts warm and mismatched equally, the state is not carrying more about *this* document, it is
just louder.

TWO FREE CONTROLS. alpha = 1.0 must reproduce the published numbers to the digit -- it is the
harness check, and on a fresh pod it is also the cross-hardware check. alpha = 0.0 should land
on cold; the preflight verifies that rather than assuming it.

COST. The model is loaded once and A/A2 are prefilled once per seed, then reused across every
alpha. Re-running replicate.py per alpha would reload 92 GB each time and dominate the runtime.

    ALPHAS=0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0 SEEDS=24 python -u alpha_sweep.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402

from experiment import load as load_tiny, make_docs, prefill, logprob  # noqa: E402
from gpu_run import load_real  # noqa: E402
from phase0.state_surgery import snapshot, restore  # noqa: E402

# SEED_START offsets the range so a run can use DIFFERENT documents. Every result so far
# used seeds 0..23, so cross-run agreement proved determinism and not generalisation.
_S0 = int(os.environ.get("SEED_START", 0))
SEEDS = list(range(_S0, _S0 + int(os.environ.get("SEEDS", 24))))
ALPHAS = [float(a) for a in os.environ.get(
    "ALPHAS", "0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0").split(",")]
OUT = os.environ.get("OUT", "alpha")
# TINY=1 runs the whole sweep against the Phase 0 random 4-layer model on CPU.
# The numbers are meaningless; it exists to exercise the loop before renting a GPU.
TINY = bool(os.environ.get("TINY"))
# None keeps the 20-sentence synthetic carrier (experiment 1). An integer switches A and A2
# to real post-cutoff prose of about that many tokens; B stays short either way, so the KV
# is flushed before scoring and probe cost does not grow with |A|.
TARGET_TOKENS = int(os.environ["TARGET_TOKENS"]) if os.environ.get("TARGET_TOKENS") else None


# ----------------------------------------------------------------- hardware identity
def gpu_identity():
    """
    Record which physical card this ran on. A running pod does not migrate, but nothing in
    the earlier ladder logged device identity, so same-hardware was an assumption rather than
    a record. The UUID is stable per card; printing it makes it a grep.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,serial,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or "(nvidia-smi returned nothing)"
    except Exception as e:                                    # noqa: BLE001
        return f"(nvidia-smi unavailable: {e})"


# ----------------------------------------------------------------- statistics
def _betainc(a, b, x):
    """Regularised incomplete beta, continued fraction. Enough for a t-distribution CDF."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _cf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _cf(b, a, 1.0 - x) / b


def _cf(a, b, x, itmax=200, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d, h = 1.0 / d, 1.0 / d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        for num in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                    -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + num * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + num / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def ttest_1samp(xs):
    """Paired t against 0 over SEED means -- items within a seed share a state and a document."""
    n = len(xs)
    if n < 2:
        return 0.0, n - 1, 1.0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    if var <= 0.0:
        return (0.0 if mean == 0 else math.inf), n - 1, (1.0 if mean == 0 else 0.0)
    t = mean / math.sqrt(var / n)
    df = n - 1
    p = _betainc(df / 2.0, 0.5, df / (df + t * t))
    return t, df, p


def summarise(per_seed):
    """per_seed: one mean per seed. Returns the seed-level statistic, which is the honest one."""
    n = len(per_seed)
    mean = sum(per_seed) / n
    t, df, p = ttest_1samp(per_seed)
    return {"mean": mean, "t": t, "df": df, "p": p,
            "seeds_positive": sum(x > 0 for x in per_seed), "seeds": n}


def fmt(s):
    return (f"{s['mean']:+.4f}  t({s['df']}) = {s['t']:+6.2f}  p = {s['p']:.4f}  "
            f"{s['seeds_positive']}/{s['seeds']} seeds")


# ----------------------------------------------------------------- measurement
def margins(ctx, cache, facts):
    return [logprob(ctx, cache, f.stem, f.correct) - logprob(ctx, cache, f.stem, f.wrong)
            for f in facts]


def scaled(ctx, snap, alpha):
    """Rebuild a transplant cache from a stored snapshot, scaling S by alpha."""
    cache = restore(ctx.cache_cls, ctx.config, snap, keep=("recurrent_states",))
    if alpha != 1.0:
        for i, s in enumerate(cache.recurrent_states):
            if s is not None:
                cache.recurrent_states[i] = s * alpha
    return cache


def free():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def preflight(ctx):
    """
    Is alpha=0 the same as cold?

    cold passes cache=None, so the KDA layers get initial_state=None. alpha=0 hands them an
    explicit zero tensor. Phase 0 T4 established None == zeros for conv_states, but never
    checked recurrent_states. If these diverge, the alpha=0 endpoint is not the cold condition
    and the left edge of every curve below is mislabelled.
    """
    a, b, _ = make_docs(tok=ctx.tok, seed=0, n_docs=3,
                        target_tokens=TARGET_TOKENS, short_docs=(1,))
    src = prefill(ctx, a.text)
    snap = snapshot(src)
    del src
    free()

    cold = prefill(ctx, b.text, cache=None)
    m_cold = margins(ctx, cold, a.facts)
    del cold
    free()

    zero = prefill(ctx, b.text, cache=scaled(ctx, snap, 0.0))
    m_zero = margins(ctx, zero, a.facts)
    del zero
    free()

    d = max(abs(c - z) for c, z in zip(m_cold, m_zero))
    verdict = "IDENTICAL" if d == 0.0 else ("within tolerance" if d < 1e-3 else "DIVERGENT")
    print(f"[preflight] alpha=0 vs cold: max|diff| = {d:.3e}  -> {verdict}", flush=True)
    if d >= 1e-3:
        print("[preflight] WARNING: a zero state is not the same as no state. The alpha=0",
              "column below is its own condition, not a second copy of cold.", flush=True)
    return d


def main():
    print(f"[gpu] {gpu_identity()}", flush=True)
    print(f"[cfg] seeds={len(SEEDS)}  alphas={ALPHAS}  "
          f"target_tokens={TARGET_TOKENS or '600 (synthetic carrier)'}", flush=True)

    ctx = load_tiny(True) if TINY else load_real()
    preflight_diff = preflight(ctx)

    # per_seed[alpha][contrast] -> one mean per seed
    per_seed = {a: {k: [] for k in ("wc", "wm", "mc", "warm_acc", "cold_acc", "mism_acc",
                                    "warm_n", "mism_n", "cold_n", "warmB", "mismB", "coldB",
                                    "warm_bacc", "cold_bacc")}
                for a in ALPHAS}
    rows = []

    for seed in SEEDS:
        a, b, a2 = make_docs(tok=ctx.tok, seed=seed, n_docs=3,
                             target_tokens=TARGET_TOKENS, short_docs=(1,))

        # cold does not depend on alpha -- measure it once per seed
        cold = prefill(ctx, b.text, cache=None)
        m_cold = margins(ctx, cold, a.facts)
        mB_cold = margins(ctx, cold, b.facts)
        del cold
        free()

        # prefill A and A2 once; every alpha reuses these 40 MiB snapshots
        src = prefill(ctx, a.text)
        snap_a = snapshot(src)
        del src
        free()
        src = prefill(ctx, a2.text)
        snap_a2 = snapshot(src)
        del src
        free()

        for alpha in ALPHAS:
            warm = prefill(ctx, b.text, cache=scaled(ctx, snap_a, alpha))
            m_warm = margins(ctx, warm, a.facts)
            mB_warm = margins(ctx, warm, b.facts)
            del warm
            free()

            mism = prefill(ctx, b.text, cache=scaled(ctx, snap_a2, alpha))
            m_mism = margins(ctx, mism, a.facts)
            mB_mism = margins(ctx, mism, b.facts)
            del mism
            free()

            k = len(a.facts)
            acc = lambda ms: sum(x > 0 for x in ms) / len(ms)          # noqa: E731
            d = per_seed[alpha]
            d["wc"].append(sum(w - c for w, c in zip(m_warm, m_cold)) / k)
            d["wm"].append(sum(w - m for w, m in zip(m_warm, m_mism)) / k)
            d["mc"].append(sum(m - c for m, c in zip(m_mism, m_cold)) / k)
            d["warm_acc"].append(acc(m_warm) - acc(m_cold))
            d["cold_acc"].append(acc(m_cold))
            d["mism_acc"].append(acc(m_mism) - acc(m_cold))
            d["warm_n"].append(sum(m_warm) / k)
            d["mism_n"].append(sum(m_mism) / k)
            d["cold_n"].append(sum(m_cold) / k)
            d["warmB"].append(sum(mB_warm) / k)
            d["mismB"].append(sum(mB_mism) / k)
            d["coldB"].append(sum(mB_cold) / k)
            d["warm_bacc"].append(acc(mB_warm))
            d["cold_bacc"].append(acc(mB_cold))

            for i, f in enumerate(a.facts):
                rows.append(dict(seed=seed, alpha=alpha, item=i, depth=round(f.depth, 4),
                                 stem=f.stem, correct=f.correct.strip(),
                                 wrong=f.wrong.strip(), cold=m_cold[i],
                                 warm=m_warm[i], mismatched=m_mism[i],
                                 coldB=mB_cold[i], warmB=mB_warm[i], mismB=mB_mism[i]))

            print(f"  seed {seed:2d}  alpha {alpha:<5g}  warm-cold {d['wc'][-1]:+.4f}   "
                  f"warm-mism {d['wm'][-1]:+.4f}   mism-cold {d['mc'][-1]:+.4f}", flush=True)

        del snap_a, snap_a2
        free()

    # ------------------------------------------------------------- report
    out = {"seeds": len(SEEDS), "alphas": ALPHAS, "target_tokens": TARGET_TOKENS,
           "gpu": gpu_identity(),
           "preflight_alpha0_vs_cold_maxdiff": preflight_diff, "by_alpha": {}}

    print("\n=== warm - cold  (the benefit: is transplanting worth doing at all) ===")
    for alpha in ALPHAS:
        print(f"  alpha {alpha:<5g}  {fmt(summarise(per_seed[alpha]['wc']))}")
    print("\n=== warm - mismatched  (document-specificity: is it about THIS document) ===")
    for alpha in ALPHAS:
        print(f"  alpha {alpha:<5g}  {fmt(summarise(per_seed[alpha]['wm']))}")
    print("\n=== mismatched - cold  (the cost of carrying any state) ===")
    for alpha in ALPHAS:
        print(f"  alpha {alpha:<5g}  {fmt(summarise(per_seed[alpha]['mc']))}")
    print("\n=== accuracy on A-facts, points over cold ===")
    for alpha in ALPHAS:
        s = summarise(per_seed[alpha]["warm_acc"])
        m = summarise(per_seed[alpha]["mism_acc"])
        print(f"  alpha {alpha:<5g}  warm {100*s['mean']:+6.2f} pts  t({s['df']}) = {s['t']:+6.2f}"
              f"   |  mismatched {100*m['mean']:+6.2f} pts")
    print("\n=== B-facts mean margin (damage check; cold is the reference) ===")
    for alpha in ALPHAS:
        d = per_seed[alpha]
        n = len(d["warmB"])
        print(f"  alpha {alpha:<5g}  cold {sum(d['coldB'])/n:+8.3f}   "
              f"warm {sum(d['warmB'])/n:+8.3f}   mismatched {sum(d['mismB'])/n:+8.3f}")

    for alpha in ALPHAS:
        d = per_seed[alpha]
        n = len(d["wc"])
        out["by_alpha"][str(alpha)] = {
            "warm_minus_cold": summarise(d["wc"]),
            "warm_minus_mismatched": summarise(d["wm"]),
            "mismatched_minus_cold": summarise(d["mc"]),
            "acc_warm_over_cold": summarise(d["warm_acc"]),
            "acc_mismatched_over_cold": summarise(d["mism_acc"]),
            "cold_acc": sum(d["cold_acc"]) / n,
            "mean_margin": {"cold": sum(d["cold_n"]) / n, "warm": sum(d["warm_n"]) / n,
                            "mismatched": sum(d["mism_n"]) / n},
            "mean_margin_B": {"cold": sum(d["coldB"]) / n, "warm": sum(d["warmB"]) / n,
                              "mismatched": sum(d["mismB"]) / n},
        }

    # ------------------------------------------------------------- cost-aware view
    # warm - cold alone is not the objective anyone would optimise: it ignores what carrying
    # the state does to B, the document actually in context. Points per nat of B damage makes
    # the trade explicit. Reported alongside the benefit-only verdict, not instead of it.
    print("\n=== benefit against cost ===")
    print(f"  {'alpha':>6} {'A acc gain':>11} {'B damage':>10} {'B acc':>8} "
          f"{'specificity':>13} {'pts/nat':>9}")
    best_ratio, best_a = -1e9, None
    for a in ALPHAS:
        d = per_seed[a]
        n = len(d["wc"])
        gain = 100.0 * sum(d["warm_acc"]) / n                      # points over cold
        bdam = sum(w - c for w, c in zip(d["warmB"], d["coldB"])) / n
        bacc = sum(d["warm_bacc"]) / n if "warm_bacc" in d else float("nan")
        spec = summarise(d["wm"])
        ratio = gain / abs(bdam) if bdam else float("inf")
        if gain > 0 and ratio > best_ratio:
            best_ratio, best_a = ratio, a
        print(f"  {a:>6g} {gain:>+10.2f} {bdam:>+10.4f} {bacc:>8.4f} "
              f"{spec['mean']:>+8.4f} t={spec['t']:>+5.2f} {ratio:>9.1f}")
    if best_a is not None:
        out["best_alpha_by_benefit_per_cost"] = best_a
        print(f"  best benefit-per-damage: alpha = {best_a:g} at {best_ratio:.1f} "
              f"accuracy points per nat of B damage")

    # ------------------------------------------------------------- verdict
    # Every alpha is measured on the SAME seeds and documents, so alpha-vs-alpha is paired:
    # per-seed difference first, then a t across seeds. Comparing two independent summaries
    # would throw away that pairing and badly understate the power.
    def paired_vs(a, ref=1.0):
        d = [x - y for x, y in zip(per_seed[a]["wc"], per_seed[ref]["wc"])]
        return summarise(d)

    if 1.0 in ALPHAS:
        print("\n=== warm - cold at each alpha, PAIRED against alpha = 1.0 ===")
        for a in ALPHAS:
            if a == 1.0:
                continue
            print(f"  alpha {a:<5g} - alpha 1.0   {fmt(paired_vs(a))}")
            out["by_alpha"][str(a)]["paired_vs_alpha1"] = paired_vs(a)

    best = max(ALPHAS, key=lambda a: summarise(per_seed[a]["wc"])["mean"])
    out["best_alpha_by_warm_minus_cold"] = best
    means = {a: summarise(per_seed[a]["wc"])["mean"] for a in ALPHAS}
    print(f"\nhighest warm - cold on this grid: alpha = {best:g}  ({means[best]:+.4f})")

    # argmax over a grid is optimistically biased -- the winner is picked partly on noise.
    # Bonferroni over the alphas actually compared keeps the claim honest.
    ncmp = max(1, len([a for a in ALPHAS if a != 1.0]))
    thresh = 0.05 / ncmp
    if best == 1.0:
        print("VERDICT: no alpha on this grid beats leaving the state alone. A uniform scale "
              "buys nothing at this length.")
    elif 1.0 in ALPHAS:
        st = paired_vs(best)
        print(f"         vs alpha=1.0: {fmt(st)}")
        print(f"         Bonferroni threshold for {ncmp} comparisons: p < {thresh:.4f}")
        if st["p"] < thresh:
            print(f"VERDICT: alpha={best:g} beats the untouched transplant and survives "
                  "correction. A uniform scale is a live knob; a learned map has room to work.")
        else:
            print(f"VERDICT: alpha={best:g} is the grid maximum but does NOT separate from "
                  "alpha=1.0 after correction. Read this as no effect, not a small one -- "
                  "the argmax of a noisy grid is above the mean by construction.")
    if 0.0 in ALPHAS and means[best] <= means[0.0]:
        print("         NOTE: no alpha beats switching the state off entirely. On this "
              "evidence the best thing to do with the state is not use it.")

    with open(f"{OUT}_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(f"{OUT}_per_item.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}_summary.json and {OUT}_per_item.csv ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
