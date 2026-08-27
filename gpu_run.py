"""
Phase 1 on the real 48B, single model load.

  Step 1  EXACTNESS on the real model with the REAL Triton kernels. Phase 0 passed only
          through CPU shims, so nothing below is trustworthy until this does.
  Step 2  The forced-choice experiment: warm (S transplanted from A) vs cold.

Run on the pod:  python gpu_run.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
# run_condition lives in experiment.py only -- it used to be duplicated here, which is how
# the first real run silently used a different copy from the one in the repo.
from experiment import Ctx, make_docs, run_condition  # noqa: E402
from phase0.state_surgery import COMPONENTS, describe, restore, snapshot, state_nbytes  # noqa: E402

REPO = "moonshotai/Kimi-Linear-48B-A3B-Base"
# eager materialises the full N x N attention matrix: fine at 555 tokens, OOM by 256K. sdpa
# computes the same function with memory-efficient kernels and handles this model's
# q_head_dim 192 / v_head_dim 128 mismatch. flash_attn is not installed on the pod.
ATTN_IMPL = os.environ.get("ATTN_IMPL", "sdpa")


def patch_gate(mod):
    """
    Version skew, unchanged from Phase 0 and NOT a CPU-only problem: modeling_kimi.py calls
        fused_kda_gate(g, A_log, head_dim, g_bias=dt_bias)
    while installed fla exposes
        fused_kda_gate(g, A_log, dt_bias=None, lower_bound=None, output_dtype=...)
    Reimplement the documented formula. It is a small elementwise op, so losing the fused
    kernel costs nothing measurable. Every other kernel stays real.
    """
    import torch.nn.functional as F

    def gate(g, A_log, head_dim, g_bias=None, lower_bound=None,
             output_dtype=torch.float32, **kw):
        g = g.view(*g.shape[:-1], -1, head_dim).float()
        if g_bias is not None:
            g = g + g_bias.view(g.shape[-2], g.shape[-1]).float()
        A = A_log.float().exp()
        if A.dim() == 1:
            A = A.unsqueeze(-1)
        return (-A * F.softplus(g)).to(output_dtype)

    mod.fused_kda_gate = gate
    print("  patched fused_kda_gate (API skew); all other kernels are real Triton")


def load_real(dtype=torch.bfloat16) -> Ctx:
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    print("[load] tokenizer + config")
    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    config = AutoConfig.from_pretrained(REPO, trust_remote_code=True)

    model_cls = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], REPO)
    modeling = sys.modules[model_cls.__module__]
    patch_gate(modeling)
    cache_cls = modeling.KimiDynamicCache

    print("[load] weights (~96 GB) -> cuda")
    model = model_cls.from_pretrained(REPO, dtype=dtype, device_map="cuda",
                                      trust_remote_code=True).eval()
    # __init__ forces flash_attention_2 and warns it ignored ours; flash-attn is not installed.
    model.config._attn_implementation = ATTN_IMPL
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = ATTN_IMPL

    print(f"  loaded. cuda mem: {torch.cuda.memory_allocated()/2**30:.1f} GiB")
    return Ctx(model, tok, config, cache_cls, device="cuda")


def exactness(ctx: Ctx, text: str) -> bool:
    """
    Gate on the SURGERY, not on splitting the sequence.

    Splitting a bf16 chunked linear recurrence reorders its accumulations, which moves logits
    by ~0.1 (mean) on a ~4.3 logit scale all by itself -- measured, and nothing to do with this
    project. Comparing a resumed run against an uninterrupted one therefore charges the model's
    own numerics to us and always "fails".

    The right control isolates the surgery:
        B = split, keep using the SAME cache object   (no surgery)
        C = split, snapshot -> restore into a FRESH cache
    B vs C must be bit-identical. A vs B is reported as context only.
    """
    ids = torch.tensor([ctx.tok.encode(text)], device=ctx.device)
    n = ids.shape[1]
    cut = 64 * (n // 128)
    print(f"\n[T1] surgery exactness  ({n} tokens, split at {cut})")

    with torch.no_grad():
        A = ctx.model(input_ids=ids, use_cache=False).logits[:, cut:, :].float()

        cache_b = ctx.cache_cls(ctx.config)
        ctx.model(input_ids=ids[:, :cut], past_key_values=cache_b, use_cache=True)
        B = ctx.model(input_ids=ids[:, cut:], past_key_values=cache_b,
                      use_cache=True).logits.float()

        cache_c = ctx.cache_cls(ctx.config)
        ctx.model(input_ids=ids[:, :cut], past_key_values=cache_c, use_cache=True)
        snap = snapshot(cache_c)
        C = ctx.model(input_ids=ids[:, cut:], use_cache=True,
                      past_key_values=restore(ctx.cache_cls, ctx.config, snap,
                                              keep=COMPONENTS)).logits.float()

    print(describe(snap, "after prefix"))
    print(f"  KDA state: {state_nbytes(snap)/2**20:.1f} MiB (fp32, not bf16)")

    ok = torch.equal(B, C)
    ab = (A - B).abs()
    print(f"  context  A vs B (splitting only): mean|diff| = {ab.mean().item():.3e}  "
          f"argmax agree = {(A.argmax(-1) == B.argmax(-1)).float().mean().item():.4f}")
    print(f"  GATE     B vs C (surgery only)  : max|diff| = "
          f"{(B - C).abs().max().item():.3e}  bit-identical = {ok}  "
          f"-> {'PASS' if ok else 'FAIL'}")

    # T2 / T4, cheap and worth re-confirming against real kernels
    warm = restore(ctx.cache_cls, ctx.config, snap, keep=("recurrent_states",))
    print(f"[T2] S-only cache get_seq_length() = {warm.get_seq_length()} "
          f"-> {'PASS' if warm.get_seq_length() == 0 else 'FAIL'}")

    snap_zero = dict(snap)
    snap_zero["conv_states"] = [None if c is None else tuple(torch.zeros_like(t) for t in c)
                               for c in snap["conv_states"]]
    with torch.no_grad():
        a = ctx.model(input_ids=ids[:, cut:], use_cache=True,
                      past_key_values=restore(ctx.cache_cls, ctx.config, snap,
                                              keep=("recurrent_states",))).logits.float()
        b = ctx.model(input_ids=ids[:, cut:], use_cache=True,
                      past_key_values=restore(ctx.cache_cls, ctx.config, snap_zero,
                                              keep=("recurrent_states", "conv_states"))).logits.float()
    d4 = (a - b).abs().max().item()
    print(f"[T4] conv None vs zeros: max|diff| = {d4:.3e} "
          f"-> {'PASS' if d4 < 1e-2 else 'FAIL'}")
    return ok


def main():
    ctx = load_real()
    doc_a, doc_b = make_docs(tok=ctx.tok)
    print(f"\ncorpus A: {len(doc_a.text.split())} words / {len(doc_a.facts)} facts"
          f"   B: {len(doc_b.text.split())} words / {len(doc_b.facts)} facts")

    if not exactness(ctx, doc_a.text):
        print("\nEXACTNESS FAILED -- stopping. Nothing below would mean anything.")
        return 1

    print("\n[experiment] warm vs cold")
    warm = run_condition(ctx, doc_a, doc_b, warm=True)
    cold = run_condition(ctx, doc_a, doc_b, warm=False)

    print(f"\n{'':10s} {'facts':>6s} {'cold acc':>9s} {'warm acc':>9s} "
          f"{'cold margin':>13s} {'warm margin':>13s} {'delta':>12s}")
    for label in ("A", "B"):
        d = warm[label]["margin"] - cold[label]["margin"]
        print(f"{'about ' + label:10s} {warm[label]['n']:6d} "
              f"{cold[label]['acc']:9.2f} {warm[label]['acc']:9.2f} "
              f"{cold[label]['margin']:13.6f} {warm[label]['margin']:13.6f} {d:+12.3e}")

    print("\n  A-facts by depth in A (0.0 = start, most decayed):")
    buckets = {"0.00-0.33": [], "0.33-0.67": [], "0.67-1.00": []}
    for dep, w_, c_ in zip(warm["A"]["depths"], warm["A"]["margins"], cold["A"]["margins"]):
        k = "0.00-0.33" if dep < 1 / 3 else ("0.33-0.67" if dep < 2 / 3 else "0.67-1.00")
        buckets[k].append(w_ - c_)
    for k, v in buckets.items():
        if v:
            print(f"    depth {k}  n={len(v):2d}  mean delta = {sum(v)/len(v):+.4f}")

    hits = sum(m > 0 for m in warm["A"]["margins"])
    print(f"\n  WARM on A-facts: {hits}/{warm['A']['n']} correct "
          f"(chance = {warm['A']['n']/2:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
