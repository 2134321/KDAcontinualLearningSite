"""
The control that separates "splitting the forward pass" from "my snapshot/restore".

  A  single uninterrupted pass                       (reference)
  B  prefill[0:cut], then continue with the SAME cache object -- no surgery at all
  C  prefill[0:cut], snapshot, restore into a FRESH cache, then continue -- surgery

If C == B exactly, the surgery is lossless and every bit of divergence from A is caused by
splitting the sequence (bf16 chunked-recurrence reordering), which is a property of the model,
not of this project. That is the result that lets Phase 1 proceed.

If C != B, the surgery is losing or corrupting something and the whole design is in trouble.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment import make_docs  # noqa: E402
from gpu_run import load_real  # noqa: E402
from phase0.state_surgery import COMPONENTS, restore, snapshot  # noqa: E402


def main():
    ctx = load_real()
    doc_a, _ = make_docs(tok=ctx.tok)
    ids = torch.tensor([ctx.tok.encode(doc_a.text)], device=ctx.device)
    n = ids.shape[1]
    cut = 256
    print(f"\nlen={n}  cut={cut}  suffix={n-cut}")

    with torch.no_grad():
        A = ctx.model(input_ids=ids, use_cache=False).logits[:, cut:, :].float()

        # B: split, no surgery -- keep using the same cache object
        cache_b = ctx.cache_cls(ctx.config)
        ctx.model(input_ids=ids[:, :cut], past_key_values=cache_b, use_cache=True)
        B = ctx.model(input_ids=ids[:, cut:], past_key_values=cache_b,
                      use_cache=True).logits.float()

        # C: split, WITH snapshot -> restore into a fresh cache
        cache_c = ctx.cache_cls(ctx.config)
        ctx.model(input_ids=ids[:, :cut], past_key_values=cache_c, use_cache=True)
        restored = restore(ctx.cache_cls, ctx.config, snapshot(cache_c), keep=COMPONENTS)
        C = ctx.model(input_ids=ids[:, cut:], past_key_values=restored,
                      use_cache=True).logits.float()

    def cmp(x, y, label):
        d = (x - y).abs()
        agree = (x.argmax(-1) == y.argmax(-1)).float().mean().item()
        print(f"{label:34s} max|diff| = {d.max().item():.3e}   "
              f"mean|diff| = {d.mean().item():.3e}   argmax agree = {agree:.4f}")

    cmp(A, B, "A vs B  (splitting only)")
    cmp(B, C, "B vs C  (SURGERY ONLY)  <-- key")
    cmp(A, C, "A vs C  (splitting + surgery)")

    same = torch.equal(B, C)
    print(f"\nB and C bit-identical: {same}")
    print("VERDICT:", "surgery is lossless; divergence is bf16 split reordering"
          if same else "SURGERY IS NOT LOSSLESS -- investigate before proceeding")


if __name__ == "__main__":
    raise SystemExit(main())
