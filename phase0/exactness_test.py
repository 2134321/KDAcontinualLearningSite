"""
Phase 0 -- validate the KDA state surgery on a tiny random KimiLinear model. CPU, no GPU,
no 96 GB download.

Three tests, in order of what they buy:

  T1 EXACTNESS   Snapshot *everything* (S + KV + conv) after a prefix, restore into a fresh
                 cache, run the suffix. Logits must match an uninterrupted single pass to fp
                 tolerance. Random weights are fine -- this is an equality check, not a
                 quality check. If T1 fails the surgery is wrong and nothing downstream means
                 anything.

  T2 SEQLEN      A cache holding transplanted S with a flushed KV must report get_seq_length()
                 == 0, so run B takes the position-0 prefill path. This is the single library
                 fact the whole design rests on.

  T3 INFLUENCE   The S-only transplant must actually change the logits vs a cold run.
                 Guards against the state being silently accepted and ignored.

  T4 CONV FORM   Leaving conv_states as None must be equivalent to installing explicit
                 zero conv tensors. Both are "flush the short conv"; if they diverge, the
                 transplant is sensitive to which one we pick and the design has to say
                 so. Cheap insurance against a silent, plausible-looking bug.

Kernel note: modeling_kimi.py calls fla's Triton `chunk_kda`, which needs CUDA. On CPU we
patch in `fla.ops.kda.naive.naive_chunk_kda` instead. That is faithful for T1-T3 because the
object under test is the *cache plumbing*, not the kernel -- but it does mean Phase 1 must
re-run T1 on GPU against the real kernels before any result is trusted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from state_surgery import COMPONENTS, describe, restore, snapshot, state_nbytes  # noqa: E402

REPO = "moonshotai/Kimi-Linear-48B-A3B-Base"

# Tiny mirror of the real architecture: 3:1 KDA:full, 1-indexed layer ids, short conv 4.
TINY = dict(
    num_hidden_layers=4,
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=16,
    kv_lora_rank=16,
    q_lora_rank=None,
    qk_nope_head_dim=16,
    qk_rope_head_dim=8,
    v_head_dim=16,
    mla_use_nope=True,
    vocab_size=512,
    # real ids are ~163k and blow past the tiny vocab
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    num_experts=4,
    num_experts_per_token=2,
    num_shared_experts=1,
    moe_intermediate_size=32,
    first_k_dense_replace=1,
    linear_attn_config={
        "kda_layers": [1, 2, 3],
        "full_attn_layers": [4],
        "num_heads": 4,
        "head_dim": 16,
        "short_conv_kernel_size": 4,
    },
)


def patch_kernels_for_cpu(mod) -> bool:
    """Swap fla's Triton KDA kernels for the pure-PyTorch reference. Returns True if patched."""
    try:
        from fla.ops.kda.naive import naive_chunk_kda, naive_recurrent_kda
    except ImportError as e:
        print(f"  ! cannot import fla naive kernels: {e}")
        return False

    def _shim(inner, fallback=None):
        def f(q, k, v, g, beta, scale=None, initial_state=None,
              output_final_state=False, use_qk_l2norm_in_kernel=False,
              cu_seqlens=None, **kw):
            if cu_seqlens is not None:
                raise NotImplementedError("varlen not supported by the CPU shim")
            if use_qk_l2norm_in_kernel:
                q = torch.nn.functional.normalize(q, p=2, dim=-1)
                k = torch.nn.functional.normalize(k, p=2, dim=-1)
            fn = inner
            # naive_chunk_kda asserts T % 64 == 0; the real Triton kernel has no such limit.
            # Same recurrence either way, so fall back rather than pad.
            if fallback is not None and q.shape[1] % 64 != 0:
                fn = fallback
            return fn(q, k, v, g, beta, scale=scale, initial_state=initial_state,
                      output_final_state=output_final_state)
        return f

    patched = []
    for name, inner, fb in (("chunk_kda", naive_chunk_kda, naive_recurrent_kda),
                            ("fused_recurrent_kda", naive_recurrent_kda, None)):
        if hasattr(mod, name):
            setattr(mod, name, _shim(inner, fb))
            patched.append(name)

    # Version skew: modeling_kimi.py (pinned to transformers 4.57.1) calls the OLD gate API
    #   fused_kda_gate(g, A_log, head_dim, g_bias=dt_bias)
    # while current fla exposes
    #   fused_kda_gate(g, A_log, dt_bias=None, lower_bound=None, output_dtype=...)
    # Reimplement the documented formula:
    #   g = -A_log.exp().unsqueeze(-1) * softplus(g + dt_bias.view(g.shape[-2:]))
    if hasattr(mod, "fused_kda_gate"):
        import torch.nn.functional as F

        def gate(g, A_log, head_dim, g_bias=None, lower_bound=None,
                 output_dtype=torch.float32, **kw):
            g = g.view(*g.shape[:-1], -1, head_dim).float()      # [..., H, K]
            if g_bias is not None:
                g = g + g_bias.view(g.shape[-2], g.shape[-1]).float()
            # A_log is stored as [1, 1, H, 1] already -- do not add an axis
            A = A_log.float().exp()
            if A.dim() == 1:
                A = A.unsqueeze(-1)
            g = -A * F.softplus(g)
            return g.to(output_dtype)

        mod.fused_kda_gate = gate
        patched.append("fused_kda_gate")
    print(f"  patched CPU kernels: {patched or 'NONE (names not found -- read modeling_kimi.py)'}")
    return bool(patched)


def patch_short_conv_for_cpu() -> bool:
    """
    fla's ShortConvolution offers only 'triton' and 'cuda' backends -- both need a GPU driver.
    Replace forward() with a pure-PyTorch depthwise causal conv.

    ShortConvolution subclasses nn.Conv1d with groups=hidden_size, padding=W-1. The cache is
    [N, D, W] holding the previous W inputs, so concatenating it in front and taking the last
    T outputs reproduces the streaming semantics exactly.
    """
    try:
        from fla.modules.conv.short_conv import ShortConvolution
    except ImportError as e:
        print(f"  ! cannot import ShortConvolution: {e}")
        return False

    import torch.nn.functional as F

    def forward(self, x, residual=None, mask=None, cache=None,
                output_final_state=False, cu_seqlens=None, chunk_indices=None, **kw):
        if cu_seqlens is not None:
            raise NotImplementedError("varlen not supported by the CPU shim")
        B, T, _ = x.shape
        W = self.kernel_size[0]
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        xt = x.transpose(1, 2)                                   # [B, D, T]
        inp = torch.cat([cache, xt], dim=-1) if cache is not None \
            else F.pad(xt, (W - 1, 0))
        y = F.conv1d(inp, self.weight, self.bias, groups=self.groups)[..., -T:]

        if self.activation is not None:                          # silu / swish
            y = F.silu(y)
        y = y.transpose(1, 2)
        if residual is not None:
            y = y + residual

        new_cache = None
        if output_final_state:
            new_cache = inp[..., -W:].detach().clone()
            if cache is not None:
                cache.copy_(new_cache)                           # real impl updates in place
                new_cache = cache
        return y, new_cache

    ShortConvolution.forward = forward
    print("  patched CPU kernels: ShortConvolution.forward (pure torch)")
    return True


def patch_gated_norm_for_cpu() -> bool:
    """
    fla's FusedRMSNormGated is a Triton kernel. Replace with plain torch:
        y = rmsnorm(x) * weight * act(gate),  act in {sigmoid, swish/silu}
    Called as `o = self.o_norm(o, g)` in modeling_kimi.py.
    """
    try:
        from fla.modules.fused_norm_gate import FusedRMSNormGated
    except ImportError as e:
        print(f"  ! cannot import FusedRMSNormGated: {e}")
        return False

    import torch.nn.functional as F

    def forward(self, x, g=None, residual=None, prenorm=False,
                residual_in_fp32=False, **kw):
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        if g is not None:
            gf = g.float()
            y = y * (torch.sigmoid(gf) if self.activation == "sigmoid" else F.silu(gf))
        return y.to(dtype)

    FusedRMSNormGated.forward = forward
    print("  patched CPU kernels: FusedRMSNormGated.forward (pure torch)")
    return True


def build(dtype=torch.float32):
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    print("[build] loading remote code (small .py files, not weights)")
    config = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    for k, val in TINY.items():
        setattr(config, k, val)
    config.torch_dtype = dtype
    config.dtype = dtype
    # the 7 MLA layers default to flash-attention-2, which is not installed on CPU
    config._attn_implementation = "eager"

    # Materialise the remote model class, then patch kernels in ITS module namespace --
    # modeling_kimi.py imports chunk_kda by name, so patching fla itself would be too late.
    model_cls = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], REPO)
    modeling = sys.modules[model_cls.__module__]
    if not patch_kernels_for_cpu(modeling):
        raise SystemExit("kernel patch failed -- cannot run on CPU")
    if not patch_short_conv_for_cpu():
        raise SystemExit("short-conv patch failed -- cannot run on CPU")
    if not patch_gated_norm_for_cpu():
        raise SystemExit("gated-norm patch failed -- cannot run on CPU")

    cache_cls = modeling.KimiDynamicCache

    print("[build] instantiating tiny random model on CPU")
    torch.manual_seed(0)
    model = model_cls(config).to(dtype).eval()
    # KimiLinearModel.__init__ overwrites _attn_implementation with flash_attention_2 and
    # warns that it ignored ours -- put it back now that construction is done.
    model.config._attn_implementation = "eager"
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"
    # modeling_kimi.py allocates two parameters with torch.empty() and never initialises them:
    #   KDA  dt_bias                  (line ~418)
    #   MoE  e_score_correction_bias  (line ~560; reset_parameters() only covers `weight`)
    # Real checkpoints overwrite both. A randomly-initialised model inherits uninitialised
    # memory, so runs are NOT reproducible -- and the router bias changes which experts fire,
    # which was moving T3 by ~5% between runs. Initialise them deterministically.
    gen = torch.Generator().manual_seed(1234)
    fixed = []
    with torch.no_grad():
        for name, mod_ in model.named_modules():
            for attr, lo, hi in (("dt_bias", -1.0, 1.0),
                                 ("e_score_correction_bias", -0.02, 0.02)):
                par = getattr(mod_, attr, None)
                if isinstance(par, torch.nn.Parameter):
                    par.copy_(torch.rand(par.shape, generator=gen) * (hi - lo) + lo)
                    fixed.append(f"{name}.{attr}")
    print(f"  initialised {len(fixed)} uninitialised params "
          f"({sum('dt_bias' in f for f in fixed)} dt_bias, "
          f"{sum('e_score' in f for f in fixed)} router bias)")

    n = sum(p.numel() for p in model.parameters())
    print(f"  params: {n/1e6:.2f}M  dtype={dtype}  layers={config.num_hidden_layers}")
    return model, config, cache_cls


def run(model, ids, cache=None):
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=cache, use_cache=cache is not None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=int, default=64)   # multiples of 64: naive_chunk_kda asserts T % 64 == 0
    ap.add_argument("--suffix", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    model, config, cache_cls = build()

    torch.manual_seed(1)
    n = args.prefix + args.suffix
    ids = torch.randint(0, config.vocab_size, (1, n))
    pre, suf = ids[:, :args.prefix], ids[:, args.prefix:]

    # ---- reference: one uninterrupted pass -------------------------------------------
    print("\n[T1] exactness: full snapshot -> restore -> resume")
    ref = run(model, ids).logits[:, args.prefix:, :]

    # ---- prefix, snapshot everything, resume ------------------------------------------
    cache = cache_cls(config)
    run(model, pre, cache)
    snap = snapshot(cache)
    print(describe(snap, "after prefix"))
    print(f"  recurrent_states size: {state_nbytes(snap)/1024:.1f} KiB "
          f"(real model: ~20 MiB, constant in context length)")

    resumed_cache = restore(cache_cls, config, snap, keep=COMPONENTS)
    got = run(model, suf, resumed_cache).logits

    diff = (ref - got).abs().max().item()
    t1 = diff < args.tol
    print(f"  max|Δlogits| = {diff:.3e}   tol = {args.tol:.0e}   -> {'PASS' if t1 else 'FAIL'}")

    # ---- T2: S-only cache must report length 0 ----------------------------------------
    print("\n[T2] S-only transplant reports get_seq_length() == 0")
    warm = restore(cache_cls, config, snap, keep=("recurrent_states",))
    print(describe(warm, "S-only cache"))
    seqlen = warm.get_seq_length()
    t2 = seqlen == 0
    print(f"  get_seq_length() = {seqlen}  -> {'PASS' if t2 else 'FAIL'}")

    # ---- T3: the transplanted state must actually do something ------------------------
    print("\n[T3] transplanted S changes the output vs a cold run")
    cold = run(model, suf, cache_cls(config)).logits
    warm_logits = run(model, suf, restore(cache_cls, config, snap,
                                          keep=("recurrent_states",))).logits
    delta = (cold - warm_logits).abs().max().item()
    t3 = delta > 1e-6
    print(f"  max|cold - warm| = {delta:.3e}  -> {'PASS' if t3 else 'FAIL (state ignored!)'}")

    # ---- T4: conv=None must be equivalent to an explicit zero conv --------------------
    print()
    print("[T4] S-only with conv=None == S-only with explicit zero conv")
    l_none = run(model, suf, restore(cache_cls, config, snap,
                                     keep=("recurrent_states",))).logits
    snap_zero = dict(snap)
    snap_zero["conv_states"] = [
        None if c is None else tuple(torch.zeros_like(t) for t in c)
        for c in snap["conv_states"]
    ]
    warm_zero = restore(cache_cls, config, snap_zero,
                        keep=("recurrent_states", "conv_states"))
    nz = sum(1 for c in snap_zero["conv_states"] if c is not None)
    print(f"  zero conv tensors installed on {nz} layers")
    l_zero = run(model, suf, warm_zero).logits
    d4 = (l_none - l_zero).abs().max().item()
    t4 = d4 < args.tol
    print(f"  max|none - zeros| = {d4:.3e}   tol = {args.tol:.0e}   "
          f"-> {'PASS (equivalent)' if t4 else 'FAIL (NOT equivalent -- use zeros)'}")

    ok = t1 and t2 and t3 and t4
    print("\n" + ("ALL PASS -- surgery validated, Phase 1 can proceed"
                  if ok else "FAILURE -- do not proceed to Phase 1"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
