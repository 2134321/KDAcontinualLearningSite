# Phase 0 — state surgery validation

**Status: ALL PASS, 2026-08-23.** Validated on a randomly-initialised 0.26M-param KimiLinear
(4 layers, 3 KDA + 1 MLA), CPU only, no weights downloaded.

```
T1 exactness   max|Δlogits|      = 7.451e-07   (tol 1e-4)   PASS
T2 seqlen      get_seq_length()  = 0 with S loaded          PASS
T3 influence   max|cold - warm|  = 5.622e-02                PASS
T4 conv form   max|none - zeros| = 0.000e+00                PASS
```

Reproducible: identical to the digit across repeat runs (see "Uninitialised parameters" below —
they were not, at first).

T1 says the save/load surgery is faithful. T2 confirms the library fact the whole design rests
on. T3 confirms a transplanted state is actually consumed rather than silently ignored.
T4 settles a specific claim: **leaving `conv_states` as None is bit-identical to installing
explicit zero conv tensors** — 0.000e+00, not merely within tolerance, and it held across three
different random weight draws before the harness was made deterministic.

### On the claim that S is ignored unless conv is non-None

It is false. `modeling_kimi.py` lines 458-464:

```
458:        conv_state_q, conv_state_k, conv_state_v = None, None, None
459:        recurrent_state = None
460:        if cache_params is not None:
461:            if cache_params.conv_states[self.layer_idx] is not None:
462:                conv_state_q, conv_state_k, conv_state_v = cache_params.conv_states[
463:                    self.layer_idx]
464:            recurrent_state = cache_params.recurrent_states[self.layer_idx]
```

Line 464 is at 12 spaces — inside `if cache_params is not None`, *not* inside the conv branch
(which is 16, per line 462). The conv read is guarded; the recurrent read is not. T3 and T4
confirm it empirically. Passing zeros instead of None is harmless but unnecessary.

## What the cache actually looks like

Dumped after a 64-token prefix. 0-indexed layers; the config's `kda_layers` / `full_attn_layers`
are **1-indexed**, so config `[1,2,3]` / `[4]` → runtime `[0,1,2]` / `[3]`:

```
recurrent_states  filled=3/4  layers=[0,1,2]  shape (1, 4, 16, 16)   = [B, H, K, V]
conv_states       filled=3/4  layers=[0,1,2]  3-tuple of (1, 64, 4)  = (q,k,v), kernel 4
key_cache         filled=1/4  layers=[3]      shape (1, 4, 64, 24)
value_cache       filled=1/4  layers=[3]      shape (1, 4, 64, 16)
```

State size scales as predicted: 12.0 KiB here (3×4×16×16×4B), and
20 layers × 32 heads × 128 × 128 × 2 B = **20 MiB** on the real model at bf16, constant in
context length. Measured on the real 48B it is **40 MiB**, because the states are held in fp32.

## Environment

**Windows is a dead end.** Both a fresh CPU `torch` wheel and `triton-windows` fail with
`WinError 1114` DLL init errors, and `fla` cannot be imported without Triton. Use WSL.

```bash
wsl -d Ubuntu
python3 -m venv ~/kda
~/kda/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/kda/bin/pip install 'transformers==4.57.1' einops flash-linear-attention
cd /path/to/KDAcontinualLearningSite/phase0 && ~/kda/bin/python exactness_test.py
```

Pin `transformers==4.57.1` to match `config.json`. Installing `flash-linear-attention` and
`fla-core` together corrupts the shared `fla` namespace — pick one (`flash-linear-attention`;
`fla-core` has no `fla.layers`). Triton installs fine on Linux and warns
"Triton is not supported on current platform, roll back to CPU", which is expected — we never
launch a kernel.

## CPU shims — and why T1 is still valid

Five things had to be replaced to run without a GPU driver:

| Shim | Why |
|---|---|
| `chunk_kda`, `fused_recurrent_kda` → `fla.ops.kda.naive.*` | Triton kernels |
| `fused_kda_gate` → reimplemented | **API version skew, see below** |
| `ShortConvolution.forward` → torch depthwise causal conv | only triton/cuda backends exist |
| `FusedRMSNormGated.forward` → torch | Triton kernel |
| `_attn_implementation = "eager"`, **set after construction** | `KimiLinearModel.__init__` overwrites it with `flash_attention_2` and warns that it ignored ours |

**T1 remains a valid check** because the reference pass and the resumed pass run the identical
shimmed code — it is a self-consistency test of the cache surgery, not of the kernels. But
fidelity to the real numerics is not established here, so **Phase 1 must re-run T1 on GPU
against the real Triton kernels before any measurement is trusted**. `diag_split.py` does
this; it passes bit-identical.

## Findings that affect Phase 1

1. **`fla` API skew is real and will recur on the H200.** `modeling_kimi.py` calls the old
   signature `fused_kda_gate(g, A_log, head_dim, g_bias=dt_bias)`; current `fla` exposes
   `fused_kda_gate(g, A_log, dt_bias=None, lower_bound=None, output_dtype=...)`. Either pin an
   `fla` version contemporary with the checkpoint, or carry the shim forward. Budget time for
   this — it is not a CPU-only problem.
2. **`A_log` is stored as `[1, 1, H, 1]`**, not `[H]`. Do not add an axis when broadcasting.
3. **`naive_chunk_kda` asserts `T % 64 == 0`.** Only affects the CPU reference path, but it
   constrains test sequence lengths.
4. **Mode is chosen by `q_len` alone**: `'fused_recurrent' if q_len <= 64 else self.mode`. The
   reference run here (T=128) took the chunk path while both resumed runs (T=64) took the
   recurrent path — and they still agreed to 6e-07, which incidentally cross-validates the two
   kernels against each other.
5. Real token ids (~163k) and `padding_idx` blow past a tiny vocab; override
   `pad/bos/eos_token_id` when shrinking the config.
6. **Uninitialised parameters — this one bites.** `modeling_kimi.py` allocates two parameters
   with `torch.empty()` and never initialises them: KDA `dt_bias` (~line 418) and MoE
   `e_score_correction_bias` (~line 560 — `reset_parameters()` only covers `weight`). Real
   checkpoints overwrite both, so this is invisible at 48B. A randomly-initialised model
   inherits **uninitialised memory**, making runs non-reproducible — and because the router
   bias changes which experts fire, T3 was drifting ~5% run to run. `build()` now initialises
   both from a seeded generator. Anyone else building a tiny KimiLinear will hit this.

## Model welfare

Phase 0 ran without restriction, deliberately. A randomly-initialised 0.26M-parameter model has
no training and no learned organisation, so nothing here warrants the caution applied to work on
the trained 48B. The distinction was drawn before running, not after.
