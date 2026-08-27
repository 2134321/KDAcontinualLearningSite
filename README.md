# KDA State Transplant

**[Results page with charts →](https://2134321.github.io/KDAcontinualLearningSite/)**

> **Active research project.** Results are still being refined and further experiments may
> change the numbers here. Last updated 27 August 2026.

**Does a Kimi Delta Attention recurrent state carry the document it came from, across a full KV
cache reset?**

Yes — measurably, replicably, and faintly. Whether it is *worth doing* is a separate
question, and past a few thousand tokens the answer is no. Both results are below.

```
warm - mismatched     +10.1 accuracy points   95% CI [+6.3, +13.5]     <- document-specific?
                      +0.396 nats             95% CI [+0.326, +0.467]
                      t(23) = 13.53,  24 of 24 seeds positive

warm - cold           +8.0 accuracy points    95% CI [+1.7, +13.9]     <- worth doing?
                      +0.393 nats             95% CI [+0.300, +0.487]
```

Prefill document A. Keep only the 40 MiB KDA recurrent state. Throw away the MLA KV cache and the
short-conv states entirely. Read a different document B. Then ask about facts that were only ever
in A. The model does better than it has any right to — about eight accuracy points over chance on
a two-way forced choice.

That is **not retrieval**. You cannot look up a value; you can detect a bias toward it across
hundreds of trials. It is evidence for a mechanism, not a usable capability.

---

## Why this should be possible at all

[Kimi Linear](https://arxiv.org/abs/2510.26692) interleaves 20 KDA layers (linear attention,
constant-size recurrent state) with 7 full-attention MLA layers in a fixed 3:1 ratio. Per token,
the KDA state update is

```
S_t = S_{t-1}(I - β_t k_t k_tᵀ) + β_t v_t k_tᵀ
```

which is exactly one step of online gradient descent on `½‖S k_t − v_t‖²`, with `β_t` as the
learning rate. The recurrent state is not a cache of past activations — it is a set of **fast
weights trained by gradient descent during the forward pass**
([Irie & Schmidhuber](https://arxiv.org/abs/2508.08435)).

If `S` is weights, then `S` is a checkpoint. This repo tests whether that checkpoint survives being
moved into a fresh run.

The prize, if it works well: context that never stops accumulating, at **constant memory** — the
state is 40 MiB whether A was 1K tokens or 1M — against the ~7.9 GB the MLA KV cache would occupy
at 1M tokens.

---

## The design

Three documents per seed, cut from one shuffle so their stems and values are fully disjoint:

| doc | role | prefilled? | scored? |
|---|---|---|---|
| **A** | source — its state is transplanted | only to produce the state | **yes** |
| **A2** | decoy source, for the mismatched control | only to produce a decoy state | no |
| **B** | the "current" document | in **all** conditions | yes (damage check) |

Three conditions, all scored on A's facts:

```
cold        :                       prefill B  ->  ask about A
mismatched  : prefill A2 -> keep S -> prefill B  ->  ask about A
warm        : prefill A  -> keep S -> prefill B  ->  ask about A
```

**Two questions, two contrasts — do not conflate them.** `warm − cold` is the *benefit*
question: the alternative to a transplant is no transplant, so this is the contrast that says
whether doing it helps at all. `warm − mismatched` is the *document-specificity* question — both
arms carry a real transplanted state of identical size and origin, so only this one can show the
state holds information about *this* document rather than a generic effect of carrying any state.
At 600 tokens the two nearly coincide, because that generic effect is nil (`mismatched − cold` =
−0.003 nats, CI [−0.092, +0.087]). They come apart badly at length — see below.

### Scoring

Two-alternative forced choice on a cloze stem, **not** generation — this is a base model, it
continues documents rather than answering questions:

```
stem     "The northern relay station operates under callsign"
correct  " QR47"     <- appeared in A
wrong    " KT19"     <- appeared nowhere
margin = log P(correct) - log P(wrong)
```

Chance is exactly 0. Both candidates come from the same generator and neither appears in any
document, so the comparison turns entirely on what is in the state.

---

## Reproducing it

**Hardware:** one H200 (141 GB). The 48B bf16 weights are ~92 GB and this deliberately avoids
quantization and tensor-parallel sharding, both of which would alter the object under study.
About $4/hr; a full run is well under an hour after the download.

```bash
python -m venv ~/kda
~/kda/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA wheel
~/kda/bin/pip install 'transformers==4.57.1' einops flash-linear-attention accelerate tiktoken blobfile

# ~92 GB
~/kda/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('moonshotai/Kimi-Linear-48B-A3B-Base')"

~/kda/bin/python replicate.py     # the headline result, 24 seeds
~/kda/bin/python gpu_run.py       # single-seed run + the exactness gate
```

`python experiment.py --tiny` runs the whole pipeline on CPU against a randomly-initialised
4-layer model in about a minute. The numbers are meaningless — it exists to prove the plumbing
before you rent anything.

### Gotchas that will cost you an afternoon

1. **`fla` API skew.** `modeling_kimi.py` calls `fused_kda_gate(g, A_log, head_dim, g_bias=...)`;
   current `flash-linear-attention` exposes `fused_kda_gate(g, A_log, dt_bias=..., ...)`.
   `gpu_run.patch_gate` reimplements the documented formula. Every other kernel stays real Triton.
2. **The model forces `flash_attention_2`** in `KimiLinearModel.__init__` and warns that it ignored
   your config. Override `_attn_implementation` *after* construction.
3. **Two parameters are never initialised** — KDA `dt_bias` and MoE `e_score_correction_bias`
   (its `reset_parameters()` only covers `weight`). Real checkpoints overwrite both, so this is
   invisible at 48B, but a randomly-initialised model inherits uninitialised memory and runs are
   not reproducible. `phase0/exactness_test.py` seeds them.
4. **Candidate values must tokenize to the same length.** `logprob()` sums over answer tokens, so a
   3-token answer carries an extra negative term and loses to a 2-token one regardless of the
   state. Measured on the real Kimi tokenizer, 13 of 24 pairs were mismatched before `_codes()`
   was made tokenizer-aware.
5. **Windows cannot run this.** `triton-windows` and recent CPU torch wheels both fail with
   `WinError 1114`, and `fla` needs Triton to import. Use WSL or Linux.
6. Do not install `fla-core` and `flash-linear-attention` together — they share the `fla`
   namespace and uninstalling either guts the other.

---

## Validating the surgery

The transplant needs no monkey-patching. `KimiDynamicCache` exposes plain Python lists, and
`get_seq_length()` reads `key_cache` only — so a cache holding transplanted `S` with a flushed KV
honestly reports length 0, and run B takes the normal position-0 prefill path while the KDA layers
still receive `S` as `initial_state`.

**The exactness gate is subtle and easy to get wrong.** Comparing a resumed run against an
uninterrupted one *always fails*: splitting a bf16 chunked linear recurrence reorders its
accumulations, moving logits by ~0.1 mean on a ~4.3 scale. That is the model's own numerics, not
the surgery. The correct control isolates them:

```
A = one uninterrupted pass
B = split, keep using the SAME cache object      (no surgery)
C = split, snapshot -> restore into a FRESH cache (surgery)
```

`B` vs `C` must be **bit-identical**. It is (`max|diff| = 0.000e+00`, real Triton kernels, real
48B). All divergence lives in `A` vs `B`. `diag_split.py` runs this.

---

## Layout

| file | what |
|---|---|
| `experiment.py` | corpus generation, prefill, transplant, cloze scoring — the mechanics |
| `replicate.py` | **experiment 1** — 24 seeds, 3 conditions, paired bootstrap CIs; also drives the length ladder via `TARGET_TOKENS` |
| `experiment3.py` | **experiment 3** — sentence completion on unmodified prose, nothing planted |
| `gpu_run.py` | model loading, the exactness gate, single-seed run |
| `diag_split.py` | the split-vs-surgery control |
| `fetch_corpus.py` | the post-cutoff Wikipedia crawler |
| `phase0/` | CPU validation on a tiny random model: save/load surgery + 4 tests |
| `corpus_2026.txt` | the haystack — Wikipedia articles created after 2026-01-01 (CC BY-SA 4.0) |
| `results/per_item.csv` | **experiment 1 raw data** — all 1,728 individual measurements |
| `results/exp2/`, `results/exp3/` | per-item data and run logs for the length ladder and sentence completion |
| `index.html` | the results page, self-contained |

---

## Beyond 600 tokens — the number that matters most

The headline above is a single document length: 600 tokens of synthetic carrier text. Two further
experiments on the [results page](https://2134321.github.io/KDAcontinualLearningSite/) grow
document A on post-cutoff Wikipedia from 600 to 50,000 tokens, and the picture changes.

| \|A\| | `warm − cold` (benefit) | t(7) | `warm − mism` (specificity) | t(7) |
|---|---|---|---|---|
| 600 | +0.038 | +4.77 | +0.054 | +2.20 |
| 1,500 | +0.020 | +1.96 | +0.040 | +2.39 |
| 4,000 | −0.056 | −1.58 | +0.026 | +1.00 |
| 16,000 | −0.125 | −2.58 | +0.018 | +0.54 |
| 50,000 | −0.253 | −4.46 | +0.009 | +1.36 |

nats/token, 8 seeds per length. Document-specificity stays positive at every length tested. The
**benefit does not** — it crosses zero between 1,500 and 4,000 tokens, because the cost of
carrying *any* state (`mismatched − cold`) grows with length while the document-specific signal
thins. At 50,000 tokens the transplant is worse than not doing it: −0.253 nats/token, t = −4.46,
8 of 8 seeds agreeing. That negative is the most solid result in the project.

One more caveat that cuts the other way: every measurement here is **zero-shot**. Nothing in
Kimi Linear's training prepared it for a transplanted state. This is a floor, not a ceiling.

`results/exp2/` and `results/exp3/` hold the raw per-item data behind both.

---

## What this does not show

- **Not retrieval.** 0.566 accuracy against 0.486 chance.
- **"Document-specific" means fact-specific.** A and A2 draw filler from the same 20-sentence pool,
  so ~45% of their prose is shared. That makes the control *tighter* — the effect cannot be generic
  register or vocabulary — but narrows the claim to the planted facts.
- **The depth effect runs backwards, and it is significant.** Facts from the *start* of A —
  the ones the forget gates have had longest to decay — transfer best: +0.613 / +0.283 / +0.293
  across depth thirds. Clustered by seed (early third minus late third, one value per seed):
  +0.320 nats, t(23) = 2.74, p ≈ 0.012, 15/24 seeds. The naive item-level test gives
  r = −0.218, t(286) = −3.77, but treats 12 items sharing a state as independent — do not quote it.
  This is the opposite of what a decay model predicts and this experiment cannot explain it.
- **No Oracle ceiling.** Without running A and B as one uninterrupted sequence, there is no way to
  say what fraction of the available signal the 40 MiB state preserves.
- **B-facts are saturated at 100% accuracy.** Gross corruption is ruled out — the margin was free
  to fall and rose instead — but a subtler cost on harder capabilities would not show here.
- One model, one corpus style, ~470-word documents, synthetic planted facts.

---

## Raw data

`results/per_item.csv` — 1,728 rows, every individual measurement behind every number here:

```
seed, condition, facts, item, depth, stem, correct, wrong, margin
```

24 seeds × 12 facts × 3 conditions × 2 fact sets. Reconstructing the headline from it:

```python
import csv, statistics as st
from collections import defaultdict
M = defaultdict(dict)
for r in csv.DictReader(open("results/per_item.csv")):
    if r["facts"] == "A":
        M[(r["seed"], r["item"])][r["condition"]] = float(r["margin"])
wm = [v["warm"] - v["mismatched"] for v in M.values()]
print(st.mean(wm), sum(x > 0 for x in wm), "/", len(wm))   # +0.3962  220 / 288
```

Everything reported is recomputable from this file without a GPU. The run that produced it
reproduced the published numbers to the last digit — all seeding is deterministic.

## Results

| condition | facts | mean margin (nats) | 95% CI | accuracy |
|---|---|---|---|---|
| cold | A | −0.083 | [−0.228, +0.060] | 0.486 |
| warm | A | **+0.310** | [+0.157, +0.463] | **0.566** |
| mismatched | A | −0.087 | [−0.237, +0.063] | 0.465 |
| cold | B | +17.330 | [17.099, 17.564] | 1.00 |
| warm | B | +18.206 | [17.977, 18.442] | 1.00 |
| mismatched | B | +18.218 | [17.986, 18.451] | 1.00 |

24 seeds × 12 facts = 288 paired observations. CIs bootstrapped over seeds, not items — items
within a seed share a state and a document and are not independent.
