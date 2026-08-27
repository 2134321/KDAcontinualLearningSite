"""
The whole experiment, one file.

    corpus A --prefill--> cache_A --keep KDA S, drop KV+conv--+
                                                              |--prefill B--> ask questions
    (nothing) -------------------------------------------------+   (cold control)

Then compare WARM vs COLD on two sets of questions:

    questions about A  -> does the transplanted state carry document-specific facts?
    questions about B  -> does carrying it DAMAGE processing of the current document?

The second set is the one people forget. If WARM loses to COLD on B-facts, the state is
hurting current-context processing (the NoPE position-collision failure mode) and any win on
A-facts has to be weighed against it.

Scoring is two-alternative forced choice on a cloze stem, NOT generation:

    stem    "The vault passcode is"
    correct " 7391"      <- appeared in A
    wrong   " 2048"      <- did not

Why: the model is a BASE model. It continues documents, it does not answer questions.
And forced choice is self-controlling -- both options are equally plausible English in an
identical context, so chance is exactly 0.5 and no perplexity baseline is needed to read
the result.

Run the pipeline end-to-end on CPU in ~1 min with a random tiny model:
    python experiment.py --tiny
Real run (needs the H200):
    python experiment.py
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field

import torch

from phase0.state_surgery import COMPONENTS, restore, snapshot

REPO = "moonshotai/Kimi-Linear-48B-A3B-Base"


# ============================================================================
# 1. DATA  -- synthetic corpus, generated. See make_docs().
# ============================================================================

@dataclass
class Fact:
    """One forced-choice probe."""
    stem: str        # cloze prefix, no trailing space
    correct: str     # the value that appeared in the document
    wrong: str       # a same-shape value that did not
    depth: float = 0.0   # 0.0 = start of the document (most decayed), 1.0 = end (freshest)


@dataclass
class Doc:
    text: str
    facts: list[Fact] = field(default_factory=list)


# The retrieval cues, written out. Subject, verb phrase and slot noun all differ between
# entries, with no shared template.
#
# THIS MATTERS: the delta rule updates S as
#     S_t = S_{t-1}(I - b_t k_t k_t^T) + b_t v_t k_t^T
# and that (I - b k k^T) term ERASES the component of S along k before writing. Similar keys
# therefore overwrite each other, so a set of facts sharing one template would measure mutual
# interference rather than transfer. (--uniform-keys runs exactly that, on purpose.)
#
# A stem is used verbatim twice: appended with its value to build the document sentence, and
# again in run B as the query. Add more by writing them here; 2 * n_facts are needed.
STEMS = [
    "The northern relay station operates under callsign",
    "Feld's field notebook carries catalogue reference",
    "The winter supply manifest lists lot number",
    "The survey team's theodolite bears serial number",
    "The harbour pilot's licence shows registration code",
    "The sealed crate in the annex is marked with inventory tag",
    "The meteorological buoy broadcasts on transponder code",
    "Ansel's transit permit was issued as document number",
    "The eastern pumping station runs on circuit label",
    "The archive microfilm reel is filed under shelf mark",
    "The customs declaration quotes reference code",
    "The observatory's spare mirror was ordered as part number",
    "The freight siding at Kelder appears on the plan as track",
    "The medical supply locker was closed with seal number",
    "The radio operator's logbook continues in volume",
    "The glacier survey stake is recorded as marker",
    "The auxiliary generator was tagged asset number",
    "The cartography office safe opens with combination",
    "The shipment of core samples travelled under consignment code",
    "The lighthouse relief roster follows schedule",
    "The diesel storage tank is stencilled vessel number",
    "The seismograph at Ridge Camp reports as instrument",
    "The border checkpoint ledger opens at entry code",
    "The salvage tender was registered with hull number",
    "The botanical specimen case holds accession number",
    "The telegraph repeater hut answers to station code",
    "The ice core drilling rig was delivered as unit",
    "The quartermaster's tally sheet uses form number",
    "The coastal watchtower is designated post",
    "The expedition's spare sledge carries equipment tag",
    "The assay laboratory furnace is a model",
    "The mail packet from Vlissen came in pouch number",
    "The reserve fuel depot is listed at site code",
    "The hydrographic chart set reached edition",
    "The signal lamp housing takes fitting code",
    "The winter quarters roof panel came from batch",
]

# Neutral in-distribution filler. Facts are planted INTO this, so the document reads as prose
# rather than as a list of key-value pairs -- the forget gates were trained on language, and a
# bare list is badly off-distribution.
CARRIER = [
    "The expedition reached the northern station in late autumn.",
    "Snow had already closed the upper pass by the time the second party arrived.",
    "Provisions were counted twice, once on arrival and once at the end of the week.",
    "The generator ran for three weeks without maintenance.",
    "Correspondence from the mainland arrived irregularly and often out of order.",
    "Two of the surveyors spent the morning repairing a damaged sledge runner.",
    "The wind dropped shortly after dawn and the sky cleared for the first time in days.",
    "A dispute over the watch rota was settled without much difficulty.",
    "Ice conditions in the sound were noted as unusually severe for the season.",
    "The cook improvised a passable bread from the remaining flour.",
    "Repairs to the outer door took longer than anyone had expected.",
    "Readings were taken every six hours and entered into the standing log.",
    "One of the dogs went lame and was rested for the remainder of the week.",
    "The stove smoked badly until the flue was cleared.",
    "Visibility fell to almost nothing during the afternoon.",
    "A relief party was expected but did not appear before the month ended.",
    "The men spent the evening mending harness and sorting stores.",
    "Temperatures held steady well below freezing for eleven consecutive days.",
    "Some of the older tins were found to have spoiled and were discarded.",
    "The last of the lamp oil was rationed from the middle of the month.",
]


def _codes(rng: random.Random, n: int, tok=None) -> list[str]:
    """
    Arbitrary unguessable values: two letters + two digits. Correct and wrong come from the
    SAME generator so neither is a priori more plausible -- the forced choice then turns
    entirely on what is in the state.

    If `tok` is given, keep only values that tokenize to the SAME number of tokens. This is
    not cosmetic: logprob() sums over answer tokens, so a 3-token answer carries an extra
    negative term and loses to a 2-token one regardless of the state. Measured on the real
    Kimi tokenizer, 13 of 24 pairs were mismatched before this filter.
    """
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    seen, pool = set(), []
    while len(pool) < max(n * 20, 200):
        v = f"{rng.choice(letters)}{rng.choice(letters)}{rng.randint(10, 99)}"
        if v not in seen:
            seen.add(v)
            pool.append(v)

    if tok is None:
        return pool[:n]

    by_len: dict[int, list[str]] = {}
    for v in pool:
        by_len.setdefault(len(tok.encode(f" {v}")), []).append(v)
    best = max(by_len.values(), key=len)
    if len(best) < n:
        raise ValueError(f"only {len(best)} values share a token length; need {n}")
    return best[:n]


def _assemble(carrier: list[str], sentences: list[str], depths: list[float]) -> str:
    """Insert each fact sentence at its target depth through the carrier prose."""
    out = list(carrier)
    # insert from the back so earlier indices stay valid
    for sent, d in sorted(zip(sentences, depths), key=lambda p: -p[1]):
        out.insert(min(int(round(d * len(carrier))), len(carrier)), sent)
    return " ".join(out)


# --------------------------------------------------------------------------
# Real carrier corpus, for documents too long to build from the CARRIER list
# --------------------------------------------------------------------------

_CORPUS: list[str] | None = None


def load_corpus(path: str = "corpus_2026.txt") -> list[str]:
    """
    Paragraphs of prose written after the model's training cutoff (see fetch_corpus.py).

    The CARRIER list above is 20 sentences. Filling a 100K-token document from it would repeat
    each sentence ~3,750 times, and that repetition would dominate the recurrent state far more
    than any planted fact. Long documents need real text.
    """
    global _CORPUS
    if _CORPUS is None:
        with open(path, encoding="utf-8") as f:
            _CORPUS = [ln.strip() for ln in f if len(ln.split()) >= 25]
        if not _CORPUS:
            raise ValueError(f"{path} has no usable paragraphs")
    return _CORPUS


def _carrier_from_corpus(rng: random.Random, target_words: int, used: set[int],
                         corpus: list[str]) -> list[str]:
    """
    Draw paragraphs until target_words is reached, never reusing one already taken by another
    document in this seed. `used` is mutated -- that is what keeps A, B and A2 disjoint in prose,
    which the 20-sentence CARRIER could not do (it left ~45% of A's sentences also present in A2,
    narrowing the result to fact-specific rather than document-specific transfer).
    """
    idx = [i for i in range(len(corpus)) if i not in used]
    rng.shuffle(idx)
    out, words = [], 0
    for i in idx:
        used.add(i)
        out.append(corpus[i])
        words += len(corpus[i].split())
        if words >= target_words:
            return out
    raise ValueError(f"corpus exhausted: needed {target_words} words, got {words}. "
                     f"Fetch more with fetch_corpus.py --words")


def make_docs(n_facts: int = 12, uniform_keys: bool = False, seed: int = 0,
              carrier_len: int = 30, tok=None, n_docs: int = 2,
              target_tokens: int | None = None, corpus_path: str = "corpus_2026.txt",
              short_docs: tuple[int, ...] = ()) -> tuple[Doc, ...]:
    """
    Build corpus A and corpus B.

    n_facts       facts planted in EACH document, at evenly spaced depths
    uniform_keys  False -> distinct semantic keys (the main result)
                  True  -> every fact shares one stem, "The answer to question N is".
                           Maximally similar keys, so this measures the delta rule's mutual
                           erasure. It is the worst-case floor of the capacity sweep,
                           not a bug -- but never report it as the headline number.
    carrier_len   filler sentences per document; controls how far apart facts sit, and how
                  much decay a depth-0 fact has to survive

    Depth is the point. The forget gates decay continuously, so a fact at depth 0.0 has the
    whole document decaying on top of it while depth 1.0 is fresh. Scoring by depth turns this
    single run into the decay curve (M2).

    NOTE the wrong values appear in NEITHER document. If a wrong value appeared in B, the warm
    condition would have just seen it in recent context and the test would be biased against
    the correct answer.
    """
    if not uniform_keys and n_docs * n_facts > len(STEMS):
        raise ValueError(f"need {n_docs * n_facts} stems, STEMS has {len(STEMS)}")
    rng = random.Random(seed)

    # All documents are cut from ONE shuffle and ONE code pool, so every document's stems and
    # values are disjoint from every other's. This matters for the mismatched-S control: if the
    # mismatch document reused a stem with a different value, that item would silently become
    # a "same key, different value" collision -- a much stronger disruption -- and the
    # control would blend two different effects.
    codes = _codes(rng, 2 * n_docs * n_facts, tok)
    stems = list(STEMS)
    rng.shuffle(stems)
    depths = [i / max(n_facts - 1, 1) for i in range(n_facts)]

    # target_tokens switches the carrier from the 20-sentence CARRIER list to real post-cutoff
    # prose. Roughly 0.75 words per token for English. short_docs names document indices that
    # stay short regardless -- B is the "current document" and does not need to be long.
    corpus = load_corpus(corpus_path) if target_tokens else None
    used: set[int] = set()

    def build(d):
        tag = "ABC"[d] if d < 3 else str(d)
        stemset = stems[d * n_facts:(d + 1) * n_facts]
        correct = codes[d * n_facts:(d + 1) * n_facts]
        wrong = codes[(n_docs + d) * n_facts:(n_docs + d + 1) * n_facts]
        facts, sentences = [], []
        for i, (val, bad) in enumerate(zip(correct, wrong)):
            stem = f"The answer to question {tag}{i + 1} is" if uniform_keys else stemset[i]
            sentences.append(f"{stem} {val}.")
            facts.append(Fact(stem=stem, correct=f" {val}", wrong=f" {bad}", depth=depths[i]))
        if corpus is not None and d not in short_docs:
            carrier = _carrier_from_corpus(rng, int(target_tokens * 0.75), used, corpus)
        else:
            carrier = [rng.choice(CARRIER) for _ in range(carrier_len)]
        return Doc(text=_assemble(carrier, sentences, depths), facts=facts)

    return tuple(build(d) for d in range(n_docs))


# ============================================================================
# 2. MODEL
# ============================================================================

@dataclass
class Ctx:
    """Everything the ops below need, bundled so signatures stay short."""
    model: object
    tok: object
    config: object
    cache_cls: type
    device: str = "cpu"


class _FakeTok:
    """STUB tokenizer for --tiny: hashes words to ids so the pipeline is exercisable."""

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [abs(hash(w)) % self.vocab_size for w in text.split()] or [0]


def load(tiny: bool) -> Ctx:
    if tiny:
        # Reuse the Phase 0 harness: random weights, 4 layers, CPU, no download.
        # Lets you watch the pipeline run before spending H200 hours.
        from phase0.exactness_test import build
        model, config, cache_cls = build()
        return Ctx(model, _FakeTok(config.vocab_size), config, cache_cls)

    import sys

    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    config = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    model_cls = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], REPO)
    cache_cls = sys.modules[model_cls.__module__].KimiDynamicCache
    del tok, cache_cls  # referenced by the stub below

    # STUB: real load. ~96 GB of weights -> single H200.
    #   model = model_cls.from_pretrained(REPO, dtype=torch.bfloat16, device_map="cuda",
    #                                     trust_remote_code=True).eval()
    #   return Ctx(model, tok, config, cache_cls, device="cuda")
    # Expect the `fla` version skew here: modeling_kimi.py calls the old fused_kda_gate
    # signature. See phase0/README.md before enabling.
    raise NotImplementedError("real 48B load -- see phase0/README.md")


def ids(ctx: Ctx, text: str) -> torch.Tensor:
    return torch.tensor([ctx.tok.encode(text)], device=ctx.device)


# ============================================================================
# 3. CORE OPS -- the experiment is these three functions
# ============================================================================

def prefill(ctx: Ctx, text: str, cache=None):
    """Run `text` through the model, accumulating into `cache` (fresh one if None)."""
    if cache is None:
        cache = ctx.cache_cls(ctx.config)
    with torch.no_grad():
        ctx.model(input_ids=ids(ctx, text), past_key_values=cache, use_cache=True)
    return cache


def transplant(ctx: Ctx, cache):
    """
    THE INTERVENTION. Keep the KDA recurrent states, drop everything else.

    The new cache's get_seq_length() reads key_cache, which is now empty, so it reports 0
    and run B proceeds from position 0 -- while the KDA layers still receive S as
    initial_state. That is the whole trick (verified by Phase 0 T2).
    """
    return restore(ctx.cache_cls, ctx.config, snapshot(cache), keep=("recurrent_states",))


def logprob(ctx: Ctx, cache, stem: str, answer: str) -> float:
    """
    Sum logP(answer | stem), continuing from `cache`. The cache is NOT mutated -- we score
    many facts against the same state, so each scoring pass gets a private copy. Forgetting
    this silently contaminates every result after the first.
    """
    work = restore(ctx.cache_cls, ctx.config, snapshot(cache), keep=COMPONENTS)

    stem_ids, ans_ids = ctx.tok.encode(stem), ctx.tok.encode(answer)
    seq = torch.tensor([stem_ids + ans_ids], device=ctx.device)

    with torch.no_grad():
        logits = ctx.model(input_ids=seq, past_key_values=work, use_cache=True).logits

    # logits[t] predicts token t+1, so answer token i is predicted at len(stem)-1+i
    lp = torch.log_softmax(logits.float(), dim=-1)[0]
    start = len(stem_ids) - 1
    return sum(lp[start + i, tid].item() for i, tid in enumerate(ans_ids))


# ============================================================================
# 4. CONDITIONS
# ============================================================================

def run_condition(ctx: Ctx, doc_a: Doc, doc_b: Doc, warm: bool) -> dict:
    """
    warm=True : prefill A -> transplant S -> prefill B -> ask
    warm=False: (control)                    prefill B -> ask
    """
    cache = transplant(ctx, prefill(ctx, doc_a.text)) if warm else None
    cache = prefill(ctx, doc_b.text, cache=cache)

    out = {}
    for label, doc in (("A", doc_a), ("B", doc_b)):
        margins = [
            logprob(ctx, cache, f.stem, f.correct) - logprob(ctx, cache, f.stem, f.wrong)
            for f in doc.facts
        ]
        n = max(len(margins), 1)
        out[label] = {
            "n": len(margins),
            "acc": sum(m > 0 for m in margins) / n,
            "margin": sum(margins) / n,
            "margins": margins,
            "depths": [f.depth for f in doc.facts],
        }
    return out


# ============================================================================
# 5. MAIN
# ============================================================================

HOW_TO_READ = """
How to read this:
  about A   cold should sit at chance (acc 0.50, margin 0.0). warm above chance = the
            transplanted state carries document-specific facts. THE RESULT.
  about B   warm should MATCH cold. warm below cold = the state is damaging current-context
            processing, and any A-gain has to be weighed against that loss.

  Accuracy needs a binomial test. At the default n=12 per document:
      12/12 -> p = 0.0002    11/12 -> p = 0.003    10/12 -> p = 0.019    9/12 -> p = 0.073
  So 12 is the smallest n that still says something after a couple of misses.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-facts", type=int, default=12)
    ap.add_argument("--carrier-len", type=int, default=30)
    ap.add_argument("--uniform-keys", action="store_true",
                    help="all facts share one stem -- the delta-rule interference floor, "
                         "not the headline result")
    ap.add_argument("--tiny", action="store_true",
                    help="random 4-layer model on CPU -- exercises the pipeline, "
                         "results are meaningless")
    args = ap.parse_args()

    ctx = load(args.tiny)
    doc_a, doc_b = make_docs(n_facts=args.n_facts, carrier_len=args.carrier_len,
                             uniform_keys=args.uniform_keys, tok=ctx.tok)
    print(f"  corpus A: {len(doc_a.text.split())} words, {len(doc_a.facts)} facts"
          f"   corpus B: {len(doc_b.text.split())} words, {len(doc_b.facts)} facts"
          f"   keys={'uniform' if args.uniform_keys else 'distinct'}")

    warm = run_condition(ctx, doc_a, doc_b, warm=True)
    cold = run_condition(ctx, doc_a, doc_b, warm=False)

    # Report the warm-cold delta explicitly. It is the quantity the experiment is about,
    # and at --tiny scale the raw margins agree to 3 decimals while the delta is ~1e-3 --
    # round them and a real effect reads as no effect.
    print(f"\n{'':10s} {'facts':>6s} {'cold acc':>9s} {'warm acc':>9s} "
          f"{'cold margin':>13s} {'warm margin':>13s} {'delta':>12s}")
    for label in ("A", "B"):
        d = warm[label]["margin"] - cold[label]["margin"]
        print(f"{'about ' + label:10s} {warm[label]['n']:6d} "
              f"{cold[label]['acc']:9.2f} {warm[label]['acc']:9.2f} "
              f"{cold[label]['margin']:13.6f} {warm[label]['margin']:13.6f} "
              f"{d:+12.3e}")
    # Free M2: the same runs give the decay curve, because facts sit at known depths in A.
    da, dw, dc = warm["A"]["depths"], warm["A"]["margins"], cold["A"]["margins"]
    if len(da) >= 3:
        print()
        print("  A-facts by depth in A (0.0 = start, most decayed):")
        buckets = {"0.00-0.33": [], "0.33-0.67": [], "0.67-1.00": []}
        for d, w_, c_ in zip(da, dw, dc):
            key = "0.00-0.33" if d < 1 / 3 else ("0.33-0.67" if d < 2 / 3 else "0.67-1.00")
            buckets[key].append(w_ - c_)
        for k, v in buckets.items():
            if v:
                print(f"    depth {k}  n={len(v):2d}  mean delta = {sum(v)/len(v):+.3e}")
    print(HOW_TO_READ)


if __name__ == "__main__":
    main()
