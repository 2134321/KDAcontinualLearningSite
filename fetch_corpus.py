"""
Build a carrier corpus the model provably has not been trained on.

Kimi-Linear was released 2025-10. Anything written in 2026 is therefore new to it. This pulls
Wikipedia articles CREATED in 2026 -- not merely about 2026 -- via the recentchanges API, so the
prose itself is post-cutoff regardless of subject.

Why it matters: the original carrier was 20 sentences on a loop. At 100K tokens each would repeat
~3,750 times, and that repetition would dominate the recurrent state far more than any planted
fact. A real corpus also lets A and A2 be drawn from disjoint text, which removes the ~45% shared
prose that limited the earlier result to "fact-specific" rather than "document-specific".

Output: corpus_2026.txt, one paragraph per line.
Licence: Wikipedia text is CC BY-SA 4.0. corpus_2026_sources.txt records every article used.

    python fetch_corpus.py --words 400000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://en.wikipedia.org/w/api.php"
UA = "KDA-continual-learning-research/1.0 (https://github.com/2134321/KDAcontinualLearningSite)"

# 2026-01-01. Articles created after this are necessarily unseen by an Oct-2025 model.
CUTOFF = "2026-01-01T00:00:00Z"


_LAST = [0.0]
THROTTLE = 1.1          # seconds between requests; Wikipedia 429s well below this rate


def api(**params) -> dict:
    """
    One API call, politely. Wikipedia rate-limits hard: an earlier version fired ~270 requests
    back to back and died on HTTP 429 with retries exhausted. Fixed rate limiting plus honouring
    Retry-After is what makes a long crawl finish.
    """
    params.setdefault("format", "json")
    params.setdefault("maxlag", 5)          # back off when the cluster is busy, as the API asks
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(8):
        wait = THROTTLE - (time.time() - _LAST[0])
        if wait > 0:
            time.sleep(wait)
        _LAST[0] = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if attempt == 7:
                raise
            delay = int(e.headers.get("Retry-After", 0) or 0) or min(60, 5 * 2 ** attempt)
            print(f"    HTTP {e.code}, sleeping {delay}s (attempt {attempt+1}/8)", flush=True)
            time.sleep(delay)
        except Exception as e:
            if attempt == 7:
                raise
            print(f"    {type(e).__name__}, retry {attempt+1}/8", flush=True)
            time.sleep(min(60, 3 * 2 ** attempt))
    return {}


def new_page_titles(limit_total: int):
    """Titles of articles created since CUTOFF, newest first, paged."""
    seen, cont = [], None
    while len(seen) < limit_total:
        p = dict(action="query", list="recentchanges", rcnamespace=0, rctype="new",
                 rclimit=500, rcprop="title|timestamp", rcend=CUTOFF, rcdir="older")
        if cont:
            p["rccontinue"] = cont
        d = api(**p)
        rows = d.get("query", {}).get("recentchanges", [])
        if not rows:
            break
        seen += [r["title"] for r in rows]
        cont = d.get("continue", {}).get("rccontinue")
        if not cont:
            break
        print(f"  collected {len(seen)} candidate titles", flush=True)
    return seen[:limit_total]


def extracts(titles: list[str], min_words: int = 60):
    """Plain-text article bodies, 20 titles per request."""
    for i in range(0, len(titles), 20):
        d = api(action="query", prop="extracts", explaintext=1, exlimit=20,
                titles="|".join(titles[i:i + 20]))
        for page in d.get("query", {}).get("pages", {}).values():
            txt = page.get("extract") or ""
            if len(txt.split()) >= min_words:      # skip stubs
                yield page["title"], txt


def clean(text: str) -> list[str]:
    """Drop headings, references, and short fragments. Keep real paragraphs."""
    out = []
    for para in text.split("\n"):
        para = para.strip()
        if not para or para.startswith("=") or len(para.split()) < 25:
            continue
        para = re.sub(r"\s+", " ", para)
        para = re.sub(r"\[\d+\]", "", para)
        out.append(para)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=400_000,
                    help="target word count; 100K tokens is roughly 75K words per document")
    ap.add_argument("--out", default="corpus_2026.txt")
    ap.add_argument("--candidates", type=int, default=30_000,
                    help="new-page titles to consider. Yield is low -- most newly created "
                         "articles are stubs -- so this needs to be ~50x the article count.")
    ap.add_argument("--min-words", type=int, default=60,
                    help="skip articles shorter than this")
    ap.add_argument("--resume", action="store_true",
                    help="keep what is already on disk and skip articles already fetched")
    args = ap.parse_args()

    print(f"[1/2] finding articles created after {CUTOFF}")
    titles = new_page_titles(limit_total=args.candidates)
    print(f"  {len(titles)} candidates")

    # APPEND AS WE GO. An earlier version accumulated everything in memory and wrote once at the
    # end, so any crash -- a dropped connection outliving the retries, a laptop sleeping -- threw
    # away the whole crawl. Flushing per article means a killed run keeps everything it had, and
    # --resume picks up from there instead of starting over.
    src_path = args.out.replace(".txt", "_sources.txt")
    done: set[str] = set()
    if args.resume and os.path.exists(src_path):
        with open(src_path, encoding="utf-8") as f:
            done = {ln.strip() for ln in f if ln.strip() and not ln.startswith(("Wikipedia", "#"))}
        with open(args.out, encoding="utf-8") as f:
            words = sum(len(ln.split()) for ln in f)
        print(f"  resuming: {len(done)} articles, {words:,} words already on disk")
    else:
        words = 0
        open(args.out, "w", encoding="utf-8").close()
        with open(src_path, "w", encoding="utf-8") as f:
            f.write("Wikipedia articles created after 2026-01-01. Text is CC BY-SA 4.0.\n")

    print(f"[2/2] fetching text, target {args.words:,} words")
    todo = [t for t in titles if t not in done]
    n_new = 0
    for title, txt in extracts(todo, args.min_words):
        ps = clean(txt)
        if not ps:
            continue
        with open(args.out, "a", encoding="utf-8") as f:
            f.write("\n".join(ps) + "\n")
        with open(src_path, "a", encoding="utf-8") as f:
            f.write(title + "\n")
        n_new += 1
        words += sum(len(p.split()) for p in ps)
        if n_new % 100 == 0:
            print(f"  {words:,} words from {len(done) + n_new} articles", flush=True)
        if words >= args.words:
            break
    print(f"\nwrote {args.out}: {words:,} words, {len(done) + n_new} articles "
          f"({n_new} new this run)")
    print(f"enough for ~{words // 75000} documents of 100K tokens, "
          f"~{words // 37500} of 50K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
