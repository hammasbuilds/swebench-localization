"""Phase 2: score BM25, embeddings and an LLM as file localizers on SWE-bench Lite.

Phase 1 asked whether the issue text names the file to fix. This asks the follow-up:
when it does not, can anything find it anyway?

Every retriever sees the same candidate set - the repository's source files - and returns
a ranked list. Scoring is recall@k, cut by the Phase 1 discoverability tiers, because an
aggregate number cannot distinguish a retriever that reads from one that string-matches.

Resumable by construction: each retriever appends one JSON line per instance and skips
what is already on disk. An interrupted run loses at most one instance.

    python run_phase2.py                 # everything
    python run_phase2.py --only bm25     # one retriever
    python run_phase2.py --limit 20      # smoke test
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data import load
from evaluate import fabrication_rate, print_report, resolve, summarize
from mention_analysis import tier_for
from retrieval import (
    BM25,
    cosine_rank,
    embed,
    llm_locate,
    llm_rerank,
    ollama_ready,
)
from trees import (
    cached_repos,
    commit_listings,
    listing_for,
    source_files,
)

OUT = Path(__file__).resolve().parent / "data" / "phase2"
EMB = Path(__file__).resolve().parent / "data" / "embeddings"
TOP = 20  # deepest k reported; ranking beyond this is not scored
POOL = 30  # candidates handed to the reranker


def _done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written line from a killed run
            rows[r["instance_id"]] = r
    return rows


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


class Corpora:
    """Candidate files per instance, plus the indexes built over them.

    Most instances share their repository's listing, but the ones whose gold file was
    renamed between commits get their own. Keying every structure by corpus id rather
    than by repo keeps those two cases on the same code path, so a per-commit instance
    cannot silently fall back to a listing that does not contain its answer.
    """

    def __init__(self, repo_listings: dict, commit_lists: dict):
        self.repo_listings = repo_listings
        self.commit_lists = commit_lists
        self._files: dict[str, list[str]] = {}
        self._bm25: dict[str, BM25] = {}
        self._vecs: dict[str, list[list[float]]] = {}

    def key(self, inst) -> str:
        if (inst.repo, inst.base_commit[:12]) in self.commit_lists:
            return f"{inst.repo}@{inst.base_commit[:12]}"
        return inst.repo

    def files(self, inst) -> list[str]:
        k = self.key(inst)
        if k not in self._files:
            self._files[k] = source_files(listing_for(inst, self.repo_listings, self.commit_lists))
        return self._files[k]

    def bm25(self, inst) -> BM25:
        k = self.key(inst)
        if k not in self._bm25:
            self._bm25[k] = BM25(self.files(inst))
        return self._bm25[k]

    def vectors(self, inst) -> list[list[float]] | None:
        k = self.key(inst)
        if k in self._vecs:
            return self._vecs[k]
        EMB.mkdir(parents=True, exist_ok=True)
        cache = EMB / f"{k.replace('/', '__').replace('@', '__at__')}.json.gz"
        files = self.files(inst)
        if cache.exists():
            with gzip.open(cache, "rt", encoding="utf-8") as fh:
                blob = json.load(fh)
            if blob["files"] == files:
                self._vecs[k] = blob["vectors"]
                return self._vecs[k]
        vecs = embed(files)
        if vecs is None:
            return None
        with gzip.open(cache, "wt", encoding="utf-8") as fh:
            json.dump({"key": k, "files": files, "vectors": vecs}, fh)
        self._vecs[k] = vecs
        return vecs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["bm25", "embed", "llm", "llm_rerank"], action="append")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default="qwen2.5-coder:14b")
    args = ap.parse_args()
    which = args.only or ["bm25", "embed", "llm", "llm_rerank"]

    repo_listings = cached_repos()
    if not repo_listings:
        print("No repository listings cached. Run: python src/trees.py")
        return 1
    commit_lists = commit_listings()

    data = [d for d in load() if d.gold_files and d.repo in repo_listings]
    if args.limit:
        data = data[: args.limit]

    corpora = Corpora(repo_listings, commit_lists)

    # Coverage against the listing each instance will actually be scored on.
    present = sum(1 for d in data if d.gold_files[0] in set(corpora.files(d)))
    print(f"repos cached      : {len(repo_listings)}/12")
    print(f"per-commit repairs: {len(commit_lists)}")
    print(f"instances scorable: {len(data)}")
    print(
        f"gold file present : {present}/{len(data)} ({present / len(data):.1%})"
        "   <- ceiling for every retriever"
    )

    if {"llm", "llm_rerank"} & set(which) and not ollama_ready(args.model):
        print(f"  {args.model} not available; skipping model arms")
        which = [w for w in which if not w.startswith("llm")]

    for name in which:
        path = OUT / f"{name}.jsonl"
        done = _done(path)
        todo = [d for d in data if d.instance_id not in done]
        print(f"\n{name}: {len(done)} done, {len(todo)} to go", flush=True)
        if not todo:
            continue

        # Embed every query in one batch. Done per-instance instead, each call evicts
        # and reloads the embedder against the 14B's VRAM - 2.1 s versus 18 ms.
        qvecs: dict[str, list[float]] = {}
        if name == "embed":
            for inst in todo:
                if corpora.vectors(inst) is None:
                    print("  embed: corpus embedding failed; skipping arm")
                    todo = []
                    break
            if not todo:
                continue
            qv = embed([d.problem_statement for d in todo])
            if qv is None:
                print("  embed: query batch failed; skipping arm")
                continue
            qvecs = {d.instance_id: v for d, v in zip(todo, qv)}

        t0 = time.time()
        for i, inst in enumerate(todo, 1):
            files = corpora.files(inst)
            if name == "bm25":
                ranked = [p for p, _ in corpora.bm25(inst).rank(inst.problem_statement, TOP)]
            elif name == "embed":
                ranked = [
                    p
                    for p, _ in cosine_rank(
                        qvecs[inst.instance_id], corpora.vectors(inst), files, TOP
                    )
                ]
            elif name == "llm":
                ranked = llm_locate(inst.repo, inst.problem_statement, k=10, model=args.model)
            pool_has_gold = None
            if name == "llm_rerank":
                pool = [p for p, _ in corpora.bm25(inst).rank(inst.problem_statement, POOL)]
                # The reranker can only return what BM25 handed it, so its ceiling is
                # BM25 recall@POOL, not 100%. Recorded per instance because without it
                # a low rerank score reads as the model failing when the gold file was
                # never in the list it was given.
                pool_has_gold = int(inst.gold_files[0] in pool)
                ranked = llm_rerank(inst.repo, inst.problem_statement, pool, k=10, model=args.model)

            row = {
                "instance_id": inst.instance_id,
                "repo": inst.repo,
                "gold": inst.gold_files[0],
                "tier": tier_for(inst, inst.problem_statement),
                "ranked": ranked,
                "retriever": name,
            }
            if pool_has_gold is not None:
                row["pool_has_gold"] = pool_has_gold
            _append(path, row)
            if i % 10 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                print(
                    f"  {i}/{len(todo)}  {rate:.1f}s/instance  "
                    f"eta {rate * (len(todo) - i) / 60:.0f} min",
                    flush=True,
                )

    results: list[dict] = []
    for name in ["bm25", "embed", "llm", "llm_rerank"]:
        results.extend(_done(OUT / f"{name}.jsonl").values())
    if not results:
        print("nothing scored")
        return 1

    # The generative arm names modules, not repository paths. Resolve them against the
    # real listing before scoring; without this every correct matplotlib answer is a
    # miss, because its package sits under `lib/`. Selective arms need no resolution -
    # they can only return paths that were already in the candidate set.
    by_instance = {d.instance_id: d for d in data}
    for r in results:
        if r["retriever"] == "llm" and r["instance_id"] in by_instance:
            r["ranked"] = resolve(r["ranked"], set(corpora.files(by_instance[r["instance_id"]])))

    summary = summarize(results)
    print_report(summary)

    # Fabrication only means something for the arm that generates paths; the reranker
    # picks from a supplied list and cannot invent one, which is the point of having it.
    # Union across every listing seen for a repo. Keyed by repo alone, an instance
    # scored on a per-commit listing would decide existence for the whole repository,
    # and a path that is real at another commit would be counted as invented.
    listings_by_repo: dict[str, list[str]] = {}
    _seen: dict[str, set[str]] = {}
    for d in data:
        s = _seen.setdefault(d.repo, set())
        s.update(corpora.files(d))
    listings_by_repo = {r: sorted(s) for r, s in _seen.items()}
    llm_rows = [r for r in results if r["retriever"] == "llm"]
    if llm_rows:
        fab = fabrication_rate(llm_rows, listings_by_repo)
        print("\n" + "=" * 74)
        print("LLM PATH VALIDITY - does the model name files that exist?")
        print("=" * 74)
        print(f"  paths named      : {fab['paths_named']}")
        print(f"  paths that exist : {fab['paths_that_exist']}  ({fab['rate_exists']:.1%})")
        print(f"  fabricated       : {1 - fab['rate_exists']:.1%}")
        summary["_llm_path_validity"] = fab

    rr = [r for r in results if r["retriever"] == "llm_rerank" and "pool_has_gold" in r]
    if rr:
        ceiling = sum(r["pool_has_gold"] for r in rr) / len(rr)
        got = sum(1 for r in rr if r["gold"] in r["ranked"][:10]) / len(rr)
        print("\n" + "=" * 74)
        print(f"RERANKER CEILING - gold file present in the BM25 top-{POOL} it was given")
        print("=" * 74)
        print(f"  ceiling (BM25 recall@{POOL}) : {ceiling:.1%}")
        print(f"  reranker recall@10          : {got:.1%}")
        print(f"  share of ceiling realised   : {got / ceiling if ceiling else 0:.1%}")
        summary["_rerank_ceiling"] = {
            "pool": POOL,
            "ceiling": ceiling,
            "recall@10": got,
            "share_realised": got / ceiling if ceiling else 0.0,
        }

    summary["_coverage"] = {"present": present, "total": len(data), "rate": present / len(data)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
