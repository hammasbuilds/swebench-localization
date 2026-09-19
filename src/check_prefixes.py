"""Does nomic-embed-text's task prefix change the embedding arm's score?

nomic-embed-text is trained with instruction prefixes - `search_document: ` on the
indexed side and `search_query: ` on the query side - and asymmetric retrieval models
typically lose accuracy when they are omitted. The first Phase 2 run omitted them, so
the embedding arm may have been handicapped against BM25.

This measures the difference on one repository rather than assuming it. Reporting a
baseline that was configured wrong would make every comparison against it meaningless.

    python src/check_prefixes.py sympy/sympy
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load
from evaluate import recall_at_k
from mention_analysis import tier_for
from retrieval import cosine_rank, embed
from trees import cached_repos, source_files


def score(files, insts, doc_prefix: str, query_prefix: str) -> dict:
    dv = embed([doc_prefix + f for f in files])
    if dv is None:
        return {}
    qv = embed([query_prefix + i.problem_statement for i in insts])
    if qv is None:
        return {}
    hits = {1: 0, 5: 0, 10: 0, 20: 0}
    nm = [0, 0]
    for inst, q in zip(insts, qv):
        ranked = [p for p, _ in cosine_rank(q, dv, files, 20)]
        for k in hits:
            hits[k] += recall_at_k(ranked, inst.gold_files[0], k)
        if tier_for(inst, inst.problem_statement) == "not_mentioned":
            nm[1] += 1
            nm[0] += recall_at_k(ranked, inst.gold_files[0], 10)
    n = len(insts)
    return {
        **{f"recall@{k}": v / n for k, v in hits.items()},
        "not_mentioned@10": nm[0] / nm[1] if nm[1] else 0.0,
        "n_not_mentioned": nm[1],
    }


def main() -> int:
    repo = sys.argv[1] if len(sys.argv) > 1 else "sympy/sympy"
    listings = cached_repos()
    if repo not in listings:
        print(f"{repo} not cached")
        return 1
    files = source_files(listings[repo])
    insts = [d for d in load() if d.repo == repo and d.gold_files]
    print(f"{repo}: {len(files)} files, {len(insts)} instances\n")

    variants = {
        "no prefix (as run)": ("", ""),
        "nomic task prefixes": ("search_document: ", "search_query: "),
    }
    rows = {}
    for name, (dp, qp) in variants.items():
        r = score(files, insts, dp, qp)
        if not r:
            print(f"  {name}: FAILED")
            continue
        rows[name] = r
        print(
            f"  {name:22} @1 {r['recall@1']:6.1%}  @10 {r['recall@10']:6.1%}  "
            f"@20 {r['recall@20']:6.1%}  not_mentioned@10 {r['not_mentioned@10']:6.1%}"
        )

    if len(rows) == 2:
        a, b = rows.values()
        d = b["recall@10"] - a["recall@10"]
        print(f"\n  prefix effect on recall@10: {d:+.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
