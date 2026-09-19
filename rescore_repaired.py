"""Drop rows for instances whose candidate set changed, so they can be rescored.

The 23 instances whose gold file was missing from their repository listing get their
exact `base_commit` tree fetched separately. That changes the candidate set they are
scored against, which invalidates any BM25 or embedding row already collected for them -
those were ranked over a corpus that did not contain the answer.

The two model arms are unaffected: `llm` generates paths without a corpus, and
`llm_rerank` is re-derived from BM25, so its rows are dropped alongside BM25's.

    python rescore_repaired.py           # show what would be dropped
    python rescore_repaired.py --apply   # drop them, then re-run run_phase2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data import load
from trees import commit_listings

OUT = Path(__file__).resolve().parent / "data" / "phase2"
# Arms whose ranking depends on the candidate set.
CORPUS_DEPENDENT = ("bm25", "embed", "llm_rerank")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    repaired = set(commit_listings())
    if not repaired:
        print("no per-commit listings on disk; nothing to rescore")
        return 0

    affected = {
        d.instance_id for d in load() if d.gold_files and (d.repo, d.base_commit[:12]) in repaired
    }
    print(f"per-commit listings : {len(repaired)}")
    print(f"instances affected  : {len(affected)}")

    for arm in CORPUS_DEPENDENT:
        path = OUT / f"{arm}.jsonl"
        if not path.exists():
            continue
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        keep = [r for r in rows if r["instance_id"] not in affected]
        drop = len(rows) - len(keep)
        print(f"  {arm:12} {len(rows):4} rows, dropping {drop}")
        if args.apply and drop:
            path.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")

    if args.apply:
        print("\ndropped. now re-run:  python run_phase2.py")
    else:
        print("\ndry run; pass --apply to drop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
