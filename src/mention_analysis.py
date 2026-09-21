"""How discoverable is the gold file from the issue text alone?

Before asking whether a *model* can localize the fix, ask whether the information is
even present. If the issue names the file, retrieval is lookup. If it names nothing,
any retriever must infer the location from behaviour described in prose - a different
and much harder task.

This splits the 300 instances into difficulty tiers using nothing but string matching,
so the result is deterministic and reproducible: no model, no seed, no variance.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from data import Instance, load

TIERS = ["full_path", "basename", "stem_only", "not_mentioned"]


def tier_for(inst: Instance, text: str) -> str:
    """Strongest form in which the gold file appears in the text."""
    gold = inst.gold_files[0]
    path = Path(gold)
    basename = path.name  # separable.py
    stem = path.stem  # separable

    # Normalise separators so windows/posix spellings both match.
    haystack = text.replace("\\", "/")

    if gold in haystack:
        return "full_path"
    if basename in haystack:
        return "basename"
    # Word-boundary match so "separable" does not match inside "inseparable".
    if re.search(rf"\b{re.escape(stem)}\b", haystack):
        return "stem_only"
    return "not_mentioned"


def report(data: list[Instance], field: str, text_of) -> Counter:
    tiers = Counter(tier_for(d, text_of(d)) for d in data)
    total = len(data)
    print(f"\n--- gold file discoverability from {field} ---")
    for tier in TIERS:
        n = tiers[tier]
        bar = "#" * round(40 * n / total)
        print(f"  {tier:14} {n:4}  {n / total:6.1%}  {bar}")
    findable = total - tiers["not_mentioned"]
    print(f"  {'-' * 60}")
    print(f"  mentioned in some form: {findable}/{total} ({findable / total:.1%})")
    print(
        f"  NOT mentioned at all  : {tiers['not_mentioned']}/{total} "
        f"({tiers['not_mentioned'] / total:.1%})  <- retrieval must infer these"
    )
    return tiers


def main() -> None:
    every = load()
    # Lite is single-file by construction; Full is not. `tier_for` reads gold_files[0],
    # so a multi-file fix would be scored on whichever file the diff happened to list
    # first - and would silently be counted as if the whole fix were discoverable from it.
    # Filtering keeps the two splits measuring the same thing, and the exclusion is
    # printed rather than assumed.
    data = [i for i in every if i.is_single_file]
    dropped = len(every) - len(data)
    print(f"instances: {len(data)} single-file fixes")
    if dropped:
        print(f"  {dropped} multi-file instances excluded - gold_files[0] is not the fix")

    report(data, "problem_statement", lambda d: d.problem_statement)
    report(data, "problem_statement + hints", lambda d: d.problem_statement + "\n" + d.hints_text)

    # Does difficulty vary by repo? A per-repo skew would mean an aggregate score is
    # partly measuring repo mix rather than method quality.
    print("\n--- 'not mentioned' rate by repo (problem_statement only) ---")
    by_repo: dict[str, list[int]] = {}
    for d in data:
        hit = tier_for(d, d.problem_statement) == "not_mentioned"
        by_repo.setdefault(d.repo, []).append(int(hit))
    for repo, hits in sorted(by_repo.items(), key=lambda kv: -len(kv[1])):
        rate = sum(hits) / len(hits)
        print(f"  {repo:28} n={len(hits):4}  not-mentioned {rate:6.1%}")

    # How long is the issue text? Long prose with no filename is the hard case.
    lengths = sorted(len(d.problem_statement) for d in data)
    mid = lengths[len(lengths) // 2]
    print(f"\nproblem_statement length: min {lengths[0]}, median {mid}, max {lengths[-1]} chars")


if __name__ == "__main__":
    main()
