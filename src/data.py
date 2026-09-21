"""Load SWE-bench Lite from the local Hugging Face cache and recover gold file paths.

No network, no repo clones. The parquet is already cached; the gold patch tells us
which files a correct solution touches, which is all the localization experiment needs.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Lite is 300 instances, Full is 2,294 of the same kind from the same repositories.
# The headline here - how often the issue text names the file you must change - is a
# property of how people write bug reports, and 300 is a sample of that rather than the
# population. Set SWEBENCH_SPLIT=full to read the larger one; the code path is identical
# because the two parquets share a schema.
SPLITS = {
    "lite": ("princeton-nlp/SWE-bench_Lite", "datasets--princeton-nlp--SWE-bench_Lite"),
    "full": ("princeton-nlp/SWE-bench", "datasets--princeton-nlp--SWE-bench"),
}
SPLIT = os.environ.get("SWEBENCH_SPLIT", "lite").lower()
if SPLIT not in SPLITS:
    raise SystemExit(f"SWEBENCH_SPLIT must be one of {sorted(SPLITS)}, got {SPLIT!r}")
HF_REPO, CACHE_DIR_NAME = SPLITS[SPLIT]

# A unified diff names the file twice; the b/ side is the post-image, which is the
# one that exists after a fix (the a/ side is /dev/null for added files).
DIFF_B_PATH = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Instance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    gold_files: tuple[str, ...]

    @property
    def is_single_file(self) -> bool:
        return len(self.gold_files) == 1


def gold_files_from_patch(patch: str) -> tuple[str, ...]:
    """Files a correct patch modifies, in the order they appear in the diff."""
    found = DIFF_B_PATH.findall(patch or "")
    # /dev/null appears for deletions; drop it and de-duplicate while keeping order.
    seen: dict[str, None] = {}
    for path in found:
        path = path.strip()
        if path and path != "/dev/null":
            seen.setdefault(path, None)
    return tuple(seen)


def _hf_cache_roots() -> list[Path]:
    """Every plausible Hugging Face cache location, most specific first.

    Resolved at call time rather than hardcoded: HF_HOME and HF_HUB_CACHE are the
    documented overrides, and ~/.cache/huggingface/hub is the default on every
    platform. A hardcoded path only works on the machine that wrote it.
    """
    roots = []
    if env := os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(env))
    if env := os.environ.get("HF_HOME"):
        roots.append(Path(env) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def find_parquet() -> Path | None:
    """Locate the cached test split without importing `datasets`."""
    for root in _hf_cache_roots():
        hits = sorted((root / CACHE_DIR_NAME).glob("snapshots/*/data/test-*.parquet"))
        if hits:
            return hits[0]
    return None


def _frame() -> pd.DataFrame:
    """Read the split from cache; fall back to downloading it (1.2 MB)."""
    cached = find_parquet()
    if cached is not None:
        return pd.read_parquet(cached)
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise FileNotFoundError(
            f"{HF_REPO} is not in the local Hugging Face cache and `datasets` is not "
            "installed, so it cannot be fetched.\n"
            "Fix either one:\n"
            "  pip install datasets        # then it downloads (~1.2 MB)\n"
            "  or set HF_HOME / HF_HUB_CACHE to the cache that already holds it.\n"
            f"Looked in: {', '.join(str(r) for r in _hf_cache_roots())}"
        ) from exc
    return load_dataset(HF_REPO, split="test").to_pandas()


def load(limit: int | None = None) -> list[Instance]:
    frame = _frame()
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            Instance(
                instance_id=row["instance_id"],
                repo=row["repo"],
                base_commit=row["base_commit"],
                problem_statement=row["problem_statement"] or "",
                hints_text=row["hints_text"] or "",
                gold_files=gold_files_from_patch(row["patch"]),
            )
        )
        if limit and len(rows) >= limit:
            break
    return rows


if __name__ == "__main__":
    data = load()
    print(f"instances      : {len(data)}")
    print(f"repos          : {len({d.repo for d in data})}")
    no_gold = [d for d in data if not d.gold_files]
    print(f"no gold file   : {len(no_gold)}")
    counts = [len(d.gold_files) for d in data]
    print(
        f"single-file fix: {sum(1 for c in counts if c == 1)} "
        f"({sum(1 for c in counts if c == 1) / len(data):.1%})"
    )
    print(
        f"files per fix  : min {min(counts)}, max {max(counts)}, "
        f"mean {sum(counts) / len(counts):.2f}"
    )
    print("\nsample gold paths:")
    for d in data[:5]:
        print(f"  {d.instance_id:32} {', '.join(d.gold_files)}")
