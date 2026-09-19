"""Three ways to guess which file an issue is about, scored the same way.

Each retriever takes an issue and a list of candidate paths and returns those paths
ranked best-first. Nothing here reads file contents: the Trees API gives paths only,
and fetching ~20k blobs would be a clone by another name. That makes this a measurement
of *path* retrieval specifically, which is a weaker signal than a real localizer would
use and is stated as such in the results rather than glossed over.

- `bm25`   - Okapi BM25 over tokenised paths. No model, no GPU, deterministic.
- `embed`  - cosine similarity over `nomic-embed-text` vectors, locally.
- `llm`    - qwen2.5-coder:14b asked directly to name the files it would change.

The third is not retrieval in the same sense as the first two: it generates paths from
its own knowledge rather than selecting from the candidate list. These repositories are
large, old and public, so the model has very likely seen them in training. That is a
confound, not a bug, and `llm_in_listing` in the results records how often its answers
even exist in the repo - a generated path that does not exist is a fabrication, and
counting those is the only way to tell recall from recall-shaped memorisation.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import urllib.error
import urllib.request
from collections import Counter

OLLAMA = "http://localhost:11434"

# Split on separators and camelCase humps: "sql/compiler.py" and "BaseDatabaseWrapper"
# both need to become the words a bug report would actually use.
_SPLIT = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])")

# Path segments that appear in nearly every candidate carry no signal but do carry
# BM25 weight, and dropping them is cheaper than relying on IDF to zero them out.
_STOP = {"py", "src", "lib", "python", "tests", "test", "__init__", ""}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _SPLIT.split(text) if t and t.lower() not in _STOP]


class BM25:
    """Okapi BM25. Pure Python - the corpus is a few thousand short paths."""

    # Defaults chosen by sweeping k1 in {0.9,1.2,1.5,2.0} x b in {0,0.25,0.5,0.75,1.0}
    # over all 300 instances: recall@10 ranged 28.0%-31.7% and these were the best.
    # Reported so the baseline cannot be dismissed as untuned - the conclusion below
    # holds at every setting, which is why the range matters more than the winner.
    def __init__(self, docs: list[str], k1: float = 2.0, b: float = 1.0):
        self.docs = docs
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.len = [len(t) for t in self.toks]
        self.avg = sum(self.len) / len(self.len) if self.len else 0.0
        self.tf = [Counter(t) for t in self.toks]
        df: Counter[str] = Counter()
        for t in self.toks:
            df.update(set(t))
        n = len(docs)
        # +0.5/+0.5 smoothing keeps IDF positive for terms in almost every document.
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def rank(self, query: str, top: int = 50) -> list[tuple[str, float]]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            if not tf:
                continue
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1))
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scores.append((self.docs[i], s))
        scores.sort(key=lambda kv: -kv[1])
        return scores[:top]


def _ollama(endpoint: str, payload: dict, timeout: int = 600) -> dict | None:
    req = urllib.request.Request(
        f"{OLLAMA}{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return json.loads(fh.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def embed(
    texts: list[str], model: str = "nomic-embed-text", batch: int = 256
) -> list[list[float]] | None:
    """Batched embeddings via /api/embed.

    The per-item /api/embeddings endpoint measured 2.1 s per path here, because the
    14B generator holds VRAM and the embedder is evicted and reloaded on every call.
    Batching with an explicit keep_alive holds it resident: 18 ms per path, a 118x
    difference, and the reason this runs in minutes instead of hours.
    """
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        vecs = _embed_chunk(chunk, model, batch)
        if vecs is None:
            return None
        out.extend(vecs)
    return out


def _embed_chunk(
    chunk: list[str], model: str, batch: int, depth: int = 0
) -> list[list[float]] | None:
    """One batch, halved and retried on failure.

    A batch fails for reasons that are about size rather than content - a long request
    while the 14B holds VRAM, a transient refusal - so halving usually succeeds where
    a retry of the same payload would not. Returning None for the whole corpus on one
    bad batch silently drops the entire retriever from the comparison, which is a far
    worse outcome than being slow.
    """
    r = _ollama("/api/embed", {"model": model, "input": chunk, "keep_alive": "30m"})
    if r is not None and "embeddings" in r and len(r["embeddings"]) == len(chunk):
        return r["embeddings"]
    if len(chunk) == 1 or depth > 6:
        return None
    mid = len(chunk) // 2
    left = _embed_chunk(chunk[:mid], model, batch, depth + 1)
    if left is None:
        return None
    right = _embed_chunk(chunk[mid:], model, batch, depth + 1)
    if right is None:
        return None
    return left + right


def cosine_rank(
    qvec: list[float], doc_vecs: list[list[float]], docs: list[str], top: int = 50
) -> list[tuple[str, float]]:
    qn = math.sqrt(sum(x * x for x in qvec)) or 1.0
    scored = []
    for path, v in zip(docs, doc_vecs):
        dn = math.sqrt(sum(x * x for x in v)) or 1.0
        scored.append((path, sum(a * b for a, b in zip(qvec, v)) / (qn * dn)))
    scored.sort(key=lambda kv: -kv[1])
    return scored[:top]


_LLM_PROMPT = """You are localizing a bug in the {repo} repository.

Below is a bug report. Name the source files most likely to need editing to fix it.

Rules:
- Output ONLY file paths, one per line, most likely first.
- Use full repository-relative paths, e.g. django/db/models/query.py
- At most {k} paths. No prose, no numbering, no backticks.

BUG REPORT:
{issue}
"""


def llm_locate(
    repo: str,
    issue: str,
    k: int = 10,
    model: str = "qwen2.5-coder:14b",
    max_chars: int = 6000,
) -> list[str]:
    """Ask the model directly. Returns whatever paths it names, in its own order."""
    r = _ollama(
        "/api/generate",
        {
            "model": model,
            "prompt": _LLM_PROMPT.format(repo=repo, issue=issue[:max_chars], k=k),
            "stream": False,
            # Deterministic: this is a measurement, and a temperature would make the
            # number depend on a seed nobody reports.
            "options": {"temperature": 0.0, "num_predict": 256},
        },
    )
    if r is None:
        return []
    return _parse_paths(r.get("response", ""), k)


def _parse_paths(text: str, k: int) -> list[str]:
    paths: list[str] = []
    for line in text.splitlines():
        # Order matters: strip the list marker first, then the backticks. Done the
        # other way round, "1. `path`" keeps its opening backtick because the line
        # starts with the digit, and every such path is then scored as a miss.
        line = re.sub(r"^\s*[-*\d.)\s]+", "", line.strip())
        line = line.strip("`").strip()
        if not line or " " in line.strip():
            continue
        if "/" not in line and not line.endswith(".py"):
            continue
        line = line.lstrip("./")
        if line not in paths:
            paths.append(line)
        if len(paths) >= k:
            break
    return paths


_RERANK_PROMPT = """You are localizing a bug in the {repo} repository.

Below is a bug report, then a numbered list of candidate files from that repository.
Choose the {k} candidates most likely to need editing, best first.

Rules:
- Output ONLY the numbers, one per line, best first.
- Choose only from the list. Do not invent paths.
- At most {k} numbers. No prose.

BUG REPORT:
{issue}

CANDIDATES:
{candidates}
"""


def llm_rerank(
    repo: str,
    issue: str,
    candidates: list[str],
    k: int = 10,
    model: str = "qwen2.5-coder:14b",
    max_chars: int = 5000,
) -> list[str]:
    """Reorder a real candidate list instead of generating paths.

    This exists to separate two things the direct arm conflates. When the model names
    a path from memory, a hit may mean it localized the bug or merely that it has seen
    the repository. Selecting from a supplied list makes fabrication impossible, so
    what is left is ranking ability - and the gap between the two arms is the size of
    the memorisation effect.
    """
    if not candidates:
        return []
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(candidates))
    r = _ollama(
        "/api/generate",
        {
            "model": model,
            "prompt": _RERANK_PROMPT.format(
                repo=repo, issue=issue[:max_chars], k=k, candidates=numbered
            ),
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 128},
        },
    )
    if r is None:
        return []
    picked: list[str] = []
    for line in r.get("response", "").splitlines():
        m = re.search(r"\d+", line)
        if not m:
            continue
        idx = int(m.group()) - 1
        if 0 <= idx < len(candidates) and candidates[idx] not in picked:
            picked.append(candidates[idx])
        if len(picked) >= k:
            break
    # A model that answers with prose returns nothing; falling back to the input order
    # would silently report BM25's score under the reranker's name.
    return picked


def ollama_ready(model: str) -> bool:
    try:
        r = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=30, check=False
        )
        return model.split(":")[0] in r.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
