# Phase 2 - method

[<- back to README](../README.md)

Phase 1 asked whether a SWE-bench Lite issue names the file that has to change. Phase 2
asks the question that follows from it: when the issue does not name the file, can
anything find it?

## What is being measured

Each retriever receives an issue and the list of source files in that repository, and
returns those paths ranked best-first. The score is **recall@k** - whether the file the
gold patch modifies appears in the top k.

The 300 Lite instances are all single-file fixes, so "the gold file" is unambiguous and
recall@1 is a meaningful number rather than an artefact of averaging over multi-file
patches.

Every score is reported **per discoverability tier** from Phase 1. That split is the
result, not a presentation choice. A retriever can post a strong aggregate purely by
succeeding on the half of the benchmark where the answer is quoted in the text, and an
aggregate cannot tell that apart from one that understands the codebase.

## The four retrievers

| Name | What it does | Model |
|---|---|---|
| `bm25` | Okapi BM25 over tokenised paths | none |
| `embed` | cosine similarity over path embeddings | `nomic-embed-text` |
| `llm` | asked directly to name the files it would change | `qwen2.5-coder:14b` |
| `llm_rerank` | reorders the BM25 top-30 | `qwen2.5-coder:14b` |

Paths are tokenised on separators and camelCase humps, so `sql/compiler.py` and
`BaseDatabaseWrapper` both become the words a bug report would use. Segments that occur
in nearly every candidate - `py`, `src`, `lib`, `tests` - are dropped; they carry BM25
weight without carrying signal.

`llm` and `llm_rerank` both run at temperature 0. A measurement whose value depends on a
seed nobody reports is not a measurement.

### Why both LLM arms exist

`llm` generates paths from its own knowledge. These repositories are large, old and
public, so the model has very likely seen them during training, and a hit may mean it
localized the bug or merely that it remembers the layout. `llm_rerank` selects from a
list it is given, which makes fabrication impossible. The gap between the two arms is
where the memorisation question lives.

`llm_rerank` is also capped: it can only return what BM25 put in front of it, so its
ceiling is BM25 recall@30, not 100%. That ceiling is recorded per instance and reported
alongside the score, because otherwise a capped arm reads as a failing one.

## Candidate sets

Candidates come from the GitHub Trees API - **paths only, no file contents**. Fetching
~20k blobs would be a clone by another name, and the bandwidth here does not allow it.

This makes Phase 2 a measurement of *path* retrieval specifically. A production localizer
would read code as well, so these numbers are a floor for what retrieval can do, not a
ceiling. The `llm` arm is the exception: it has read this code before, during training.

### One tree per repository

The 300 instances span 297 distinct base commits. Fetching a tree per commit means 297
multi-megabyte responses, and on this connection the recursive fetch of `django/django`
is killed mid-transfer. File *paths* barely move between commits of one repository, so a
single listing per repo covers every instance in it.

That approximation is not free, and it was measured rather than assumed: 277 of 300 gold
files were present in their repository's listing, **92.3%**. The shortfall was not evenly
spread - scikit-learn was 65.2%, because it moved most modules behind underscore names
(`encoders.py` -> `_encoders.py`) between the commits involved.

Rather than report around a 7.7% floor of automatic misses, the 23 affected instances had
their exact `base_commit` tree fetched individually. Any instance still missing its gold
file is kept in the denominator and scored as a miss by every retriever, so the numbers
are pessimistic by exactly that amount and never optimistic.

### Large trees

A recursive fetch that dies is retried as a walk: the root listing first, then one
recursive call per top-level subtree. Each response is a fraction of the size, which is
what makes `django/django` (6,054 files) arrive at all. A subtree that cannot be fetched
after three attempts fails the whole listing rather than returning a partial one - an
incomplete candidate set lowers every score computed from it, invisibly.

## Embedding prefixes

`nomic-embed-text` is trained with task prefixes - `search_document: ` on the indexed
side, `search_query: ` on the query side - and asymmetric retrieval models usually lose
accuracy without them. Omitting them would have handicapped the embedding arm, so it was
tested rather than assumed.

On sympy (1,099 files, 77 instances) the prefixes made it **worse**: recall@10 fell from
45.5% to 37.7%, and on the `not_mentioned` tier from 24.3% to 21.6%. The prefixed
configuration is therefore not used.

The likely reason is that neither side here is what the prefixes were trained for. The
"documents" are file paths, not prose passages, and the "queries" are multi-thousand
character bug reports, not search queries.

## BM25 parameters

BM25 was swept over `k1 in {0.9, 1.2, 1.5, 2.0}` x `b in {0, 0.25, 0.5, 0.75, 1.0}`
across all 300 instances. Recall@10 ranged from 28.0% to 31.7%, and recall@10 on the
`not_mentioned` tier from 5.8% to 8.4%. The reported configuration is the best of the
twenty, `k1=2.0, b=1.0`.

Reporting the best rather than the default is deliberate: a weak baseline would make the
model arms look better for free. The range matters more than the winner - the conclusion
is the same at every setting, so it does not rest on a parameter choice.

## Reproducing

```bash
python src/trees.py            # fetch the 12 repository listings (network)
python run_phase2.py           # all four retrievers
python run_phase2.py --only bm25 --limit 20   # quick check, no model needed
```

Each retriever appends one JSON line per instance to `data/phase2/<name>.jsonl` and skips
what is already there, so an interrupted run resumes and loses at most one instance.

## What this does not measure

- **Whether the patch is correct.** Finding the file is necessary, not sufficient. The
  question Phase 2 leaves open is the one in `FUTURE.md`: given the right file, how often
  is the fix right?
- **Multi-file fixes.** Lite is single-file by construction. Full SWE-bench is not, and
  whether these proportions survive there is untested.
- **Retrieval over file contents.** Paths only, for the bandwidth reason above.
- **Any model larger than 14B**, or any hosted model. One local coder model, one local
  embedder.
