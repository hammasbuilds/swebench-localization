# Phase 2 results - retrieval scored as recall@k

[<- back to README](../README.md) &middot; [Method](PHASE2.md) &middot; [Phase 1 results](RESULTS.md)

Phase 1 asked whether a SWE-bench Lite issue names the file that has to change: 51.3% do
not. Phase 2 asks what happens next. When the issue does not name the file, can anything
find it?

All 300 instances. Reproduce with `python run_phase2.py`; the raw output is in
`data/phase2/final_report.log`.

## Headline

| Retriever | @1 | @5 | @10 | @20 |
|---|---:|---:|---:|---:|
| BM25 over paths | 9.7% | 24.3% | 34.0% | 43.7% |
| `nomic-embed-text` over paths | 14.7% | 33.7% | 43.0% | 53.7% |
| `qwen2.5-coder:14b` as locator | **47.7%** | **70.3%** | **72.7%** | 72.7% |
| the same 14B reranking BM25's top-30 | 36.7% | 43.7% | 46.7% | 46.7% |

The ceiling for every retriever is 99.7% - one instance's gold file could not be fetched
and is scored as a miss for everyone.

## The result: who wins depends entirely on whether the issue names the file

Recall@10 by Phase 1 discoverability tier:

| Tier | n | BM25 | embed | LLM | LLM rerank |
|---|---:|---:|---:|---:|---:|
| `full_path` | 51 | 74.5% | 56.9% | **98.0%** | 86.3% |
| `basename` | 18 | 66.7% | 50.0% | **94.4%** | 83.3% |
| `stem_only` | 77 | 50.6% | 59.7% | **84.4%** | 70.1% |
| `not_mentioned` | 154 | **8.4%** | 29.2% | **55.8%** | 17.5% |

**BM25 collapses from 74.5% to 8.4%.** On the half of the benchmark where the issue never
names the file, lexical matching over paths has almost nothing to match. That is the
Phase 1 claim turned into a retrieval number: a single SWE-bench score largely rewards
string matching, and the half that needs something else is scored on a retrieval failure.

FUTURE.md predicted BM25 might win overall. **It did not** - embeddings beat it by 9 points
at recall@10. But the aggregate hides a cleaner story: BM25 wins wherever the path is
quoted, embeddings win wherever it is not, and the crossover sits exactly at `stem_only`.

## Why embeddings lose where the path is quoted

Embeddings scoring *lower* on `full_path` (56.9%) than on `stem_only` (59.7%) is
non-monotonic and looked like a bug. It is not:

| Tier | median issue length | mean distinct `.py` paths quoted |
|---|---:|---:|
| `full_path` | 1827 chars | 3.5 |
| `basename` | 2907 chars | 5.2 |
| `stem_only` | 1033 chars | 0.4 |
| `not_mentioned` | 894 chars | 0.8 |

Issues that quote a full path are **tracebacks**: long, and naming three to five different
files. That makes them a disambiguation problem, not a similarity problem. The embedding of
a 1,800-character traceback is diffuse, and the gold file competes with four other quoted
paths that are all about the same topic. BM25 still matches the quoted string, and the
model reads the traceback and reasons about it.

## The model's advantage is knowledge of the repository, not ranking

The generative arm scores 72.7%. The obvious question is whether it has localized the bug
or simply remembers these repositories, which are large, old and public. The reranker arm
was built to separate those.

| Tier | reranker @10 | its ceiling (BM25@30) | share of ceiling realised |
|---|---:|---:|---:|
| `full_path` | 86.3% | 88.2% | 97.8% |
| `basename` | 83.3% | 83.3% | 100.0% |
| `stem_only` | 70.1% | 72.7% | 96.4% |
| `not_mentioned` | 17.5% | 23.4% | 75.0% |
| **overall** | **46.7%** | **50.7%** | **92.1%** |

The reranker is not weak. It extracts 92.1% of everything BM25 hands it, and on three of
four tiers it takes 96-100% of what is available. It is close to the best possible reranker
over that candidate list.

It still loses to the generative arm by 26 points, and on `not_mentioned` by a factor of
three. The reason is in the ceiling column: **on `not_mentioned`, BM25's entire top-30
contains the gold file only 23.4% of the time.** The generative arm scores 55.8% there -
more than double what BM25 can surface at any depth.

So the same model, on the same issues, at the same temperature, does far better when
allowed to name files than when restricted to ranking them. It is not ranking better. It
is producing candidates that first-stage retrieval never proposes, which means it is
drawing on knowledge of these codebases rather than on the issue text alone.

That is consistent with memorisation and is not proof of it - the model may have learned
transferable conventions about where Django puts its query compiler. Distinguishing those
needs a repository the model cannot have seen, which is item 2 in
[FUTURE.md](FUTURE.md).

**One in five paths it names does not exist.**

```
paths named      : 1984
paths that exist : 1570  (79.1%)
fabricated       : 20.9%
```

A generative localizer that is right 72.7% of the time also invents a fifth of its answers.
Recall alone cannot tell those apart, which is why this is reported next to it.

## By repository

Recall@10. The spread is larger than the difference between methods.

| Repo | n | BM25 | embed | LLM | rerank |
|---|---:|---:|---:|---:|---:|
| django/django | 114 | 10.5% | 35.1% | 77.2% | 22.8% |
| sympy/sympy | 77 | 45.5% | 50.6% | 67.5% | 59.7% |
| scikit-learn | 23 | 60.9% | 52.2% | 73.9% | 73.9% |
| matplotlib | 23 | 52.2% | 26.1% | 78.3% | 69.6% |
| pytest-dev/pytest | 17 | 47.1% | 52.9% | 58.8% | 58.8% |
| sphinx-doc/sphinx | 16 | 25.0% | 56.2% | 68.8% | 37.5% |
| pylint-dev/pylint | 6 | 33.3% | 16.7% | 33.3% | 33.3% |
| psf/requests | 6 | 33.3% | 50.0% | 83.3% | 50.0% |
| astropy/astropy | 6 | 83.3% | 83.3% | 83.3% | 83.3% |
| pydata/xarray | 5 | 40.0% | 40.0% | 80.0% | 60.0% |
| mwaskom/seaborn | 4 | 75.0% | 25.0% | 75.0% | 75.0% |
| pallets/flask | 3 | 100.0% | 66.7% | 100.0% | 100.0% |

django is 38% of Lite and BM25's worst repository at 10.5%, which drags the aggregate
hard. Any SWE-bench evaluation reporting one number is reporting partly on benchmark
composition - the same conclusion Phase 1 reached from the other direction.

## Things that were tested rather than assumed

**BM25 was tuned.** Swept `k1 in {0.9, 1.2, 1.5, 2.0}` x `b in {0, 0.25, 0.5, 0.75, 1.0}`
over all 300. Recall@10 ranged 28.0%-31.7% and `not_mentioned` 5.8%-8.4%. The best
configuration is reported. An untuned baseline would flatter every other arm for free, and
the conclusion holds at all twenty settings.

**The embedder's documented prefixes were tested and rejected.** `nomic-embed-text` is
trained with `search_document:` / `search_query:` prefixes, and omitting them usually costs
accuracy. Here they *cost* accuracy: on sympy, recall@10 fell 45.5% -> 37.7%. Neither side
is what the prefixes were trained for - the documents are file paths, not passages, and the
queries are multi-thousand-character bug reports.

**Path resolution was audited.** The generative arm names module paths, and matplotlib's
package lives under `lib/`, so scoring literally marked every correct matplotlib answer as
a miss - that tier read 22.2% while another read 96.1%. Generated paths are now resolved
against the real listing when exactly one file matches on a directory boundary. The audit:
it rewrites 6.2% of paths, never a bare filename, and the only prefixes it ever adds are
`lib` (90) and `src` (34).

**The candidate sets were repaired.** One listing per repository misses gold files that
were renamed between commits - scikit-learn moved its modules behind underscore names, and
coverage there was 65.2%. Fetching the exact `base_commit` tree for the 22 affected
instances took overall coverage from 92.3% to 99.7%.

## Limits

- **Paths only, no file contents.** The Trees API gives paths, and fetching ~20k blobs is a
  clone by another name. These are a floor for what retrieval can do. The generative arm is
  the exception - it has read this code, during training.
- **One instance is unreachable.** `django__django-15388`'s tree could not be fetched after
  two attempts; it is scored as a miss for every retriever.
- **One model, 14B, local.** Nothing here says how this scales with size or family.
- **Finding the file is not fixing the bug.** Whether localization predicts patch success is
  item 1 in [FUTURE.md](FUTURE.md), and it is the number that actually matters.
- `basename` has n=18. Treat that row as indicative.
