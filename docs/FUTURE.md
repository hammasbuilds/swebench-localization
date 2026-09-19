# Future work

[<- back to README](../README.md) &middot; [Phase 2 method](PHASE2.md) &middot; [Phase 2 results](RESULTS-PHASE2.md)

Phase 2 is done: BM25, embeddings, an LLM-as-locator and an LLM reranker scored as
recall@k, cut by the Phase 1 discoverability tiers. What follows is what it opened up.

## 1. Condition patch validity on localization

The number that actually matters: *given the right file was found, how often is the patch
correct?* That separates retrieval failure from reasoning failure cleanly, and it is the
metric a model-size comparison should move.

Phase 2 makes this tractable - it produces, per instance, whether each retriever found the
file. Conditioning patch success on that is the obvious next join.

## 2. Test the memorisation hypothesis directly

Phase 2's sharpest result is that the same model scores far higher generating paths from
memory than ranking a list it is handed. That is consistent with the model knowing these
repositories from training, but it does not prove it.

The clean test is a repository the model cannot have seen: instances built from commits
after the training cutoff, or from private code. If the generative advantage survives
there, it is localization ability; if it collapses to the reranker's level, it was recall.

## 3. Raise the reranker's ceiling

The reranker can only return what BM25 handed it, so its ceiling is BM25 recall@30. It
realises about 95% of that ceiling, which means the bottleneck is entirely the candidate
list and not the ranking.

Worth trying: a much larger pool, a union of BM25 and embedding candidates, or two rounds.
The interesting question is whether a reranker over a *better* first stage closes the gap
to the generative arm.

## 4. Retrieval over file contents

Phase 2 ranks paths, because the Trees API gives paths and fetching ~20k blobs is a clone
by another name. A real localizer reads code. These numbers are therefore a floor for what
retrieval can do, and the gap between path-only and content-aware retrieval is itself the
result.

## 5. Extend to full SWE-bench

2,294 instances, many multi-file. Lite is single-file by construction, and checking
whether that distorts these proportions is the obvious validity check on everything here.

Multi-file fixes also make recall@k ambiguous - all files, or any file? - which is a
measurement design question, not just more data.

## 6. Correlate difficulty with issue length

`problem_statement` spans 230 to 24,770 characters. Long prose naming no file is likely
the hardest tier, and that is directly testable with data already loaded.

## 7. Semantic discoverability, not just string matching

An issue quoting a unique error message effectively names the file without naming it.
Phase 1's tiers are string-matching only, so such instances sit in `not_mentioned` and
depress the apparent ceiling there.

Phase 2 gives a way to measure it: an instance in `not_mentioned` that every retriever
finds was probably discoverable after all.

## 8. Larger models, and other model families

One local 14B coder model. Whether the generative advantage scales with size, and whether
it is specific to this family, is untested.
