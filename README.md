# swebench-localization

[![ci](https://github.com/hammasbuilds/swebench-localization/actions/workflows/ci.yml/badge.svg)](https://github.com/hammasbuilds/swebench-localization/actions/workflows/ci.yml)
[![license](https://img.shields.io/github/license/hammasbuilds/swebench-localization)](LICENSE)
![python](https://img.shields.io/badge/python-3.11%2B-blue)

Solving a SWE-bench instance needs two different things: **find the file**, then **write the
patch**. We built a tool that measures how often the issue text even names the file that has
to change, and then scores four retrievers on finding it. It is for anyone building or
evaluating coding agents who wants to know whether a failure is search or reasoning. **51.3%
of SWE-bench Lite (48.3% of Full's single-file instances) never mention the file you must
fix.**

## What it does

- Parses the gold file(s) out of every reference patch, on SWE-bench Lite (300) and Full
  (2,294; 1,725 single-file).
- Places each instance on a strict ladder by how the issue refers to the gold file: full
  path, filename only, module name only (word-boundary match on the stem), or not at all.
  The strongest form present wins, so each instance lands in exactly one tier.
- Repeats the measurement with `hints_text` included, and per repository.
- Phase 2: scores BM25, embeddings (`nomic-embed-text`), an LLM locator and an LLM reranker
  (`qwen2.5-coder:14b`) as recall@k against every source file in the repository, cut by
  those tiers, and reports how many of the model's named paths do not exist.

## Sample input / output

`python src/mention_analysis.py` on SWE-bench Lite (real run, trimmed):

```text
instances: 300 single-file fixes

--- gold file discoverability from problem_statement ---
  full_path        51   17.0%  #######
  basename         18    6.0%  ##
  stem_only        77   25.7%  ##########
  not_mentioned   154   51.3%  #####################
  ------------------------------------------------------------
  mentioned in some form: 146/300 (48.7%)
  NOT mentioned at all  : 154/300 (51.3%)  <- retrieval must infer these

--- gold file discoverability from problem_statement + hints ---
  full_path        99   33.0%  #############
  basename         19    6.3%  ###
  stem_only        68   22.7%  #########
  not_mentioned   114   38.0%  ###############

--- 'not mentioned' rate by repo (problem_statement only) ---
  django/django                n= 114  not-mentioned  54.4%
  sympy/sympy                  n=  77  not-mentioned  48.1%
  matplotlib/matplotlib        n=  23  not-mentioned  30.4%
  sphinx-doc/sphinx            n=  16  not-mentioned  87.5%
  astropy/astropy              n=   6  not-mentioned  16.7%
  ...
```

## Results

| How the gold file is referenced | Lite (300) | Full (1,725 single-file) |
|---|---:|---:|
| Full path, verbatim | 17.0% | 22.0% |
| Filename only | 6.0% | 4.7% |
| Module name only | 25.7% | 25.0% |
| **Never mentioned at all** | **51.3%** | **48.3%** |

- **Hints change the task.** With `hints_text`, "never mentioned" drops to 38.0% on Lite and
  36.3% on Full. Results with and without hints are not comparable.
- **The aggregate hides a repo effect.** On Full, "never mentioned" ranges from matplotlib
  27.7% to pylint 66.7%; `django/django` is 38% of Lite and 39% of Full.
- **A quarter of Full needs more than one file changed** (569 of 2,294). Lite contains none
  of them, so every localization result on Lite is about the easier half.

**Phase 2: can anything find the file?** recall@10 on Lite, every source file in the
repository as the candidate set:

| Tier | n | BM25 | embeddings | 14B locator | 14B reranking BM25 |
|---|---:|---:|---:|---:|---:|
| `full_path` | 51 | 74.5% | 56.9% | **98.0%** | 86.3% |
| `stem_only` | 77 | 50.6% | 59.7% | **84.4%** | 70.1% |
| `not_mentioned` | 154 | **8.4%** | 29.2% | **55.8%** | 17.5% |

- BM25 falls from 74.5% to 8.4% on the half that never names the file.
- The 14B naming files freely scores 55.8% on `not_mentioned`; reranking BM25's top 30 it
  scores 17.5%, because BM25's top 30 contains the answer only 23.4% of the time. As a
  reranker it realises 92.1% of that ceiling.
- The locator invents **20.9%** of the paths it names.

More: [Phase 1 tables](docs/RESULTS.md) · [Phase 2 tables](docs/RESULTS-PHASE2.md) ·
[method](docs/METHOD.md) · [Phase 2 method](docs/PHASE2.md).

## Tech stack

- Python 3.11+
- pandas + PyArrow (SWE-bench parquet from the Hugging Face cache)
- Okapi BM25 in pure Python
- Ollama: `nomic-embed-text` (embeddings), `qwen2.5-coder:14b` (locator and reranker)
- GitHub Trees API for repository file listings (cached under `data/trees/`)
- pytest, ruff, uv, GitHub Actions

## Run it

```bash
git clone https://github.com/hammasbuilds/swebench-localization
cd swebench-localization
uv sync --all-groups

uv run python src/data.py              # dataset integrity + gold file extraction
uv run python src/mention_analysis.py  # the Phase 1 numbers
uv run python run_phase2.py            # Phase 2 retrievers (needs Ollama), resumable
uv run pytest -q                       # 45 tests, no network, no dataset, no model
```

SWE-bench Lite (1.2 MB) is read from the local Hugging Face cache (`princeton-nlp/SWE-bench_Lite`);
install the `fetch` extra (`datasets`) and load it once if it is not there. Phase 2 outputs are committed
in `data/phase2/`.

```
src/data.py               load from HF cache, parse gold files from patches
src/mention_analysis.py   discoverability tiers + per-repo breakdown
src/trees.py              repository file listings via the Trees API, cached
src/retrieval.py          BM25, embeddings, LLM locator, LLM reranker
src/evaluate.py           recall@k by tier, fabrication rate, path resolution
run_phase2.py             Phase 2 driver, resumable per instance
```

## Licence

MIT, see [LICENSE](LICENSE). SWE-bench (Jimenez et al., ICLR 2024) is MIT.
