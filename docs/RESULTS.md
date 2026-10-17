# Results

[<- back to README](../README.md)

All 300 instances of `princeton-nlp/SWE-bench_Lite` (test split). Every number produced by
`src/mention_analysis.py`.

## Dataset integrity

| Property | Value |
|---|---|
| Instances | 300 |
| Instances with no gold file | 0 |
| Single-file fixes | **300 (100%)** |
| Repositories | 12 |
| `problem_statement` length | min 230, median 1,046, max 24,770 chars |

Every instance being a single-file fix is what makes localization cleanly defined: rank the
one correct file first.

## Discoverability from the issue text alone

| Tier | Instances | Share |
|---|---:|---:|
| Full path given verbatim | 51 | **17.0%** |
| Filename only | 18 | 6.0% |
| Module name only | 77 | 25.7% |
| **Never mentioned** | **154** | **51.3%** |

## Adding hints_text changes the task

| Tier | issue only | + hints |
|---|---:|---:|
| Full path | 17.0% | **33.0%** |
| Filename only | 6.0% | 6.3% |
| Module name only | 25.7% | 22.7% |
| **Never mentioned** | **51.3%** | **38.0%** |

`hints_text` is discussion from the issue thread. Including it nearly doubles the share of
instances where the file is handed to you.

**Results that use hints are not comparable to results that do not** - and papers do not
always state which setting they used.

## Per-repository difficulty

| Repo | n | never mentioned |
|---|---:|---:|
| sphinx-doc/sphinx | 16 | **87.5%** |
| psf/requests | 6 | 66.7% |
| pylint-dev/pylint | 6 | 66.7% |
| pytest-dev/pytest | 17 | 58.8% |
| django/django | 114 | 54.4% |
| sympy/sympy | 77 | 48.1% |
| scikit-learn/scikit-learn | 23 | 47.8% |
| pydata/xarray | 5 | 40.0% |
| pallets/flask | 3 | 33.3% |
| matplotlib/matplotlib | 23 | 30.4% |
| mwaskom/seaborn | 4 | 25.0% |
| astropy/astropy | 6 | **16.7%** |

Difficulty spans **16.7% to 87.5%**. `django/django` alone is 38% of the benchmark, so an
aggregate SWE-bench score is partly a measurement of **which repositories the benchmark
happens to contain**.

Several of these repos have fewer than 10 instances, so their individual rates are noisy -
the spread is the point, not any single row.

## Reproducing

```bash
python src/mention_analysis.py
```
