"""Phase 2 tests: no network, no model, no dataset.

Every case is built from a hand-written corpus or a canned model response, so a CI run
tells you whether the scoring is correct, not whether GitHub and Ollama were up.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate import fabrication_rate, recall_at_k, resolve, summarize
from retrieval import BM25, _parse_paths, cosine_rank, tokenize
from trees import coverage, source_files

CORPUS = [
    "django/db/models/query.py",
    "django/db/models/sql/compiler.py",
    "django/forms/fields.py",
    "django/contrib/admin/options.py",
    "tests/queries/test_qs_combinators.py",
]

# --- tokenisation --------------------------------------------------------------------


def test_splits_on_path_separators():
    assert tokenize("django/db/models/query.py") == ["django", "db", "models", "query"]


def test_splits_camel_case():
    # A report says "BaseDatabaseWrapper"; the path says "base_database_wrapper".
    assert tokenize("BaseDatabaseWrapper") == ["base", "database", "wrapper"]


def test_drops_uninformative_segments():
    # "py" and "src" are in nearly every candidate, so they cannot discriminate.
    assert "py" not in tokenize("src/thing.py")
    assert "src" not in tokenize("src/thing.py")


# --- BM25 ----------------------------------------------------------------------------


def test_bm25_ranks_the_obvious_match_first():
    bm = BM25(CORPUS)
    ranked = [p for p, _ in bm.rank("QuerySet.union() crashes in the sql compiler")]
    assert ranked[0] == "django/db/models/sql/compiler.py"


def test_bm25_returns_nothing_when_no_term_overlaps():
    bm = BM25(CORPUS)
    assert bm.rank("completely unrelated wording here") == []


def test_bm25_respects_top_k():
    bm = BM25(CORPUS)
    assert len(bm.rank("django models query fields admin", top=2)) <= 2


def test_bm25_handles_empty_corpus():
    assert BM25([]).rank("anything") == []


# --- cosine --------------------------------------------------------------------------


def test_cosine_orders_by_similarity():
    docs = ["a.py", "b.py", "c.py"]
    vecs = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    ranked = [p for p, _ in cosine_rank([1.0, 0.0], vecs, docs)]
    assert ranked == ["a.py", "b.py", "c.py"]


def test_cosine_survives_a_zero_vector():
    # A degenerate embedding must not raise; it should just rank last.
    ranked = cosine_rank([1.0, 0.0], [[0.0, 0.0], [1.0, 0.0]], ["z.py", "a.py"])
    assert ranked[0][0] == "a.py"


# --- parsing model output ------------------------------------------------------------


def test_parses_plain_path_list():
    out = _parse_paths("django/db/models/query.py\ndjango/forms/fields.py", 10)
    assert out == ["django/db/models/query.py", "django/forms/fields.py"]


def test_strips_numbering_bullets_and_backticks():
    raw = "1. `django/db/models/query.py`\n- django/forms/fields.py\n* zzz/other.py"
    assert _parse_paths(raw, 10) == [
        "django/db/models/query.py",
        "django/forms/fields.py",
        "zzz/other.py",
    ]


def test_ignores_prose_lines():
    raw = "Here are the files you should look at:\ndjango/db/models/query.py\nHope this helps!"
    assert _parse_paths(raw, 10) == ["django/db/models/query.py"]


def test_deduplicates_and_honours_k():
    raw = "a/b.py\na/b.py\nc/d.py\ne/f.py"
    assert _parse_paths(raw, 2) == ["a/b.py", "c/d.py"]


def test_strips_leading_dot_slash():
    assert _parse_paths("./django/forms/fields.py", 5) == ["django/forms/fields.py"]


# --- recall --------------------------------------------------------------------------


def test_recall_at_k_is_a_prefix_check():
    ranked = ["a.py", "b.py", "c.py"]
    assert recall_at_k(ranked, "c.py", 3) == 1
    assert recall_at_k(ranked, "c.py", 2) == 0


def test_recall_on_empty_ranking_is_zero():
    assert recall_at_k([], "a.py", 10) == 0


def _row(iid, tier, ranked, gold="gold.py", repo="r/r", retriever="bm25"):
    return {
        "instance_id": iid,
        "repo": repo,
        "gold": gold,
        "tier": tier,
        "ranked": ranked,
        "retriever": retriever,
    }


def test_summary_splits_by_tier():
    rows = [
        _row("1", "full_path", ["gold.py"]),
        _row("2", "not_mentioned", ["other.py"]),
    ]
    s = summarize(rows, ks=(1,))
    assert s["bm25"]["by_tier"]["full_path"]["recall@1"] == 1.0
    assert s["bm25"]["by_tier"]["not_mentioned"]["recall@1"] == 0.0
    assert s["bm25"]["overall"]["recall@1"] == 0.5


def test_summary_keeps_retrievers_separate():
    rows = [
        _row("1", "full_path", ["gold.py"], retriever="bm25"),
        _row("1", "full_path", ["miss.py"], retriever="llm"),
    ]
    s = summarize(rows, ks=(1,))
    assert s["bm25"]["overall"]["recall@1"] == 1.0
    assert s["llm"]["overall"]["recall@1"] == 0.0


# --- fabrication ---------------------------------------------------------------------


def test_fabrication_counts_paths_absent_from_the_repo():
    rows = [_row("1", "full_path", ["real.py", "invented.py"], repo="r/r")]
    fab = fabrication_rate(rows, {"r/r": ["real.py"]})
    assert fab["paths_named"] == 2
    assert fab["paths_that_exist"] == 1
    assert fab["rate_exists"] == 0.5


def test_fabrication_ignores_repos_with_no_listing():
    rows = [_row("1", "full_path", ["x.py"], repo="unknown/repo")]
    assert fabrication_rate(rows, {})["paths_named"] == 0


# --- coverage ------------------------------------------------------------------------


class _Inst:
    def __init__(self, iid, repo, gold):
        self.instance_id, self.repo, self.gold_files = iid, repo, (gold,)


def test_coverage_counts_a_missing_gold_file():
    insts = [_Inst("1", "r/r", "present.py"), _Inst("2", "r/r", "absent.py")]
    cov = coverage(insts, {"r/r": ["present.py"]})
    assert cov["covered"] == 1 and cov["total"] == 2
    assert cov["rate"] == 0.5
    assert cov["missing"] == [("2", "absent.py")]


def test_coverage_skips_repos_never_fetched():
    cov = coverage([_Inst("1", "nope/nope", "x.py")], {})
    assert cov["total"] == 0 and cov["rate"] == 0.0


# --- candidate filtering -------------------------------------------------------------


def test_source_files_excludes_non_source():
    paths = ["a.py", "b.pyx", "c.png", "docs/d.rst", "e.pxd"]
    assert source_files(paths) == ["a.py", "b.pyx", "e.pxd"]


# --- listing separation --------------------------------------------------------------


def test_per_commit_listings_do_not_overwrite_repo_listings(tmp_path, monkeypatch):
    """A repo listing and a per-commit listing carry the same `repo` field.

    Globbing both into one dict lets the per-commit one win, which would narrow the
    candidate set for every other instance in that repository.
    """
    import gzip
    import json

    import trees

    monkeypatch.setattr(trees, "CACHE", tmp_path)
    for name, paths in [
        ("a__b.json.gz", ["full/a.py", "full/b.py", "full/c.py"]),
        ("at__a__b__0123456789ab.json.gz", ["only/one.py"]),
    ]:
        with gzip.open(tmp_path / name, "wt", encoding="utf-8") as fh:
            json.dump({"repo": "a/b", "ref": "0123456789ab", "paths": paths}, fh)

    assert trees.cached_repos()["a/b"] == ["full/a.py", "full/b.py", "full/c.py"]
    assert trees.commit_listings()[("a/b", "0123456789ab")] == ["only/one.py"]


# --- path resolution -----------------------------------------------------------------


def test_resolve_maps_a_module_path_to_the_repository_path():
    # matplotlib's package lives under lib/, so the model names the import path.
    listing = {"lib/matplotlib/widgets.py", "lib/matplotlib/figure.py"}
    assert resolve(["matplotlib/widgets.py"], listing) == ["lib/matplotlib/widgets.py"]


def test_resolve_leaves_exact_matches_alone():
    listing = {"django/forms/widgets.py"}
    assert resolve(["django/forms/widgets.py"], listing) == ["django/forms/widgets.py"]


def test_resolve_refuses_ambiguous_matches():
    # Two real files end with the same suffix; picking one would be a free guess.
    listing = {"a/util.py", "b/util.py"}
    assert resolve(["util.py"], listing) == ["util.py"]


def test_resolve_requires_a_directory_boundary():
    # "widgets.py" must not resolve to "my_widgets.py".
    assert resolve(["widgets.py"], {"lib/my_widgets.py"}) == ["widgets.py"]


def test_resolve_leaves_fabricated_paths_unresolved():
    assert resolve(["does/not/exist.py"], {"real.py"}) == ["does/not/exist.py"]
