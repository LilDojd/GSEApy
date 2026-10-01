import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from gseapy import dotplot, enrich, gsva, heatmap, prerank  # noqa: E402
from gseapy.frames import to_pandas, to_polars  # noqa: E402

pl = pytest.importorskip("polars")

RNK = "tests/data/edb/gsea_data.gsea_data.rnk"
GMT = "tests/data/genes.gmt"
GSVA_EXPR = "tests/data/expr.gsva.csv"
GSVA_GMT = "tests/data/geneset.gsva.gmt"


@pytest.fixture
def ranking() -> pd.DataFrame:
    return pd.read_csv(RNK, sep="\t", header=None, names=["gene", "score"])


def _prerank(rnk):
    return prerank(rnk, GMT, permutation_num=20, seed=7, threads=1, outdir=None, no_plot=True)


def test_prerank_accepts_polars_like_pandas(ranking):
    expected = _prerank(ranking).res2d
    actual = _prerank(pl.from_dict({column: ranking[column].tolist() for column in ranking}))
    pd.testing.assert_frame_equal(actual.res2d, expected)


def test_prerank_accepts_a_lazy_frame(ranking):
    expected = _prerank(ranking).res2d
    lazy = pl.LazyFrame({column: ranking[column].tolist() for column in ranking})
    pd.testing.assert_frame_equal(_prerank(lazy).res2d, expected)


def test_prerank_rejects_a_polars_series(ranking):
    with pytest.raises(TypeError, match="no gene names"):
        _prerank(pl.Series("score", ranking["score"].tolist()))


def test_results_convert_to_polars(ranking):
    result = _prerank(ranking)
    frame = result.to_polars()
    assert frame.columns == [str(column) for column in result.res2d.columns]
    assert frame.height == len(result.res2d)
    assert frame["Term"].to_list() == result.res2d["Term"].to_list()
    assert frame["NES"].to_list() == result.res2d["NES"].tolist()


def test_gsva_accepts_polars():
    expr = pd.read_csv(GSVA_EXPR, index_col=0)
    expected = gsva(expr, GSVA_GMT, outdir=None).res2d
    polars_expr = pl.DataFrame(
        {"gene": expr.index.tolist(), **{str(column): expr[column].to_numpy() for column in expr}}
    )
    pd.testing.assert_frame_equal(gsva(polars_expr, GSVA_GMT, outdir=None).res2d, expected)


def test_enrich_accepts_a_polars_series():
    genes = pd.read_csv("tests/data/gene_list.txt", header=None).iloc[:, 0].tolist()
    expected = enrich(genes, GMT, outdir=None, cutoff=1.0, no_plot=True).res2d
    actual = enrich(pl.Series("gene", genes), GMT, outdir=None, cutoff=1.0, no_plot=True)
    pd.testing.assert_frame_equal(actual.res2d, expected)
    assert actual.to_polars().height == len(expected)


def test_plots_accept_polars(ranking):
    results = _prerank(ranking).to_polars()
    assert dotplot(results, column="FDR q-val", cutoff=1.0, top_term=5) is not None
    expr = pl.DataFrame({"gene": ["A", "B", "C"], "s1": [1.0, 2.0, 3.0], "s2": [3.0, 1.0, 2.0]})
    assert heatmap(expr, z_score=0) is not None


def test_to_polars_round_trips_missing_values():
    frame = pd.DataFrame(
        {
            "term": pd.Series(["a", None, "c"], dtype="str"),
            "score": [1.0, float("nan"), 3.0],
            "count": pd.Series([1, None, 3], dtype="Int64"),
        }
    )
    converted = to_polars(frame)
    assert converted["term"].to_list() == ["a", None, "c"]
    assert converted["count"].to_list() == [1, None, 3]
    assert to_pandas(converted)["score"].isna().tolist() == [False, True, False]
