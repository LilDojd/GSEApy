import numpy as np
import pandas as pd

from gseapy.gsea import Prerank


def test_infinite_rankings_take_their_nearest_finite_neighbour():
    rnk = pd.DataFrame(
        {
            "gene": ["A", "B", "C", "D", "E"],
            "score": [np.inf, 2.0, 1.0, -1.0, -np.inf],
        }
    )
    ranking = Prerank(rnk=rnk, gene_sets={"set": ["A", "B"]}, outdir=None).load_ranking()
    assert ranking.to_dict() == {"A": 2.0, "B": 2.0, "C": 1.0, "D": -1.0, "E": -1.0}
