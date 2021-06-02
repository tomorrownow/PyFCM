import pyfcm
from pyfcm.load import load_csv
from pandas.core.frame import DataFrame

import pytest


def test_load_csv(shared_datadir):
    df = load_csv(shared_datadir / "test_adjacency_matrix.csv")
    assert isinstance(df, DataFrame)
