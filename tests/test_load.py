import fcm
from fcm.load import load_csv, load_xlsx
from pandas.core.frame import DataFrame

import pytest


def test_load_csv(shared_datadir):
    df = load_csv(shared_datadir / "test_adjacency_matrix.csv")
    assert isinstance(df, DataFrame)


def test_load_xlsx(shared_datadir):
    df = load_xlsx(shared_datadir / "Adjacency_Matrix_Example.xlsx")
    assert isinstance(df, DataFrame)
