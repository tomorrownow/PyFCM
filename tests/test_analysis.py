import pyfcm
import pyfcm.analysis as analysis


import pytest

# test_fcm = np.n()

# analysis.tools._transform()


def test_transform():
    import pandas as pd

    file_location = "../tests/data/Adjacency_Matrix_Example.xlsx"
    df = pd.read_excel(file_location, index_col=0).fillna(0)
    df.head(5)
    # analysis.tools._transform()
    assert 1 == 1
