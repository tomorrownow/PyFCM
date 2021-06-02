import pyfcm
from pyfcm.load import load_csv
from pyfcm.analysis import scenario
import numpy as np

import pytest


def test_scenario(shared_datadir):
    df = load_csv(shared_datadir / "test_adjacency_matrix.csv")
    result = scenario.scenario_analysis(
        data=df.values,
        columns=df.columns,
        scenarios={"c1": 1},
        noise_threshold=0,
        lambda_thres=1,
        principles=[],
        f_type="tanh",
        infer_rule="k",
    )
    print(result)
    assert result == {
        "c1": 0.0,
        "c2": 0.8771805720335079,
        "c3": 0.7615984906926053,
        "c4": 0.0,
        "c5": -0.36340194800116987,
    }
