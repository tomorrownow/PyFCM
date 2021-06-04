from pyfcm.load import load_csv
from pyfcm.analysis import scenario


def test_scenario_tanh_k_one_variable(shared_datadir):
    df = load_csv(shared_datadir / "test_adjacency_matrix.csv")
    result = scenario.scenario_analysis(
        data=df.values,
        columns=df.columns,
        scenarios={"c1": 1},
        noise_threshold=0.0,
        lambda_thres=1,
        principles=[],
        f_type="tanh",
        infer_rule="k",
    )

    assert result == {
        "c1": 0.0,
        "c2": 0.8771805720335079,
        "c3": 0.7615984906926053,
        "c4": 0.0,
        "c5": -0.36340194800116987,
    }


def test_scenario_sig_k_variable(shared_datadir):
    df = load_csv(shared_datadir / "test_adjacency_matrix.csv")
    result = scenario.scenario_analysis(
        data=df.values,
        columns=df.columns,
        scenarios={"c1": 1},
        noise_threshold=0.0,
        lambda_thres=1,
        principles=[],
        f_type="sig",
        infer_rule="k",
    )
    print(result)
    assert result == {
        "c1": 0.0,
        "c2": 0.09222567649793934,
        "c3": 0.07798569160599089,
        "c4": 0.0,
        "c5": -0.00969006709092668,
    }


def test_scenario_triv_mk_variable(shared_datadir):
    df = load_csv(shared_datadir / "test_adjacency_matrix.csv")
    result = scenario.scenario_analysis(
        data=df.values,
        columns=df.columns,
        scenarios={"c1": 1},
        noise_threshold=0.0,
        lambda_thres=1,
        principles=[],
        f_type="triv",
        infer_rule="mk",
    )
    print(result)
    assert result == {"c1": 0.0, "c2": 0.0, "c3": 0.0, "c4": 0.0, "c5": 0.0}


def test_scenario_biv_r_variable(shared_datadir):
    df = load_csv(shared_datadir / "test_adjacency_matrix.csv")
    result = scenario.scenario_analysis(
        data=df.values,
        columns=df.columns,
        scenarios={"c1": 1},
        noise_threshold=0.0,
        lambda_thres=1,
        principles=[],
        f_type="triv",
        infer_rule="r",
    )
    print(result)
    assert result == {"c1": 0.0, "c2": 0.0, "c3": 0.0, "c4": 0.0, "c5": 0.0}
