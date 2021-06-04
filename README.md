# FCM

[![Tests](https://github.com/tomorrownow/PyFCM/actions/workflows/template.yaml/badge.svg)](https://github.com/tomorrownow/PyFCM/actions/workflows/template.yaml)

This is a refactoring and expansion of the original PyFCM [https://github.com/payamaminpour/PyFCM/](https://github.com/payamaminpour/PyFCM) developed in order to fuctionally run FCM analysis.

This is a set of Python scripts written by Payam Aminpour for Researchers who want to run more robust analysis with Fuzzy Cognitive Maps. In this package we offer FCM aggregation techniques, FCM Clustering, FCM Scenario Analysis, FCM Sensitivity and Uncertainty Analysis, Credibility test, and data visualization.

## Load Data

```python
# Load your csv or xsl data
df = load_csv(shared_datadir / "test_adjacency_matrix.csv")
```

## Run a Scenario

```python
# Load your csv or xsl data
df = load_csv(shared_datadir / "test_adjacency_matrix.csv")

# Run a scenario on one or many components in the FCM
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

# Output
result = {
    "c1": 0.0,
    "c2": 0.8771805720335079,
    "c3": 0.7615984906926053,
    "c4": 0.0,
    "c5": -0.36340194800116987,
}
```


## Development

Activate a new python environment

```bash
source venv/bin/activate
```

### Black and Flake8

```bash
pre-commit run --all-files
```

### Testing

Pyenv is used for testing.

```bash
pytest
```
