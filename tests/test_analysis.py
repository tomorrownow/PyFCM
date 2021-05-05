import pyfcm
from pyfcm.analysis.tools import _infer_rule, InferenceRule
import numpy as np

import pytest


def test_infer_rule_kosko(datadir):
    concepts = ["c1", "c2", "c3"]

    adj_matrices = np.array([[1.0, -1.0, 0.0], [0.5, 1.0, 0.0], [1.0, 1.0, -0.5]])

    expected_result = np.array([2.5, 1.0, -0.5])

    n_concepts = len(concepts)
    activation_vec = np.ones(n_concepts)
    result = _infer_rule(
        n_concepts, activation_vec, adj_matrices.T, InferenceRule.K.value
    )
    print(result)
    assert np.array_equal(result, expected_result)


def test_infer_rule_modified_kosko(datadir):
    concepts = ["c1", "c2", "c3"]
    adj_matrices = np.array([[1.0, -1.0, 0.0], [0.5, 1.0, 0.0], [1.0, 1.0, -0.5]])

    expected_result = np.array([3.5, 2.0, 0.5])

    n_concepts = len(concepts)
    activation_vec = np.ones(n_concepts)
    result = _infer_rule(
        n_concepts, activation_vec, adj_matrices.T, InferenceRule.MK.value
    )
    print(result)
    assert np.array_equal(result, expected_result)


def test_infer_rule_rescaled_kosko(datadir):
    concepts = ["c1", "c2", "c3"]
    adj_matrices = np.array([[1.0, -1.0, 0.0], [0.5, 1.0, 0.0], [1.0, 1.0, -0.5]])

    expected_result = np.array([3.5, 2.0, 0.5])

    n_concepts = len(concepts)
    activation_vec = np.ones(n_concepts)
    result = _infer_rule(
        n_concepts, activation_vec, adj_matrices.T, InferenceRule.R.value
    )
    print(result)

    assert np.array_equal(result, expected_result)


def test_infer_rule_failure(datadir):
    with pytest.raises(ValueError):
        concepts = ["c1", "c2", "c3"]
        adj_matrices = np.array([[1.0, -1.0, 0.0], [0.5, 1.0, 0.0], [1.0, 1.0, -0.5]])

        n_concepts = len(concepts)
        activation_vec = np.ones(n_concepts)
        _infer_rule(n_concepts, activation_vec, adj_matrices.T, "dls")
