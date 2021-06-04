import fcm
from fcm.analysis.tools import (
    _infer_rule,
    InferenceRule,
    reduce_noise,
    _transform,
    SquashingFucntion,
)
import numpy as np

import pytest


# Inference Rule Tests
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


# Transform Function Tests
def test_transform_sig():
    concepts = ["c1", "c2", "c3"]
    input_vector = np.array([3.5, 2.0, 0.5])
    n_concepts = len(concepts)
    expected_result = np.array([0.5, 0.5, 0.5])
    result = _transform(
        act_vect=input_vector, n=n_concepts, f_type=SquashingFucntion.SIG.value, landa=0
    )
    print(result)
    assert np.array_equal(result, expected_result)


def test_transform_tanh():
    concepts = ["c1", "c2", "c3"]
    input_vector = np.array([3.5, 2.0, 0.5])
    n_concepts = len(concepts)
    expected_result = np.array([0.0, 0.0, 0.0])
    result = _transform(
        act_vect=input_vector,
        n=n_concepts,
        f_type=SquashingFucntion.TANH.value,
        landa=0,
    )
    print(result)
    assert np.array_equal(result, expected_result)


def test_transform_biv():
    concepts = ["c1", "c2", "c3"]
    input_vector = np.array([3.5, 2.0, 0.5])
    n_concepts = len(concepts)
    expected_result = np.array([1.0, 1.0, 1.0])
    result = _transform(
        act_vect=input_vector, n=n_concepts, f_type=SquashingFucntion.BIV.value, landa=0
    )
    print(result)
    assert np.array_equal(result, expected_result)


def test_transform_triv():
    concepts = ["c1", "c2", "c3"]
    input_vector = np.array([3.5, 2.0, 0.5])
    n_concepts = len(concepts)
    expected_result = np.array([1.0, 1.0, 1.0])
    result = _transform(
        act_vect=input_vector,
        n=n_concepts,
        f_type=SquashingFucntion.TRIV.value,
        landa=0,
    )
    print(result)
    assert np.array_equal(result, expected_result)


# Reduce Noise Tests
def test_reduce_noise(datadir):
    concepts = ["c1", "c2", "c3"]
    adj_matrices = np.array([[1.0, -1.0, 0.0], [0.5, 1.0, 0.0], [1.0, 1.0, -0.5]])
    n_concepts = len(concepts)
    expected_result = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, -0.0]])

    result = reduce_noise(adj_matrices, n_concepts, 0.5)
    assert np.array_equal(result, expected_result)
