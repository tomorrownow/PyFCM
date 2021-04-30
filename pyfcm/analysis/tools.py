"""
Created on Fri Apr 30 13:40:00 2021

@author: Corey White
         North Carolina State University
         ctwhite@ncsu.edu
"""

from enum import Enum
import numpy as np
import math


class SquashingFucntion(Enum):
    SIG = "sig"
    TANH = "tanh"
    BIV = "biv"
    TRIV = "triv"


class InferenceRule(Enum):
    K = "k"
    MK = "mk"
    R = "r"


# 'SIG' in SquashingFucntion.__members__


def _transform(x, n, f_type, landa):
    """
    Squashing function applied to FCM

    Parameters
      ----------
      x : numpy.ndarray
          Activation vector after inference rule is applied.
      n : int
          The number of concepts in the adjacency matrix.
      f_type : str
          Sigmoid = "sig", Hyperbolic Tangent = "tanh", Bivalent = "biv", Trivalent = "triv"
      landa : int
          The lambda threshold value used in the squashing fuciton between 0 - 10

      Returns
          -------
          Activation Vector : numpy.ndarray
    """
    if f_type == "sig":
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = 1 / (1 + math.exp(-landa * x[i]))
        return x_new

    if f_type == "tanh":
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = math.tanh(landa * x[i])
        return x_new

    if f_type == "biv":
        x_new = np.zeros(n)
        for i in range(n):
            if x[i] > 0:
                x_new[i] = 1
            else:
                x_new[i] = 0
        return x_new

    if f_type == "triv":
        x_new = np.zeros(n)
        for i in range(n):
            if x[i] > 0:
                x_new[i] = 1
            elif x[i] == 0:
                x_new[i] = 0
            else:
                x_new[i] = -1
        return x_new


def infer_steady(init_vec, adjmatrix, n, landa, f_type="sig", infer_rule="mk"):
    """
     Every concept in the FCM graph has a value Ai that expresses the quantity of its
     corresponding physical value and it is derived by the transformation of the fuzzy values
     assigned by who developed the FCM to numerical values. The value Ai of each concept Ci is
     calculated during each simulation step, computing the influence of other concepts to the
     specific concept by selecting one of the following equations (inference rules).

     k = Kasko
     mk = Modified Kasko
     r = Rescaled Kasko

    Parameters
       ----------
       init_vec : numpy.ndarray
           Inital activation vector.
       adjmatrix : numpy.ndarray
           Adjacency matrix of the fuzzy congintive model.
       n : int
           The number of concepts in the adjacency matrix.
       landa : int
           The lambda threshold value used in the squashing fuciton between 0 - 10.
       f_type : str (optional)
           Sigmoid = "sig", Hyperbolic Tangent = "tanh", Bivalent = "biv", Trivalent = "triv"
       infer_rule : str (optional)
           Kasko = "k", Modified Kasko = "mk", Rescaled Kasko = "r" :Default: "mk"

       Returns
           -------
           Activation Vector : numpy.ndarray
    """
    act_vec_old = init_vec
    resid = 1
    while resid > 0.00001:
        act_vec_new = np.zeros(n)
        x = np.zeros(n)
        if infer_rule == "k":
            x = np.matmul(adjmatrix, act_vec_old)
        if infer_rule == "mk":
            x = act_vec_old + np.matmul(adjmatrix, act_vec_old)
        if infer_rule == "r":
            x = (2 * act_vec_old - np.ones(n)) + np.matmul(
                adjmatrix, (2 * act_vec_old - np.ones(n))
            )

        act_vec_new = _transform(x, n, f_type, landa)
        resid = max(abs(act_vec_new - act_vec_old))
        act_vec_old = act_vec_new

    return act_vec_new


# TODO: Merge remove duplicated code between infer_scenario and infer_steady fuctions
def infer_scenario(
    scenario_concept,
    init_vec,
    adjmatrix,
    n,
    landa,
    f_type="sig",
    infer_rule="mk",
    change_level=1,
):
    """
    Infer teh scenario

     k = Kasko
     mk = Modified Kasko
     r = Rescaled Kasko

    Parameters
       ----------
       init_vec : numpy.ndarray
           Inital activation vector.
       adjmatrix : numpy.ndarray
           Adjacency matrix of the fuzzy congintive model.
       n : int
           The number of concepts in the adjacency matrix.
       landa : int
           The lambda threshold value used in the squashing fuciton between 0 - 10.
       f_type : str (optional)
           Sigmoid = "sig", Hyperbolic Tangent = "tanh", Bivalent = "biv", Trivalent = "triv"
       infer_rule : str (optional)
           Kasko = "k", Modified Kasko = "mk", Rescaled Kasko = "r" :Default: "mk"

       Returns
           -------
           Activation Vector : numpy.ndarray
    """
    act_vec_old = init_vec
    resid = 1
    while resid > 0.00001:
        act_vec_new = np.zeros(n)
        x = np.zeros(n)
        if infer_rule == "k":
            x = np.matmul(adjmatrix, act_vec_old)
        if infer_rule == "mk":
            x = act_vec_old + np.matmul(adjmatrix, act_vec_old)
        if infer_rule == "r":
            x = (2 * act_vec_old - np.ones(n)) + np.matmul(
                adjmatrix, (2 * act_vec_old - np.ones(n))
            )

        act_vec_new = _transform(x, n, f_type, landa)
        # This is the only differenc inbetween infer_steady and  infer_scenario
        act_vec_new[scenario_concept] = change_level
        resid = max(abs(act_vec_new - act_vec_old))
        act_vec_old = act_vec_new

    return act_vec_new
