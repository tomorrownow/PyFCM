# Methods used while analyzing fuzzy cognitive models
# Copyright (C) 2018-2021 Corey White and others (see below)

# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.

# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.

# You should have received a copy of the GNU General Public License along with
# this program; if not, see https://www.gnu.org/licenses/gpl-2.0.html

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
def _infer_rule(n, act_vec_old, adjmatrix, infer_rule):
    """
    Infer Rules

    k = Kasko
    mk = Modified Kasko
    r = Rescaled Kasko

    Parameters
       ----------
       n : int
           The number of concepts in the adjacency matrix.
       act_vec_old : numpy.ndarray
           Olde activation vector.
       adjmatrix: numpy.ndarray
           Transposed adjacency matrix of the fuzzy congintive model.
       infer_rule : InferenceRule (Enum)
           Kasko = "k", Modified Kasko = "mk", Rescaled Kasko = "r" (default is mk)

       Returns
           -------
           Activation Vector : numpy.ndarray
    """

    x = np.zeros(n)
    if infer_rule == InferenceRule.K.value:
        x = np.matmul(adjmatrix, act_vec_old)
    elif infer_rule == InferenceRule.MK.value:
        x = act_vec_old + np.matmul(adjmatrix, act_vec_old)
    elif infer_rule == InferenceRule.R.value:
        x = (2 * act_vec_old - np.ones(n)) + np.matmul(
            adjmatrix, (2 * act_vec_old - np.ones(n))
        )
    else:
        raise ValueError(
            "An invalide inference rule was provide. Kasko = k, Modified Kasko = mk, Rescaled Kasko = r"
        )

    return x


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
    if f_type == SquashingFucntion.SIG.value:
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = 1 / (1 + math.exp(-landa * x[i]))
        return x_new

    elif f_type == SquashingFucntion.TANH.value:
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = math.tanh(landa * x[i])
        return x_new

    elif f_type == SquashingFucntion.BIV.value:
        x_new = np.zeros(n)
        for i in range(n):
            if x[i] > 0:
                x_new[i] = 1
            else:
                x_new[i] = 0
        return x_new

    elif f_type == SquashingFucntion.TRIV.value:
        x_new = np.zeros(n)
        for i in range(n):
            if x[i] > 0:
                x_new[i] = 1
            elif x[i] == 0:
                x_new[i] = 0
            else:
                x_new[i] = -1
        return x_new
    else:
        raise ValueError(
            "An invalide squashing function was provide. Please select Sigmoid = 'sig', Hyperbolic Tangent = 'tanh', Bivalent = 'biv', Trivalent = 'triv'"
        )


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
           Sigmoid = "sig", Hyperbolic Tangent = "tanh", Bivalent = "biv", Trivalent = "triv" (default is sig)
       infer_rule : str (optional)
           Kasko = "k", Modified Kasko = "mk", Rescaled Kasko = "r" (default is mk)

       Returns
           -------
           Activation Vector : numpy.ndarray
    """
    act_vec_old = init_vec
    resid = 1
    while resid > 0.00001:
        act_vec_new = np.zeros(n)
        x = _infer_rule(n, act_vec_old, adjmatrix, infer_rule)

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
    Infer the scenario

     k = Kasko
     mk = Modified Kasko
     r = Rescaled Kasko

    Parameters
       ----------
       scenorio_concept: int or list
            Index of scenorio in the activation vector, or list of indexes
       init_vec : numpy.ndarray
           Inital activation vector.
       adjmatrix : numpy.ndarray
           Adjacency matrix of the fuzzy congintive model.
       n : int
           The number of concepts in the adjacency matrix.
       landa : int
           The lambda threshold value used in the squashing fuciton between 0 - 10.
       f_type : str (optional)
           Sigmoid = "sig", Hyperbolic Tangent = "tanh", Bivalent = "biv", Trivalent = "triv" (default is sig)
       infer_rule : str (optional)
           Kasko = "k", Modified Kasko = "mk", Rescaled Kasko = "r" (default is mk)
       change_level : int (optional)
            The activation level of the concept or list of concpects between [-1,1] (default is 1)
       Returns
           -------
           Activation Vector : numpy.ndarray
    """
    act_vec_old = init_vec
    resid = 1
    while resid > 0.00001:
        act_vec_new = np.zeros(n)
        x = _infer_rule(n, act_vec_old, adjmatrix, infer_rule)

        act_vec_new = _transform(x, n, f_type, landa)
        # This is the only differenc inbetween infer_steady and  infer_scenario
        # TODO: Change the data structure being used here to a dictonary
        if isinstance(scenario_concept, list) and isinstance(change_level, list):
            for c in scenario_concept:
                act_vec_new[c] = change_level[c]
        elif isinstance(scenario_concept, int) and isinstance(change_level, int):
            act_vec_new[scenario_concept] = change_level
        else:
            act_vec_new[scenario_concept] = change_level

        resid = max(abs(act_vec_new - act_vec_old))
        act_vec_old = act_vec_new

    return act_vec_new
