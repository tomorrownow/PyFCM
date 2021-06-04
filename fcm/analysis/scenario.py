"""
Created on Mon May 03 9:12:22 2021

@author: Corey White
         North Carolina State University
         ctwhite@ncsu.edu
"""

import matplotlib.pyplot as plt

# import xlrd
import numpy as np
import math
import networkx as nx
from fcm.analysis.tools import infer_steady, infer_scenario, reduce_noise


def scenario_analysis(
    data,
    columns,
    scenarios,
    noise_threshold=0,
    lambda_thres=0,
    principles=None,
    f_type="tanh",
    infer_rule="mk",
):
    """
    Run FMC scenario by asking 'what if' questions

    Parameters
    ----------
    data: numpy.ndarray
            Adjacency matrix of the fuzzy congintive model.
    columns: pandas.core.indexes.base.Index
            List of columns that matches the order of the adjacency matrix.
    noise_threshold: float
            Sometimes you need to remove the links with significantly low weights to avoid messiness.
            Noise_Threshold is a number in [0,1] which defines a boundary below which all links will be removed from the FCM.
            E.g. Noise_Threshold = 0.15 means that all edges with weight <= 0.15 will be removed from FCM. (default is 0)
    lambda_thres : int (optional)
        The lambda threshold value used in the squashing fuciton between 0 - 10. (default is 0)
    principles : List
        In each FCM you have some variables which are more important and
        considered to be the main principles of the system. For example, in one FCM my
        main variables are "water pollution" and "CO2 emission". These are the system
        indicators. By defining these principles you would be able to build an additional list
        for keeping track of changes in only these principles not all of the concepts. (default is None)
    scenarios: Dict
        Dictionary of which variables you want to activate during the scenario using the concept as the key and activation level as the value.
        {Variable: Activation Level [-1,1]} for example {'c1': 1} or {'c1': -1}
    f_type : str (optional)
        Sigmoid = "sig", Hyperbolic Tangent = "tanh", Bivalent = "biv", Trivalent = "triv" (default is sig)
    infer_rule : str (optional)
        Kasko = "k", Modified Kasko = "mk", Rescaled Kasko = "r" (default is mk)

    Returns
        -------
        Activation Vector : numpy.ndarray
    """

    n_concepts = len(columns)

    adjmatrix = reduce_noise(data, n_concepts, noise_threshold)
    activation_vec = np.ones(n_concepts)
    concepts_matrix = []

    for i in range(0, n_concepts):
        concepts_matrix.append(columns.values[i])

    G = nx.DiGraph(data)

    # label nodes with variable names
    node_name = {}
    for nod in G.nodes():
        node_name[nod] = columns[nod]

    prin_concepts_index = []
    for nod in node_name.keys():
        if node_name[nod] in principles:
            prin_concepts_index.append(nod)

    # Generate a list of indexes for varibales being ran in the scenario that match their location in the adjacency matrix
    change_level_by_index = {
        concepts_matrix.index(concept): value for concept, value in scenarios.items()
    }
    scenario_concepts = list(change_level_by_index.keys())

    steady_state = infer_steady(
        init_vec=activation_vec,
        adjmatrix=adjmatrix.T,
        n=n_concepts,
        landa=lambda_thres,
        f_type=f_type,
        infer_rule=infer_rule,
    )

    scenario_state = infer_scenario(
        scenario_concept=scenario_concepts,
        change_level=change_level_by_index,
        f_type=f_type,
        infer_rule=infer_rule,
        init_vec=activation_vec,
        adjmatrix=adjmatrix.T,
        n=n_concepts,
        landa=lambda_thres,
    )

    # Records changes to the priciple concepts
    change_in_principles = []

    change_in_all = scenario_state - steady_state

    for c in scenario_concepts:
        change_in_all[c] = 0

    for i in range(len(prin_concepts_index)):
        change_in_principles.append(change_in_all[prin_concepts_index[i]])

    changes_dic = {}
    for nod in G.nodes():
        changes_dic[node_name[nod]] = change_in_all[nod]

    return changes_dic
