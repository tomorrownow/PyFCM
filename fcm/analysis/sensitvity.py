"""
Created on Fri Apr 30 15:51:12 2021

@author: Corey White
         North Carolina State University
         ctwhite@ncsu.edu
"""

import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from fcm.analysis import infer_steady, infer_scenario, reduce_noise


def sensitivity_analysis(
    data,
    columns,
    noise_threshold=0,
    lambda_thres=0,
    principles=None,
    list_of_consepts_to_run=None,
    f_type="sig",
    infer_rule="mk",
):
    """
    Run FMC Sensitivity Analysis

    Parameters
       ----------
       data: numpy.ndarray
            Adjacency matrix of the fuzzy congintive model.
       columns: pandas.core.indexes.base.Index
            List of columns that matches the order of the adjacency matrix.
       noise_threshold: float
            (Not Currently Implemented)

            Sometimes you need to remove the links with significantly low weights to avoid messiness.
            Noise_Threshold is a number in [0,1] which defines a boundary below which all links will be removed from the FCM.
            E.g. Noise_Threshold = 0.15 means that all edges with weight <= 0.15 will be removed from FCM. (default is 0)
       lambda_thres : int (optional)
           The lambda threshold value used in the squashing fuciton between 0 - 10. (default is 0)
       principles : List (optional)
           In each FCM you have some variables which are more important and
           considered to be the main principles of the system. For example, in one FCM my
           main variables are "water pollution" and "CO2 emission". These are the system
           indicators. By defining these principles you would be able to build an additional list
           for keeping track of changes in only these principles not all of the concepts. (default is None)
        list_of_consepts_to_run : List (optional)
            The concepts getting activated during the analysis (default is None).
       f_type : str (optional)
           Sigmoid = "sig", Hyperbolic Tangent = "tanh", Bivalent = "biv", Trivalent = "triv" (default is sig)
       infer_rule : str (optional)
           Kasko = "k", Modified Kasko = "mk", Rescaled Kasko = "r" (default is mk)

       Returns
           -------
           Activation Vector : numpy.ndarray
    """

    # ax = ax or plt.gca()
    n_concepts = len(columns)
    adj_matrix = reduce_noise(data, n_concepts, noise_threshold)

    activation_vec = np.ones(n_concepts)
    concepts_matrix = []

    for i in range(1, n_concepts):
        concepts_matrix.append(columns.values[i])

    G = nx.DiGraph(data)

    # label nodes with variable names
    node_name = {}
    for nod in G.nodes():
        node_name[nod] = columns[nod]

    G = nx.relabel_nodes(G, node_name)

    prin_concepts_index = []
    for nod in node_name.keys():
        if node_name[nod] in principles:
            prin_concepts_index.append(nod)

    steady_state = infer_steady(
        init_vec=activation_vec,
        adjmatrix=adj_matrix.T,
        n=n_concepts,
        landa=lambda_thres,
        f_type=f_type,
        infer_rule=infer_rule,
    )

    # Scenario
    for name in list_of_consepts_to_run:
        # Scenario component name
        sce_con_name = name

        scenario_concept = concepts_matrix.index(sce_con_name)
        change_levels = np.linspace(0, 1, 21)

        change_in_principles = {}
        for pr in prin_concepts_index:
            change_in_principles[pr] = []

        for c in change_levels:

            scenario_state = infer_scenario(
                scenario_concept=scenario_concept,
                init_vec=activation_vec,
                adjmatrix=adj_matrix.T,
                n=n_concepts,
                landa=lambda_thres,
                f_type=f_type,
                infer_rule=infer_rule,
                change_level=c,
            )
            changes = scenario_state - steady_state

            for pr in prin_concepts_index:
                change_in_principles[pr].append(changes[pr])

        fig = plt
        fig.clf()  # Clear figure
        for pr in prin_concepts_index:
            fig.plot(
                change_levels,
                change_in_principles[pr],
                "-o",
                markersize=3,
                label=node_name[pr],
            )
            fig.legend(fontsize=8)
            plt.xlabel("activation state of {}".format(sce_con_name))
            plt.ylabel("State of system principles")

            fig.savefig("{}.png".format(sce_con_name))
        plt.show()
