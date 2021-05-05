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
from pyfcm.analysis.tools import infer_steady, infer_scenario


def scenario_analysis(
    data,
    columns,
    noise_threshold=0,
    lambda_thres=0,
    principles=[],
    list_of_consepts_to_run=[],
    function_type="tanh",
    infer_rule="mk",
    change_level={},
    what_to_show="A",
):

    n_concepts = len(columns)
    adjmatrix = np.zeros((n_concepts, n_concepts))

    activation_vec = np.ones(n_concepts)
    concepts_matrix = []

    for i in range(0, n_concepts):
        print(columns.values[i])
        concepts_matrix.append(columns.values[i])

    G = nx.DiGraph(data)

    # label nodes with variable names
    node_name = {}
    for nod in G.nodes():
        node_name[nod] = columns[nod]

    # G = nx.relabel_nodes(G, node_name)

    prin_concepts_index = []
    for nod in node_name.keys():
        if node_name[nod] in principles:
            prin_concepts_index.append(nod)

    print("Principle Concepts: {}".format(prin_concepts_index))
    # ______________________________________________________________________________

    change_level_by_index = {}
    print(concepts_matrix)
    for name in change_level.keys():
        print("name: {}".format(name))
        change_level_by_index[concepts_matrix.index(name)] = change_level[name]
    print(change_level_by_index)

    scenario_concepts = []
    for name in list_of_consepts_to_run:
        Sce_Con_name = name
        scenario_concepts.append(concepts_matrix.index(Sce_Con_name))
    print("scenario_concepts: {}".format(scenario_concepts))

    change_IN_principles = []

    steady_state = infer_steady(
        init_vec=activation_vec,
        adjmatrix=adjmatrix.T,
        n=n_concepts,
        landa=lambda_thres,
        f_type=function_type,
        infer_rule=infer_rule,
    )
    print("steady_state: {}".format(steady_state))
    scenario_state = infer_scenario(
        scenorio_concept=scenario_concepts,
        change_level=change_level_by_index,
        f_type=function_type,
        infer_rule=infer_rule,
        init_vec=activation_vec,
        adjmatrix=adjmatrix,
        n=n_concepts,
        landa=lambda_thres,
    )
    print("scenario_state: {}".format(scenario_state))
    change_IN_ALL = scenario_state - steady_state
    # print(change_IN_ALL)
    for c in scenario_concepts:
        change_IN_ALL[c] = 0

    print("Change in All: {}".format(change_IN_ALL))

    for i in range(len(prin_concepts_index)):
        change_IN_principles.append(change_IN_ALL[prin_concepts_index[i]])

    print("Change in Princples: {}".format(change_IN_principles))

    What_to_show = what_to_show  # input("You want to see the results in All (Type: 'A') or only principles (Type: 'P')?  ")

    if What_to_show == "A":
        changes = change_IN_ALL
        a = 10
        plt.figure(figsize=(a, 5))
        plt.bar(np.arange(len(changes)), changes, align="center", alpha=1, color="g")
        plt.xticks(np.arange(len(changes)), concepts_matrix, rotation="vertical")

    else:
        changes = change_IN_principles
        a = 10
        plt.figure(figsize=(a, 3))
        plt.bar(np.arange(len(changes)), changes, align="center", alpha=1, color="b")
        plt.xticks(np.arange(len(changes)), principles, rotation="vertical")

    plt.title("changes in variables")
    ax = plt.axes()
    ax.xaxis.grid()  # vertical lines
    plt.savefig("Scenario_Results.pdf")
    plt.show()

    changes_dic = {}
    for nod in G.nodes():
        changes_dic[node_name[nod]] = change_IN_ALL[nod]
    print(changes_dic)
    with open("Changes_In_All_Concepts.csv", "w") as f:
        [f.write("{0},{1}\n".format(key, value)) for key, value in changes_dic.items()]
