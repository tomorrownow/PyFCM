"""
Created on Mon May 03 9:12:22 2021

@author: Corey White
         North Carolina State University
         ctwhite@ncsu.edu
"""

import matplotlib.pyplot as plt
import xlrd
import numpy as np
import math
import networkx as nx
from .tools import infer_steady, infer_scenario

# file_location = init.file_location
# workbook = xlrd.open_workbook(file_location)
# sheet = workbook.sheet_by_index(0)

# n_concepts = sheet.nrows - 1

# Adj_matrix = np.zeros((n_concepts, n_concepts))
# activation_vec = np.ones(n_concepts)
# node_name = {}
# # ______________________________________________________________________________

# Noise_Threshold = init.Noise_Threshold

# for i in range(1, n_concepts + 1):
#     for j in range(1, n_concepts + 1):
#         if abs(sheet.cell_value(i, j)) <= Noise_Threshold:
#             Adj_matrix[i - 1, j - 1] = 0
#         else:
#             Adj_matrix[i - 1, j - 1] = sheet.cell_value(i, j)

# # ______________________________________________________________________________

# # Generating a python NetworkX graph using our Adjacancy Matrix
# # Concepts_matrix is a list to keep concept names
# Concepts_matrix = []
# for i in range(1, n_concepts + 1):
#     Concepts_matrix.append(sheet.cell_value(0, i))

# G = nx.DiGraph(Adj_matrix)
# for nod in G.nodes():
#     node_name[nod] = sheet.cell_value(nod + 1, 0)
# # ______________________________________________________________________________

# Principles = init.Principles


# prin_concepts_index = []
# for nod in node_name.keys():
#     if node_name[nod] in Principles:
#         prin_concepts_index.append(nod)


# list_of_consepts_to_run = init.list_of_consepts_to_run

# # ______________________________________________________________________________
# function_type = init.function_type
# infer_rule = init.infer_rule
# change_level = init.change_level

# change_level_by_index = {}
# for name in change_level.keys():
#     change_level_by_index[Concepts_matrix.index(name)] = change_level[name]

# Scenario_concepts = []
# for name in list_of_consepts_to_run:
#     Sce_Con_name = name
#     Scenario_concepts.append(Concepts_matrix.index(Sce_Con_name))


# change_IN_PRINCIPLES = []


# SteadyState = infer_steady(f_type=function_type, infer_rule=infer_rule)
# ScenarioState = infer_scenario(
#     Scenario_concepts,
#     change_level_by_index,
#     f_type=function_type,
#     infer_rule=infer_rule,
# )
# change_IN_ALL = ScenarioState - SteadyState

# for c in Scenario_concepts:
#     change_IN_ALL[c] = 0

# for i in range(len(prin_concepts_index)):
#     change_IN_PRINCIPLES.append(change_IN_ALL[prin_concepts_index[i]])


# What_to_show = input(
#     "You want to see the results in All (Type: 'A') or only Principles (Type: 'P')?  "
# )

# if What_to_show == "A":
#     changes = change_IN_ALL
#     a = 50
#     plt.figure(figsize=(a, 5))
#     plt.bar(np.arange(len(changes)), changes, align="center", alpha=1, color="g")
#     plt.xticks(np.arange(len(changes)), Concepts_matrix, rotation="vertical")

# else:
#     changes = change_IN_PRINCIPLES
#     a = 10
#     plt.figure(figsize=(a, 3))
#     plt.bar(np.arange(len(changes)), changes, align="center", alpha=1, color="b")
#     plt.xticks(np.arange(len(changes)), Principles, rotation="vertical")


# # plt.ylim(0,1)
# # plt.ylabel('changes')
# plt.title("changes in variables")
# ax = plt.axes()
# ax.xaxis.grid()  # vertical lines
# plt.savefig("Scenario_Results.pdf")
# plt.show()


# changes_dic = {}
# for nod in G.nodes():
#     changes_dic[node_name[nod]] = change_IN_ALL[nod]

# with open("Changes_In_All_Concepts.csv", "w") as f:
#     [f.write("{0},{1}\n".format(key, value)) for key, value in changes_dic.items()]
