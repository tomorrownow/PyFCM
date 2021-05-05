# import os
import pandas as pd

# import seaborn as sns

# class FuzzyCognitiveModel:
#     def __init__(self, name, fcm_data, concept_map=None):
#         self.name = name
#         self.group = group
#         self.data = fcm_data
#         self.concept_map = concept_map

# class SocialCognitiveModel:
#     def __init__(self, name, fcm_list):
#         self.name = name

#     def accumualtion_curve(df, ax=None, new_variables=False, n=500):
#         assert False, "TODO: finish me"


# df_graph_stats_data = []
# cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True)
# All_ADJs = []
# all_data_frames = pd.DataFrame(
#     columns=df_concepts["code"].unique(), index=df_concepts["code"].unique()
# ).fillna(0)
# for root, dirs, files in os.walk(data_location, topdown=False):
#     for name in files:
#         if "allFCMs" not in name and name != ".DS_Store":
#             file_location = os.path.join(root, name)
#             participant_organization = name.split("_")[-1].split(".")[0]
#             participant_number = name.split("_")[0]
#             df = pd.read_excel(file_location, index_col=0).fillna(0)

#             df.columns = df.columns.map(concept_map)
#             df.index = df.index.map(concept_map)
#             print(
#                 "FCMs",
#                 "%sFCM - %s - %d" % (name, participant_organization, len(All_ADJs)),
#             )

#             print(all_data_frames.columns)
#             take_not_zero = lambda s1, s2: s1 if s1.sum() != 0 else s2

#             df_copy = all_data_frames.combine(
#                 df, take_not_zero, fill_value=0, overwrite=True
#             )

#             All_ADJs.append(
#                 df_copy.loc[all_data_frames.columns, all_data_frames.columns].values
#             )

#             fig, (ax, ax1) = plt.subplots(1, 2, figsize=(20, 10))
#             plt.suptitle(
#                 "FCM - %s %s" % (participant_organization, participant_number),
#                 fontsize=14,
#             )
#             ax.set_title("Adjacency Matrix", fontsize=12)

#             sns.heatmap(df, annot=True, linewidths=0.5, ax=ax, center=0, cmap=cmap)
#             graph_stats = generate_map(df.values, df.columns, ax1)
#             ax1.set_title("Fuzzy Cognitive Map", fontsize=12)
#             graph_stats["type"] = participant_organization
#             plt.tight_layout()
#             save_path = os.path.join(
#                 save_location,
#                 "FCMs",
#                 "FCM - %s - %s" % (participant_organization, participant_number),
#             )
#             plt.savefig(save_path)
#             df_graph_stats_data.append(graph_stats)

# df_graph_stats = pd.DataFrame(df_graph_stats_data)


def load_csv(file_path, concept_map=None):
    """
    Loads a  csv file as a fuzzy cognitive map.

    Parameters
       ----------
       file_path : str
           The file path to the csv location.
       concept_map : dict
           A mapping from user defined variables to a standardized set of variables.

       Returns
           -------
           FCM : DataFrame
    """
    df = pd.read_csv(file_path, index_col=0).fillna(0)
    if concept_map:
        df.columns = df.columns.map(concept_map)
        df.index = df.index.map(concept_map)
    return df


def load_xlsx(file_path, concept_map=None):
    """
    Loads a xlsx file as a fuzzy cognitive map.

    Parameters
       ----------
       file_path : str
           The file path to the csv location.
       concept_map : dict
           A mapping from user defined variables to a standardized set of variables.

       Returns
           -------
           FCM : DataFrame
    """
    df = pd.read_excel(file_path, index_col=0).fillna(0)
    if concept_map:
        df.columns = df.columns.map(concept_map)
        df.index = df.index.map(concept_map)
    return df
