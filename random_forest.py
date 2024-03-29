# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# Number of trees in random forest
from typing import OrderedDict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, ShuffleSplit, LeaveOneOut
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from modules.extract_csv_data import get_merged_data, base_path, get_drug_data_from_csv, get_condition_data_from_csv, get_procedure_data_from_csv, get_encounter_data_from_csv
from modules.put_to_csv import put_to_csv, put_np_array_to_csv, put_unconstructed_np_array_to_csv
from modules.utility_functions import update_nan_most_frequent_category, update_nan_most_frequent_category_tuple, remove_nan
from pathlib import Path
import sys
import ast
import datetime
from collections import Counter
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt

print("start time: ", datetime.datetime.now())

# # ======================================================================
# # ----------------------------------------------------------------------
# # Extracting the requried data
# # ----------------------------------------------------------------------
# data = get_merged_data()
# medications_df = data['medications_df']
# encounters_df = data['encounters_df']
# procedures_df = data['procedures_df']
# conditions_df = data['conditions_df']
# patients_df = data['patients_df']
# # observations_df = data['observations_df']


# # # Merging encounter with patients to get more information
# merged_df = encounters_df.merge(
#     patients_df, left_on='patient_id', right_on='patient_id', how='inner')

# # Merging encounter with medication to get the target variable
# merged_df = merged_df.merge(
#     medications_df, left_on='encounter_id', right_on='encounter_id', how='inner')

# print("#2")
# print("merged_df: ", list(merged_df.columns.values))

# # First hundred/thousand entries
# # merged_df = merged_df.head(10000)

# # sys.exit(0)

# # # ----------------------------------------------------------------------
# # # Mapping the encounter, procedure and conditions with the medications.
# merged_df_1 = merged_df.merge(
#     procedures_df, left_on='encounter_id', right_on='encounter_id', how='inner')
# print("#2")
# print("merged_df_1 procedure: ", list(merged_df_1.columns.values))

# merged_df_2 = merged_df.merge(
#     conditions_df, left_on='encounter_id', right_on='encounter_id', how='inner')
# print("#3")
# print("merged_df_2 condition: ", list(merged_df_2.columns.values))

# subset_merged_df_1 = merged_df_1[[
#     'encounter_id', 'encounter_type_code', 'medication_code']]
# subset_merged_df_1 = subset_merged_df_1.drop_duplicates(
#     subset=['encounter_id', 'encounter_type_code', 'medication_code'], keep='first')

# encounter_id_set = subset_merged_df_1.values.tolist()

# encounter_id_new_map = []

# index = 0
# total_length = len(encounter_id_set)
# print("started iterations..")

# for encounter_id, encounter_type_code, medication_code in encounter_id_set:

#     index += 1
#     if index % 500 == 0:
#         print(f"processed {index} of {total_length}")

#     temp_df = merged_df_1[(merged_df_1['encounter_id'] ==
#                           encounter_id) & (merged_df_1['medication_code'] == medication_code)]
#     temp_procedures = []
#     temp_procedures_arrays = []

#     for index_2, row in temp_df.iterrows():
#         temp_procedures.append(row['procedure_type_code'])
#         temp_procedures_arrays += row['new_procedure_type_code']

#     encounter_id_new_map.append([encounter_id, temp_df['year'].values[0],
#                                  temp_df['patient_id'].values[0],
#                                  encounter_type_code,
#                                  temp_df['encounter_description'].values[0],
#                                  temp_df['new_encounter_type_code'].values[0],
#                                  medication_code,
#                                  temp_df['medication'].values[0],
#                                  temp_df['new_medication_code'].values[0],
#                                  temp_df['dose_form_code'].values[0],
#                                  temp_procedures,
#                                  temp_df['procedure_description'].values[0],
#                                  temp_procedures_arrays])

# merged_df_1 = pd.DataFrame(encounter_id_new_map, columns=['encounter_id', 'year', 'patient_id',
#                                                           'encounter_type_code', 'encounter_description', 'new_encounter_type_code', 'medication_code', 'medication',
#                                                           'new_medication_code', 'dose_form_code', 'procedure_type_code', 'procedure_description', 'new_procedure_type_code'])

# merged_df_1.set_index('encounter_id')
# put_to_csv(base_path, merged_df_1, "merged_df_1_post.csv")
# merged_df_1.to_pickle('merged_df_1_post_pickle.pkl')

# # ------------------------------

# put_to_csv(base_path, merged_df_2, "merged_df_2_pre.csv")

# subset_merged_df_2 = merged_df_2[[
#     'encounter_id', 'encounter_type_code', 'medication_code']]
# subset_merged_df_2 = subset_merged_df_2.drop_duplicates(
#     subset=['encounter_id', 'encounter_type_code', 'medication_code'], keep='first')

# encounter_id_set = subset_merged_df_2.values.tolist()

# encounter_id_new_map = []

# index = 0
# total_length = len(encounter_id_set)
# print("started iterations..")
# for encounter_id, encounter_type_code, medication_code in encounter_id_set:
#     index += 1
#     if index % 500 == 0:
#         print(f"processed {index} of {total_length}")

#     temp_df = merged_df_2[(merged_df_2['encounter_id'] ==
#                           encounter_id) & (merged_df_2['medication_code'] == medication_code)]
#     temp_conditions = []
#     temp_conditions_arrays = []

#     for index_2, row in temp_df.iterrows():
#         temp_conditions.append(row['condition_type_code'])
#         temp_conditions_arrays += row['new_condition_type_code']

#     encounter_id_new_map.append([encounter_id, temp_df['year'].values[0],
#                                  temp_df['patient_id'].values[0],
#                                  encounter_type_code,
#                                  temp_df['encounter_description'].values[0],
#                                  temp_df['new_encounter_type_code'].values[0],
#                                  medication_code,
#                                  temp_df['medication'].values[0],
#                                  temp_df['new_medication_code'].values[0],
#                                  temp_df['dose_form_code'].values[0],
#                                  temp_conditions,
#                                  temp_df['condition_description'].values[0],
#                                  temp_conditions_arrays])

# merged_df_2 = pd.DataFrame(encounter_id_new_map, columns=['encounter_id', 'year', 'patient_id',
#                                                           'encounter_type_code', 'encounter_description', 'new_encounter_type_code', 'medication_code', 'medication',
#                                                           'new_medication_code', 'dose_form_code', 'condition_type_code', 'condition_description', 'new_condition_type_code'])

# merged_df_2.set_index('encounter_id')
# put_to_csv(base_path, merged_df_2, "merged_df_2_post.csv")

# merged_df_2.to_pickle('merged_df_2_post_pickle.pkl')

# # ======================================================================
# # ----------------------------------------------------------------------
# #  Dropping the duplicates
# # ----------------------------------------------------------------------
# #

# merged_df_1 = pd.read_pickle('merged_df_1_post_pickle.pkl')
# merged_df_2 = pd.read_pickle('merged_df_2_post_pickle.pkl')

# merged_df_1 = merged_df_1.drop_duplicates(
#     subset=['encounter_id', 'medication_code'], keep='first')

# merged_df_2 = merged_df_2.drop_duplicates(
#     subset=['encounter_id', 'medication_code'], keep='first')
# # put_to_csv(base_path, merged_df_1, "merged_df_1.csv")
# # put_to_csv(base_path, merged_df_2, "merged_df_2.csv")

# # # ======================================================================
# # # ----------------------------------------------------------------------
# # # Merging the dataframes
# # # ----------------------------------------------------------------------

# # merged_df_1.to_pickle('tempdf1.pkl')
# # merged_df_2.to_pickle('tempdf2.pkl')

# # # merged_df_1 = pd.read_pickle('tempdf1.pkl')
# # # merged_df_2 = pd.read_pickle('tempdf2.pkl')

# merged_df = merged_df_1.merge(merged_df_2, on=['encounter_id', 'patient_id', 'encounter_type_code', 'encounter_description',
#                                                'medication_code', 'medication', 'dose_form_code'], how='left')

# put_to_csv(base_path, merged_df, "temp1.csv")
# merged_df.to_pickle('merged_df_heatmap.pkl')
# merged_df =  pd.read_pickle('merged_df_heatmap.pkl')

# sys.exit(0)

# print("after merging the conditions and procedures: ",
#       list(merged_df.columns.values))

# # ----------------------------------------------------------------------
# # Getting the heatmap
# heatmap_df = merged_df[[
#     'medication_code',
#     'encounter_type_code',
#     'condition_type_code',
#     'procedure_type_code',
#     'race_x',
#     'ethnicity_x',
#     'gender_x',
#     'year_x'
# ]]
# print("getting the heatmap")
# print("heatmap_df: ", list(heatmap_df.columns.values))
# print(heatmap_df)
# sns.heatmap(heatmap_df, annot=True)


def column(matrix, i):
    return [row[i] for row in matrix]


# For encounters
# plot_df = merged_df.groupby(
#     ['encounter_type_code', 'encounter_description']).size().reset_index(name='counts')
# plot_df = plot_df.sort_values(by='counts', ascending=False)
# put_to_csv(base_path, plot_df, "plot_df_encounter.csv")
# plot_df_list = plot_df.values.tolist()
# print("plot df: ", plot_df_list)
# x = column(plot_df_list[:8], 1)
# y = column(plot_df_list[:8], 2)
# plt.title("Top encounters of a hospital")
# plt.xlabel('Encounter Type')
# # plt.ylabel('RMSE')
# plt.bar(x, y)
# plt.show()

# # ----------------------------------------------------------------------

# # For medication codes
# plot_df = merged_df.groupby(
#     ['medication_code', 'medication']).size().reset_index(name='counts')
# plot_df = plot_df.sort_values(by='counts', ascending=False)
# put_to_csv(base_path, plot_df, "plot_df_medication.csv")
# plot_df_list = plot_df.values.tolist()
# print("plot df: ", plot_df_list)
# x = column(plot_df_list[:8], 1)
# y = column(plot_df_list[:8], 2)
# plt.title("Top medications of a hospital")
# plt.xlabel('Medication Type')
# # plt.ylabel('RMSE')
# # plt.bar(x, y)
# # plt.show()

# change_count = 0
# for _, _, count in plot_df.values.tolist():
#     if count > 2500:
#         change_count += 1

# # the full dataframe
# df = pd.DataFrame(
#     data={'medications': column(
#         plot_df.values.tolist()[0:change_count], 1), 'value': column(
#         plot_df.values.tolist()[0:change_count], 2)},
# ).sort_values('value', ascending=False)

# # others
# new_row = pd.DataFrame(data={
#     'medications': ['others'],
#     'value': [sum(column(plot_df.values.tolist()[change_count:], 2))]
# })

# # combining top 5 with others
# df = pd.concat([df, new_row])

# print("df now: ", df)
# # plotting -- for comparison left all countries and right
# # the others combined
# # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
# df.plot(kind='pie', y='value', labels=df['medications'], legend=False)
# plt.title('All Medications')
# plt.ylabel('')
# plt.show()

# # ----------------------------------------------------------------------

# # For condtion codes
# plot_df = merged_df.groupby(
#     ['condition_type_code', 'condition_description']).size().reset_index(name='counts')
# plot_df = plot_df.sort_values(by='counts', ascending=False)
# put_to_csv(base_path, plot_df, "plot_df_condition.csv")
# plot_df_list = plot_df.values.tolist()
# print("plot df: ", plot_df_list)
# x = column(plot_df_list[:8], 1)
# y = column(plot_df_list[:8], 2)
# plt.title("Top conditions of a hospital")
# plt.xlabel('Condition Type')
# # plt.ylabel('RMSE')
# # plt.bar(x, y)
# # plt.show()

# change_count = 0
# for _, _, count in plot_df.values.tolist():
#     if count > 2200:
#         change_count += 1

# # the full dataframe
# df = pd.DataFrame(
#     data={'conditions': column(
#         plot_df.values.tolist()[0:change_count], 1), 'value': column(
#         plot_df.values.tolist()[0:change_count], 2)},
# ).sort_values('value', ascending=False)

# # others
# new_row = pd.DataFrame(data={
#     'conditions': ['others'],
#     'value': [sum(column(plot_df.values.tolist()[change_count:], 2))]
# })

# # combining top 5 with others
# df = pd.concat([df, new_row])

# print("df now: ", df)
# # plotting -- for comparison left all countries and right
# # the others combined
# # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
# df.plot(kind='pie', y='value', labels=df['conditions'], legend=False)
# plt.title('All Conditions')
# plt.ylabel('')
# plt.show()

# # sys.exit(0)


# # ----------------------------------------------------------------------

# # ----------------------------------------------------------------------

# # For procedure codes
# plot_df = merged_df.groupby(
#     ['procedure_type_code', 'procedure_description']).size().reset_index(name='counts')
# # plot_df = plot_df[plot_df['procedure_type_code'].apply(
# #     lambda x: 428191000124101 != x)]
# plot_df = plot_df.sort_values(by='counts', ascending=False)
# put_to_csv(base_path, plot_df, "plot_df_procedure.csv")
# plot_df_list = plot_df.values.tolist()
# print("plot df: ", plot_df_list)
# x = column(plot_df_list[:8], 1)
# y = column(plot_df_list[:8], 2)
# plt.title("Top procedures in a hospital")
# plt.xlabel('Procedure Type')
# # plt.ylabel('RMSE')
# # plt.bar(x, y)
# # plt.show()

# change_count = 0
# for _, _, count in plot_df.values.tolist():
#     if count > 2500:
#         change_count += 1

# # the full dataframe
# df = pd.DataFrame(
#     data={'procedures': column(
#         plot_df.values.tolist()[0:change_count], 1), 'value': column(
#         plot_df.values.tolist()[0:change_count], 2)},
# ).sort_values('value', ascending=False)

# # others
# new_row = pd.DataFrame(data={
#     'procedures': ['others'],
#     'value': [sum(column(plot_df.values.tolist()[change_count:], 2))]
# })

# # combining top 5 with others
# df = pd.concat([df, new_row])

# print("df now: ", df)
# # plotting -- for comparison left all countries and right
# # the others combined
# # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
# df.plot(kind='pie', y='value', labels=df['procedures'], legend=False)
# plt.title('All Procedures')
# plt.ylabel('')
# plt.show()

# sys.exit(0)


# # ----------------------------------------------------------------------

# # ======================================================================
# # ----------------------------------------------------------------------
# # Writing to a file
# # ----------------------------------------------------------------------

# print("writing to file and exiting")
# put_to_csv(base_path, merged_df, "temp2.csv")
# # sys.exit(0)

# # ======================================================================
# # ----------------------------------------------------------------------
# # Reacalibrating the dataframe
# # ----------------------------------------------------------------------

# # merged_df.to_pickle('temp2.pkl')

# # merged_df = pd.read_pickle('temp2.pkl')

# # Removing entries, where the procedure was recording of drugs.
# merged_df = merged_df[merged_df['procedure_type_code'].apply(
#     lambda x: [428191000124101] != x)]

# # print("writing to file and exiting")
# # put_to_csv(base_path, merged_df, "tempPre.csv")

# # Again - Dropping columns that are not required
# # 51.9 % accuracy
# merged_df = merged_df.drop(
#     [
#         'medication_code',
#         'encounter_type_code',
#         'condition_type_code',
#         'procedure_type_code'
#     ], axis=1)

# merged_df = merged_df.rename(
#     columns={
#         'new_medication_code_x': 'medication_code',
#         'new_encounter_type_code_x': 'encounter_type_code',
#         'new_condition_type_code': 'condition_type_code',
#         'new_procedure_type_code': 'procedure_type_code'
#     })


# def merged_df_mapping(row):

#     row['encounter_type_code'] = row['new_encounter_type_code_y'] if not isinstance(
#         row['encounter_type_code'], (list, tuple, np.ndarray)) else []
#     row['medication_code'] = row['new_medication_code_y'] if not isinstance(
#         row['medication_code'], (list, tuple, np.ndarray)) else []

#     row['procedure_type_code'] = row['procedure_type_code'] if isinstance(
#         row['procedure_type_code'], (list, tuple, np.ndarray)) else []
#     row['condition_type_code'] = row['condition_type_code'] if isinstance(
#         row['condition_type_code'], (list, tuple, np.ndarray)) else []

#     return row


# merged_df = merged_df.apply(
#     lambda row: merged_df_mapping(row), axis=1)

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Getting the ratio of balanced dataset

# plot_df = merged_df.groupby(
#     ['medication_code']).size().reset_index(name='counts')
# plot_df = plot_df.sort_values(by='counts')
# print("plot df: ", plot_df)
# put_to_csv(base_path, plot_df, "plot_df.csv")
# plot_df_list = plot_df.values.tolist()
# # sys.exit(0)


# def plot_graph_list_old(plot_df_list):
#     medication_less_than_100 = 0
#     medication_less_than_1000 = 0
#     medication_1000_to_5000 = 0
#     medication_5000_to_10000 = 0
#     medication_10000_to_20000 = 0
#     medication_20000_to_40000 = 0
#     medication_greater_than_40000 = 0
#     # for medication_code, count in plot_df_list:
#     for medication_code, count in plot_df_list.items():
#         if count < 100:
#             medication_less_than_100 += 1
#         elif count < 1000:
#             medication_less_than_1000 += 1
#         elif count < 5000:
#             medication_1000_to_5000 += 1
#         elif count < 10000:
#             medication_5000_to_10000 += 1
#         elif count < 20000:
#             medication_10000_to_20000 += 1
#         elif count < 40000:
#             medication_20000_to_40000 += 1
#         else:
#             medication_greater_than_40000 += 1

#     x = ['Less than 100', '100 to 1000', '1000 to 5000', '5000 to 10000',
#          '10000 to 20000', '20000 to 40000', 'Greater than 40000']
#     y = [medication_less_than_100, medication_less_than_1000, medication_1000_to_5000, medication_5000_to_10000,
#          medication_10000_to_20000, medication_20000_to_40000, medication_greater_than_40000]

#     plt.bar(x, y)
#     plt.show()
#     # sys.exit(0)


# def plot_graph(plot_df_list):
#     medication_less_than_100 = 0
#     medication_less_than_1000 = 0
#     medication_1000_to_5000 = 0
#     medication_5000_to_10000 = 0
#     medication_10000_to_15000 = 0
#     medication_15000_to_25000 = 0
#     medication_greater_than_25000 = 0
#     # for medication_code, count in plot_df_list:
#     for medication_code, count in plot_df_list.items():
#         if count < 101:
#             medication_less_than_100 += 1
#         elif count < 1001:
#             medication_less_than_1000 += 1
#         elif count < 5001:
#             medication_1000_to_5000 += 1
#         elif count < 10001:
#             medication_5000_to_10000 += 1
#         elif count < 15001:
#             medication_10000_to_15000 += 1
#         elif count < 25001:
#             medication_15000_to_25000 += 1
#         else:
#             medication_greater_than_25000 += 1

#     x = ['Less than 100', '100 to 1000', '1000 to 5000', '5000 to 10000',
#          '10000 to 15000', '15000 to 25000', 'Greater than 25000']
#     y = [medication_less_than_100, medication_less_than_1000, medication_1000_to_5000, medication_5000_to_10000,
#          medication_10000_to_15000, medication_15000_to_25000, medication_greater_than_25000]

#     # plt.bar(x, y)
#     # plt.show()
#     # sys.exit(0)


# ----------------------------------------------------------------------
# Data Pre-preprocessing
# ----------------------------------------------------------------------

# Dropping columns that are not required

# 25 % accuracy
# merged_df = merged_df[['dose_form_code', 'encounter_type_code',
#                        'condition_type_code', 'procedure_type_code', 'medication_code']]

# print("starting mapping...")

# merged_df.to_pickle('temp1.pkl')

# merged_df = pd.read_pickle('temp1.pkl')

# ----------------------------------------------------------------------
# Dumping and loading pickle file for faster processing
# ----------------------------------------------------------------------


# merged_df.to_pickle('temp.pkl')

merged_df = pd.read_pickle('temp.pkl')

# print("writing to file and exiting")
# put_to_csv(base_path, merged_df)
# sys.exit(0)

# ----------------------------------------------------------------------

# separating the labels and the target variable.
X = merged_df.drop(['medication_code'], axis=1)
y = merged_df['medication_code']
print("pre transformation: ", type(X), type(
    X['condition_type_code']), type(X['encounter_type_code']))

# print(X)
# print(y)


# ----------------------------------------------------------------------
# Encoding categorical data
# Encoding the Independent Variable
print("starting encoding")
# encoding the procedure_type_code to multi-label encoding.
mlb = MultiLabelBinarizer()
temp_X1 = mlb.fit_transform(
    X['procedure_type_code'])

# encoding the condition_type_code to multi-label encoding.
mlb_2 = MultiLabelBinarizer()
temp_X2 = mlb_2.fit_transform(
    X['condition_type_code'])

# encoding the encounter_type_code to multi-label encoding.
mlb_3 = MultiLabelBinarizer()
temp_X3 = mlb_3.fit_transform(
    X['encounter_type_code'])

# print("after procedure_type_code: ", temp_X1)
# print("after condition_type_code: ", temp_X2)
# print("after encounter_type_code: ", temp_X3)
# put_unconstructed_np_array_to_csv(base_path, temp_X1, "tempP.csv")
# put_unconstructed_np_array_to_csv(base_path, temp_X2, "tempC.csv")
# put_unconstructed_np_array_to_csv(base_path, temp_X3, "tempE.csv")

# converting the does_form_code to one hot encoding
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='drop')
temp_X4 = ct.fit_transform(X).toarray()
# print("after dose_form_code: ", temp_X4)
# put_unconstructed_np_array_to_csv(base_path, temp_X4, "tempD.csv")

X = np.concatenate((temp_X1, temp_X2, temp_X3, temp_X4), axis=1)
# print("after ct: ", X)
# put_unconstructed_np_array_to_csv(base_path, X, "tempMerge.csv")

# ----------------------------------------------------------------------

# converting the encounter_type_code, condition_type_code, procedure_type_code to one hot encoding

# When year field is available.
# ct = ColumnTransformer(
#     transformers=[('encoder', OneHotEncoder(), [0, 1, 2])], remainder='passthrough')
# sc = StandardScaler()
# X['year'] = sc.fit_transform(X['year'].values.reshape(-1, 1))
# print("post sc:", X)

# ----------------------

# ct = ColumnTransformer(
#     transformers=[('encoder', OneHotEncoder(), [0, 1, 2, 3])], remainder='passthrough')
# X = ct.fit_transform(X).toarray()

# ----------------------------------------------------------------------

# Encoding the Dependent Variable
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)

# ----------------------------------------------------------------------
print("Performing over-sampling")
print(Counter(y))
# plot_graph(Counter(y))
over_sampling_dict = {}

for key, value in dict(Counter(y)).items():
    if (value < 15000):
        new_value = 15000
    else:
        new_value = value
    over_sampling_dict[key] = new_value

print("Over-sampling dict: ", over_sampling_dict)
# define oversampling strategy
over = RandomOverSampler(sampling_strategy=over_sampling_dict)
# fit and apply the transform
X, y = over.fit_resample(X, y)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
print("Performing under-sampling")
print(Counter(y))
under_sampling_dict = {}
for key, value in dict(Counter(y)).items():
    if (value > 15000):
        new_value = 15000
    else:
        new_value = value
    under_sampling_dict[key] = new_value
print("Under-sampling dict: ", under_sampling_dict)
# define undersampling strategy
under = RandomUnderSampler(sampling_strategy=under_sampling_dict)
# fit and apply the transform
X, y = under.fit_resample(X, y)
# ----------------------------------------------------------------------

print("sampling complete")
print(Counter(y))
# plot_graph(Counter(y))
# ----------------------------------------------------------------------

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# Feature scaling.
# There is no need of feature scaling.

# ----------------------------------------------------------------------
# Model Training
# ----------------------------------------------------------------------
print("starting training")

# Training the Decision Tree Classification model on the Training set
# classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
# classifier.fit(X_train, y_train)

# ----------------------------------------------------------------------

# Training the Random Forest model on the Training set
classifier = RandomForestClassifier(
    n_estimators=20, criterion='entropy', random_state=0,
    min_samples_split=2, min_samples_leaf=4, max_features='sqrt', max_depth=80, bootstrap=False)
classifier.fit(X_train, y_train)

# ----------------------------------------------------------------------

# Training the Random Forest model on the Training set
# classifier = RandomForestClassifier(
#     n_estimators=700, criterion='entropy', random_state=0,
#     min_samples_split=2, min_samples_leaf=4, max_features='sqrt', max_depth=80, bootstrap=False)
# classifier.fit(X_train, y_train)

# ----------------------------------------------------------------------

# SVM - Linear SVC classifier - 48%
# classifier = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, tol=0.0001,
#                        C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1,
#                        class_weight=None, verbose=0, random_state=None, max_iter=1500)
# classifier.fit(X_train, y_train)

# ----------------------------------------------------------------------

#  C- support SVM
# classifier = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True,
#                  probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
#                  decision_function_shape='ovr', break_ties=False, random_state=None)
# classifier.fit(X_train, y_train)

# ----------------------------------------------------------------------

# K-Means clustering - 37%
# classifier = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto',
#                                   leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
# classifier.fit(X_train, y_train)

# ----------------------------------------------------------------------

# n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]  # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestClassifier()
# # Random search of parameters, using 3 fold cross validation,
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100,
#                                cv=3, verbose=2, random_state=42, n_jobs=-1)  # Fit the random search model
# rf_random.fit(X_train, y_train)
# print("best params: ", rf_random.best_params_)

# ----------------------------------------------------------------------
# Test Results
# ----------------------------------------------------------------------

# Predicting the Test set results
y_pred = classifier.predict(X_test)

_, X_train_predict, _, y_train_predict = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1)
y_train_pred = classifier.predict(X_train_predict)

# ----------------------------------------------------------------------

# y_pred = rf_random.predict(X_test)

# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
print("")
print("Accuracy score: ", ac*100.0)
print("")
# print("RMSE Train: ", mean_squared_error(
#     y_train_predict, y_train_pred, squared=False))
# print("RMSE Test: ", mean_squared_error(y_test, y_pred, squared=False))
# print("")
print("end time: ", datetime.datetime.now())

# ======================================================================
# computed:
true_values = 0
total_values = 0
for i in range(len(cm)):
    for j in range(len(cm[i])):
        if i == j:
            true_values += cm[i][j]
        total_values += cm[i][j]
print("")
print("Computed Accuracy Score: ", (true_values/total_values)*100.0)
# print("Computed Precision: ", (true_values/total_values)*100.0)
print("")
# ======================================================================
# ----------------------------------------------------------------------

# Training the Random Forest model on the Training set - K-cross validation
# kfold = KFold(n_splits=15, random_state=100, shuffle=True)

# model_kfold = RandomForestClassifier(
#     n_estimators=700, criterion='entropy', random_state=0,
#     min_samples_split=2, min_samples_leaf=4, max_features='sqrt', max_depth=80, bootstrap=False)

# results_kfold = cross_val_score(model_kfold, X_train, y_train, cv=kfold)

# print("Overall: ", results_kfold)
# print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))

# ----------------------------------------------------------------------

# skfold = StratifiedKFold(n_splits=10, random_state=100, shuffle=True)

# model_skfold = RandomForestClassifier(
#     n_estimators=700, criterion='entropy', random_state=0,
#     min_samples_split=2, min_samples_leaf=4, max_features='sqrt', max_depth=80, bootstrap=False)

# results_skfold = cross_val_score(model_skfold, X_train, y_train, cv=skfold)

# print("Overall: ", results_skfold)
# print("Accuracy: %.2f%%" % (results_skfold.mean()*100.0))

# ----------------------------------------------------------------------

# loocv = LeaveOneOut()

# model_loocv = RandomForestClassifier(
#     n_estimators=700, criterion='entropy', random_state=0,
#     min_samples_split=2, min_samples_leaf=4, max_features='sqrt', max_depth=80, bootstrap=False)

# results_loocv = cross_val_score(model_loocv, X_train, y_train, cv=loocv)

# print("Overall: ", results_loocv)
# print("Accuracy: %.2f%%" % (results_loocv.mean()*100.0))

# ----------------------------------------------------------------------

# kfold2 = ShuffleSplit(n_splits=10, test_size=0.30, random_state=100)

# model_shufflecv = RandomForestClassifier(
#     n_estimators=700, criterion='entropy', random_state=0,
#     min_samples_split=2, min_samples_leaf=4, max_features='sqrt', max_depth=80, bootstrap=False)

# results_4 = cross_val_score(model_shufflecv, X_train, y_train, cv=kfold2)

# print("Overall: ", results_4)
# print("Accuracy: %.2f%% (%.2f%%)" %
#       (results_4.mean()*100.0, results_4.std()*100.0))

# ----------------------------------------------------------------------
