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
import pandas as pd
import numpy as np
from modules.extract_csv_data import get_merged_data, base_path, get_drug_data_from_csv, get_condition_data_from_csv, get_procedure_data_from_csv, get_encounter_data_from_csv
from modules.put_to_csv import put_to_csv, put_np_array_to_csv, put_unconstructed_np_array_to_csv
from modules.utility_functions import update_nan_most_frequent_category, update_nan_most_frequent_category_tuple, remove_nan
from pathlib import Path
import sys
import ast
import datetime

# ======================================================================
# ----------------------------------------------------------------------
# Extracting the requried data
# ----------------------------------------------------------------------
print("start time: ", datetime.datetime.now())
data = get_merged_data()
medications_df = data['medications_df']
encounters_df = data['encounters_df']
procedures_df = data['procedures_df']
conditions_df = data['conditions_df']
# patients_df = data['patients_df']
# observations_df = data['observations_df']


# # Merging encounter with patients to get more information
# merged_df = encounters_df.merge(
#     patients_df, left_on='patient_id', right_on='patient_id', how='inner')

# Merging encounter with medication to get the target variable
merged_df = encounters_df.merge(
    medications_df, left_on='encounter_id', right_on='encounter_id', how='inner')

print("#2")
print("merged_df: ", list(merged_df.columns.values))

# put_to_csv(base_path, merged_df, "temp1.csv")
# sys.exit(0)

# First hundred/thousand entries
# merged_df = merged_df.head(1000)

# ----------------------------------------------------------------------
# Mapping the encounter, procedure and conditions with the medications.

merged_df_2 = merged_df.merge(
    conditions_df, left_on='encounter_id', right_on='encounter_id', how='inner')
print("#3")
print("merged_df_2 condition: ", list(merged_df_2.columns.values))


# ------------------------------

# put_to_csv(base_path, merged_df_2, "merged_df_2_pre.csv")

subset_merged_df_2 = merged_df_2[[
    'encounter_id', 'encounter_type_code', 'medication_code']]
subset_merged_df_2 = subset_merged_df_2.drop_duplicates(
    subset=['encounter_id', 'encounter_type_code', 'medication_code'], keep='first')

encounter_id_set = subset_merged_df_2.values.tolist()

encounter_id_new_map = []

index = 0
total_length = len(encounter_id_set)
print("started iterations2..")
for encounter_id, encounter_type_code, medication_code in encounter_id_set:
    index += 1
    if index % 500 == 0:
        print(f"processed {index} of {total_length}")

    temp_df = merged_df_2[(merged_df_2['encounter_id'] ==
                          encounter_id) & (merged_df_2['medication_code'] == medication_code)]
    temp_conditions = []
    temp_conditions_arrays = []

    for index_2, row in temp_df.iterrows():
        temp_conditions.append(row['condition_type_code'])
        temp_conditions_arrays += row['new_condition_type_code']

    encounter_id_new_map.append([encounter_id, temp_df['year'].values[0],
                                 temp_df['patient_id'].values[0],
                                 encounter_type_code,
                                 temp_df['encounter_description'].values[0],
                                 temp_df['new_encounter_type_code'].values[0],
                                 medication_code,
                                 temp_df['medication'].values[0],
                                 temp_df['new_medication_code'].values[0],
                                 temp_df['dose_form_code'].values[0],
                                 temp_conditions,
                                 temp_df['condition_description'].values[0],
                                 temp_conditions_arrays])

merged_df_2 = pd.DataFrame(encounter_id_new_map, columns=['encounter_id', 'year', 'patient_id',
                                                          'encounter_type_code', 'encounter_description', 'new_encounter_type_code', 'medication_code', 'medication',
                                                          'new_medication_code', 'dose_form_code', 'condition_type_code', 'condition_description', 'new_condition_type_code'])

merged_df_2.set_index('encounter_id')
# put_to_csv(base_path, merged_df_2, "merged_df_2_post.csv")

merged_df_2.to_pickle('merged_df_2_post_pickle_separate.pkl')
sys.exit(0)
