# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from modules.extract_csv_data import medications_df, encounters_df, procedures_df, conditions_df, observations_df, base_path
from modules.put_to_csv import put_to_csv
from pathlib import Path

# ======================================================================
# ----------------------------------------------------------------------
# Extracting the requried data
# ----------------------------------------------------------------------

# Merging encounter with medication to get the target variable
merged_df = encounters_df.merge(
    medications_df, left_on='encounter_id', right_on='encounter_id')
print("#1")
# First hundred entries
merged_df = merged_df.head(1000)

# ----------------------------------------------------------------------
# # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#database-style-dataframe-or-named-series-joining-merging
# # Way 1 - Bringing in procedures, conditions and optionally observations together to identify the medication.
# merged_df = merged_df.merge(procedures_df, left_on='encounter_id', right_on='encounter_id', how='left')
# print ("#2")
# merged_df = merged_df.merge(conditions_df, left_on='encounter_id', right_on='encounter_id', how='left')
# print ("#3")
# # merged_df = merged_df.merge(observations_df, left_on='encounter_id', right_on='encounter_id', how='left')
# print ("#4")

# ----------------------------------------------------------------------
# # Way 2 - Bringing in procedures, conditions and optionally observations together to identify the medication.
# merged_df_1 = merged_df.merge(
#     procedures_df, left_on='encounter_id', right_on='encounter_id', how='inner')
# print("#2")
# merged_df_2 = merged_df.merge(
#     conditions_df, left_on='encounter_id', right_on='encounter_id', how='inner')
# print("#3")
# merged_df_3 = merged_df.merge(
#     observations_df, left_on='encounter_id', right_on='encounter_id', how='inner')
# print("#4")
# merged_df = merged_df_1.merge(merged_df_2, on=['encounter_id', 'patient_id', 'encounter_type_code', 'encounter_description',
#                                                'encounter_reason_code', 'encounter_reason_description', 'medication_code', 'medication',
#                                                'medication_reason_code', 'medication_reason_description'], how='outer')
# print("#5")
# merged_df = merged_df.merge(merged_df_3, on=['encounter_id', 'patient_id', 'encounter_type_code', 'encounter_description',
#                                              'encounter_reason_code', 'encounter_reason_description', 'medication_code', 'medication',
#                                              'medication_reason_code', 'medication_reason_description'], how='outer')
# print("#6")
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Way 3 - starting with the basics. mapping only the encounter and conditions with the medications.
merged_df_1 = merged_df.merge(
    conditions_df, left_on='encounter_id', right_on='encounter_id', how='inner')
print("#2")

merged_df = merged_df_1

# ======================================================================
# ----------------------------------------------------------------------
# Writing to a file
# ----------------------------------------------------------------------
# put_to_csv(base_path, merged_df)

# ======================================================================
# ----------------------------------------------------------------------
# Data Pre-preprocessing
# ----------------------------------------------------------------------

# Dropping columns that are not required
# merged_df = merged_df.drop(['encounter_id', 'patient_id', 'encounter_description', 'encounter_reason_code',
#                             'encounter_reason_description', 'medication', 'medication_reason_code', 'medication_reason_description'], axis=1)
merged_df = merged_df[['encounter_type_code',
                       'condition_type_code', 'medication_code']]


# separating the labels and the target variable.
X = merged_df.drop(['medication_code'], axis=1)
y = merged_df['medication_code']

print(X)
print(y)
put_to_csv(base_path, merged_df)

# Taking care of missing data
# so far no missing data

# Encoding categorical data
# Encoding the Independent Variable
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0, 1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Encoding the Dependent Variable
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)


# ======================================================================
