# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from modules.extract_csv_data import get_merged_data, base_path
from modules.put_to_csv import put_to_csv, put_np_array_to_csv
from modules.utility_functions import update_nan_most_frequent_category, update_nan_most_frequent_category_tuple, remove_nan
from pathlib import Path
import sys

# ======================================================================
# ----------------------------------------------------------------------
# Extracting the requried data
# ----------------------------------------------------------------------

data = get_merged_data()
medications_df = data['medications_df']
encounters_df = data['encounters_df']
procedures_df = data['procedures_df']
conditions_df = data['conditions_df']
# observations_df = data['observations_df']

# Merging encounter with medication to get the target variable
merged_df = encounters_df.merge(
    medications_df, left_on='encounter_id', right_on='encounter_id')
print("#1")
# First hundred/thousand entries
# merged_df = merged_df.head(1000)

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
    procedures_df, left_on='encounter_id', right_on='encounter_id', how='inner')
print("#2")
merged_df_2 = merged_df.merge(
    conditions_df, left_on='encounter_id', right_on='encounter_id', how='inner')
print("#3")

# ======================================================================
# ----------------------------------------------------------------------
#  Dropping the duplicates
# ----------------------------------------------------------------------
#

merged_df_1 = merged_df_1.drop_duplicates(subset=['encounter_id', 'medication_code',
                                                  'procedure_type_code'], keep='first')

merged_df_2 = merged_df_2.drop_duplicates(subset=['encounter_id', 'medication_code',
                                                  'condition_type_code'], keep='first')


# ======================================================================
# ----------------------------------------------------------------------
#  Constructing the conditions as an array
# ----------------------------------------------------------------------
#

# merged_df_1 = merged_df_1.astype({"procedure_type_code": str}, errors='raise')

# merged_df_1 = merged_df_1.groupby(['encounter_id', 'encounter_type_code', 'medication_code'])[
#     'procedure_type_code'].apply(tuple).reset_index()

# merged_df_2 = merged_df_2.astype({"condition_type_code": str}, errors='raise')

# merged_df_2 = merged_df_2.groupby(['encounter_id', 'encounter_type_code', 'medication_code'])[
#     'condition_type_code'].apply(tuple).reset_index()


# ======================================================================
# ----------------------------------------------------------------------
# Merging the dataframes
# ----------------------------------------------------------------------

merged_df = merged_df_1.merge(merged_df_2, on=['encounter_id', 'patient_id', 'encounter_type_code', 'encounter_description',
                                               'encounter_reason_code', 'encounter_reason_description', 'medication_code', 'medication',
                                               'medication_reason_code', 'medication_reason_description'], how='outer')
# merged_df = merged_df_2

# merged_df = merged_df_1.merge(merged_df_2, on=[
#                               'encounter_id', 'encounter_type_code', 'medication_code'], how='outer')

# ======================================================================
# ----------------------------------------------------------------------
# Writing to a file
# ----------------------------------------------------------------------

# print(merged_df)
# put_to_csv(base_path, merged_df)
# sys.exit(0)

# ======================================================================
# ----------------------------------------------------------------------
# Data Pre-preprocessing
# ----------------------------------------------------------------------

# Dropping columns that are not required
# 25 % accuracy
merged_df = merged_df[['encounter_type_code',
                       'condition_type_code', 'procedure_type_code', 'medication_code']]

# 28 % accuracy
# merged_df = merged_df[['encounter_type_code',
#                        'condition_type_code', 'medication_code']]

# 21 % accuracy
# merged_df = merged_df[['encounter_type_code', 'medication_code']]

# ----------------------------------------------------------------------

# separating the labels and the target variable.
X = merged_df.drop(['medication_code'], axis=1)
y = merged_df['medication_code']
print("pre transformation: ", type(X), type(
    X['condition_type_code']), type(X['encounter_type_code']))
print(X)
print(y)
# put_to_csv(base_path, merged_df)
# sys.exit(0)

# ----------------------------------------------------------------------

# Taking care of missing data
update_nan_most_frequent_category(X, merged_df_2, 'condition_type_code')
update_nan_most_frequent_category(X, merged_df_1, 'procedure_type_code')
# update_nan_most_frequent_category_tuple(X, merged_df_2, 'condition_type_code')
# update_nan_most_frequent_category_tuple(X, merged_df_1, 'procedure_type_code')
# remove_nan(X, "condition_type_code")
# remove_nan(X, "procedure_type_code")
print("post filling")
put_to_csv(base_path, X, "temp2.csv")
# sys.exit(0)

# ----------------------------------------------------------------------
# Encoding categorical data
# Encoding the Independent Variable

# # encoding the procedure_type_code to multi-label encoding.
# mlb = MultiLabelBinarizer()
# temp_X1 = mlb.fit_transform(
#     X['procedure_type_code'])
# # print("after mlb: ", type(X), type(
# #     X['condition_type_code']), type(X['encounter_type_code']), type(temp_X1))
# # print(temp_X1)

# # encoding the condition_type_code to multi-label encoding.
# mlb_2 = MultiLabelBinarizer()
# temp_X2 = mlb_2.fit_transform(
#     X['condition_type_code'])
# # print(temp_X2)
# X['procedure_type_code'] = temp_X1
# X['condition_type_code'] = temp_X2

# # converting the encounter_type_code to one hot encoding
# ct = ColumnTransformer(
#     transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# X = ct.fit_transform(X).toarray()
# # print("after ct: ", X)
# # put_np_array_to_csv(base_path, X)

# ----------------------------------------------------------------------

# converting the encounter_type_code, condition_type_code, procedure_type_code to one hot encoding
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0, 1, 2])], remainder='passthrough')
X = ct.fit_transform(X).toarray()

# ----------------------------------------------------------------------

# Encoding the Dependent Variable
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)

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

# Training the Decision Tree Classification model on the Training set
# classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
# classifier.fit(X_train, y_train)

# ----------------------------------------------------------------------

# Training the Random Forest model on the Training set
classifier = RandomForestClassifier(
    n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# ----------------------------------------------------------------------
# Test Results
# ----------------------------------------------------------------------

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
print("accuracy score: ", ac)

# ======================================================================
