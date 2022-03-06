import pandas as pd
import numpy as np
from extract_csv_data import medications_df, encounters_df, procedures_df, conditions_df, base_path
from pathlib import Path

# ======================================================================
# Extracting the requried data
# ======================================================================

merged_df = encounters_df.merge(
    medications_df, left_on='encounter_id', right_on='encounter_id')
print("#1")
merged_df = merged_df.head(100)

# ----------------------------------------------------------------------
# https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#database-style-dataframe-or-named-series-joining-merging
# Way 1
# merged_df = merged_df.merge(procedures_df, left_on='encounter_id', right_on='encounter_id', how='left')
# print ("#2")
# merged_df = merged_df.merge(conditions_df, left_on='encounter_id', right_on='encounter_id', how='left')
# print ("#3")
# # merged_df = merged_df.merge(observations_df, left_on='encounter_id', right_on='encounter_id', how='left')
# print ("#4")

# ----------------------------------------------------------------------
# Way 2
merged_df_1 = merged_df.merge(
    procedures_df, left_on='encounter_id', right_on='encounter_id', how='inner')
print("#2")
merged_df_2 = merged_df.merge(
    conditions_df, left_on='encounter_id', right_on='encounter_id', how='inner')
print("#3")
# merged_df_3 = merged_df.merge(observations_df, left_on='encounter_id', right_on='encounter_id', how='inner')
merged_df = merged_df_1.merge(merged_df_2, on=['encounter_id', 'patient_id', 'encounter_type_code', 'encounter_description',
                                               'encounter_reason_code', 'encounter_reason_description', 'medication_code', 'medication',
                                               'medication_reason_code', 'medication_reason_description'], how='outer')
print("#4")

# ----------------------------------------------------------------------


file_name = Path(base_path).resolve().joinpath('temp.csv')
merged_df.to_csv(file_name, sep='\t', encoding='utf-8')
print("#Done")
print(merged_df)

# ======================================================================
