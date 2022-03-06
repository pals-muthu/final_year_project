import pandas as pd
from pathlib import Path
import sys
import os
base_path = Path(__file__).resolve().parents[1].joinpath('temp\\csv')
# print ("base path : ", base_path)

# ======================================================================

# ======================================================================
medications_path = Path(base_path).resolve().joinpath('medications.csv')
# print ("medications path : ", medications_path, os.path.exists(medications_path))
medications_df = pd.read_csv(medications_path, low_memory=False)
medications_df = medications_df.drop(['START', 'STOP', 'PATIENT'], axis = 1)
medications_df = medications_df.rename(columns={'ENCOUNTER': 'encounter_id', 'CODE': 'medication_code', 'DESCRIPTION' : 'medication', 'REASONCODE': 'medication_reason_code', 
                                'REASONDESCRIPTION': 'medication_reason_description'})
# print ("medications_df: ", medications_df)

# ----------------------------------------------------------------------

allergies_path = Path(base_path).resolve().joinpath('allergies.csv')
allergies_df = pd.read_csv(allergies_path, low_memory=False)

# ----------------------------------------------------------------------

careplans_path = Path(base_path).resolve().joinpath('careplans.csv')
careplans_df = pd.read_csv(careplans_path, low_memory=False)

# ----------------------------------------------------------------------

conditions_path = Path(base_path).resolve().joinpath('conditions.csv')
conditions_df = pd.read_csv(conditions_path, low_memory=False)
conditions_df = conditions_df.drop(['START', 'STOP', 'PATIENT'], axis = 1)
conditions_df = conditions_df.rename(columns={'ENCOUNTER': 'encounter_id', 'CODE': 'condition_type_code', 'DESCRIPTION' : 'condition_description'})

# ----------------------------------------------------------------------

encounters_path = Path(base_path).resolve().joinpath('encounters.csv')
encounters_df = pd.read_csv(encounters_path, low_memory=False)
encounters_df = encounters_df.drop(['DATE'], axis = 1)
encounters_df = encounters_df.rename(columns={'ID': 'encounter_id', 'PATIENT': 'patient_id', 'CODE': 'encounter_type_code', 'DESCRIPTION' : 'encounter_description', 
                                'REASONCODE': 'encounter_reason_code', 
                                'REASONDESCRIPTION': 'encounter_reason_description'})
# print ("encounters_df: ", encounters_df)

# ----------------------------------------------------------------------

immunizations_path = Path(base_path).resolve().joinpath('immunizations.csv')
immunizations_df = pd.read_csv(immunizations_path, low_memory=False)

# ----------------------------------------------------------------------

observations_path = Path(base_path).resolve().joinpath('observations.csv')
observations_df = pd.read_csv(observations_path, low_memory=False)
observations_df = observations_df.drop(['DATE', 'PATIENT'], axis = 1)
observations_df = observations_df.rename(columns={'ENCOUNTER': 'encounter_id', 'CODE': 'observation_type_code', 'DESCRIPTION' : 'observation_description', 
                                'VALUE': 'value', 
                                'UNITS': 'units'})

# ----------------------------------------------------------------------

patients_path = Path(base_path).resolve().joinpath('patients.csv')
patients_df = pd.read_csv(patients_path, low_memory=False)

# ----------------------------------------------------------------------

procedures_path = Path(base_path).resolve().joinpath('procedures.csv')
procedures_df = pd.read_csv(procedures_path, low_memory=False)
procedures_df = procedures_df.drop(['DATE', 'PATIENT'], axis = 1)
procedures_df = procedures_df.rename(columns={'ENCOUNTER': 'encounter_id', 'CODE': 'procedure_type_code', 'DESCRIPTION' : 'procedure_description', 
                                'REASONCODE': 'procedure_reason_code', 
                                'REASONDESCRIPTION': 'procedure_reason_description'})


# ======================================================================

merged_df = encounters_df.merge(medications_df, left_on='encounter_id', right_on='encounter_id')
print ("#1")
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
merged_df_1 = merged_df.merge(procedures_df, left_on='encounter_id', right_on='encounter_id', how='inner')
print ("#2")
merged_df_2 = merged_df.merge(conditions_df, left_on='encounter_id', right_on='encounter_id', how='inner')
print ("#3")
# merged_df_3 = merged_df.merge(observations_df, left_on='encounter_id', right_on='encounter_id', how='inner')
merged_df = merged_df_1.merge(merged_df_2, on=['encounter_id', 'patient_id', 'encounter_type_code', 'encounter_description', 
                                                'encounter_reason_code', 'encounter_reason_description', 'medication_code', 'medication', 
                                                'medication_reason_code', 'medication_reason_description'], how='outer')
print ("#4")

# ----------------------------------------------------------------------


file_name = Path(base_path).resolve().joinpath('temp.csv')
merged_df.to_csv(file_name, sep='\t', encoding='utf-8')
print ("#Done")
# print (merged_df)


