# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import pandas as pd
from pathlib import Path
import sys
import os
base_path = Path(__file__).resolve().parents[2].joinpath('temp\\csv')
# print ("base path : ", base_path)


def get_csv_data(base_path):
    # ======================================================================
    # ----------------------------------------------------------------------
    # Medications Data
    # ----------------------------------------------------------------------
    medications_path = Path(base_path).resolve().joinpath('medications.csv')
    # print ("medications path : ", medications_path, os.path.exists(medications_path))
    medications_df = pd.read_csv(medications_path, low_memory=False)
    medications_df = medications_df.drop(['START', 'STOP', 'PATIENT'], axis=1)
    medications_df = medications_df.rename(columns={'ENCOUNTER': 'encounter_id', 'CODE': 'medication_code', 'DESCRIPTION': 'medication', 'REASONCODE': 'medication_reason_code',
                                                    'REASONDESCRIPTION': 'medication_reason_description'})
    # print ("medications_df: ", medications_df)

    # # ----------------------------------------------------------------------
    # # Allergies Data
    # # ----------------------------------------------------------------------

    # allergies_path = Path(base_path).resolve().joinpath('allergies.csv')
    # allergies_df = pd.read_csv(allergies_path, low_memory=False)

    # # ----------------------------------------------------------------------
    # # Careplans Data
    # # ----------------------------------------------------------------------

    # careplans_path = Path(base_path).resolve().joinpath('careplans.csv')
    # careplans_df = pd.read_csv(careplans_path, low_memory=False)

    # ----------------------------------------------------------------------
    # Conditions Data
    # ----------------------------------------------------------------------

    conditions_path = Path(base_path).resolve().joinpath('conditions.csv')
    conditions_df = pd.read_csv(conditions_path, low_memory=False)
    conditions_df = conditions_df.drop(['START', 'STOP', 'PATIENT'], axis=1)
    conditions_df = conditions_df.rename(columns={
        'ENCOUNTER': 'encounter_id', 'CODE': 'condition_type_code', 'DESCRIPTION': 'condition_description'})

    # ----------------------------------------------------------------------
    # Encounters Data
    # ----------------------------------------------------------------------

    encounters_path = Path(base_path).resolve().joinpath('encounters.csv')
    encounters_df = pd.read_csv(encounters_path, low_memory=False)
    encounters_df = encounters_df.drop(['DATE'], axis=1)
    encounters_df = encounters_df.rename(columns={'ID': 'encounter_id', 'PATIENT': 'patient_id', 'CODE': 'encounter_type_code', 'DESCRIPTION': 'encounter_description',
                                                  'REASONCODE': 'encounter_reason_code',
                                                  'REASONDESCRIPTION': 'encounter_reason_description'})
    # print ("encounters_df: ", encounters_df)

    # # ----------------------------------------------------------------------
    # # Immunizations Data
    # # ----------------------------------------------------------------------

    # immunizations_path = Path(
    #     base_path).resolve().joinpath('immunizations.csv')
    # immunizations_df = pd.read_csv(immunizations_path, low_memory=False)

    # # ----------------------------------------------------------------------
    # # Observations Data
    # # ----------------------------------------------------------------------

    # observations_path = Path(base_path).resolve().joinpath('observations.csv')
    # observations_df = pd.read_csv(observations_path, low_memory=False)
    # observations_df = observations_df.drop(['DATE', 'PATIENT'], axis=1)
    # observations_df = observations_df.rename(columns={'ENCOUNTER': 'encounter_id', 'CODE': 'observation_type_code', 'DESCRIPTION': 'observation_description',
    #                                                   'VALUE': 'value',
    #                                                   'UNITS': 'units'})

    # # ----------------------------------------------------------------------
    # # Patients Data
    # # ----------------------------------------------------------------------

    # patients_path = Path(base_path).resolve().joinpath('patients.csv')
    # patients_df = pd.read_csv(patients_path, low_memory=False)

    # ----------------------------------------------------------------------
    # Procedures Data
    # ----------------------------------------------------------------------
    procedures_path = Path(base_path).resolve().joinpath('procedures.csv')
    procedures_df = pd.read_csv(procedures_path, low_memory=False)
    procedures_df = procedures_df.drop(['DATE', 'PATIENT'], axis=1)
    procedures_df = procedures_df.rename(columns={'ENCOUNTER': 'encounter_id', 'CODE': 'procedure_type_code', 'DESCRIPTION': 'procedure_description',
                                                  'REASONCODE': 'procedure_reason_code',
                                                  'REASONDESCRIPTION': 'procedure_reason_description'})

    # ======================================================================

    return {
        "medications_df": medications_df,
        "encounters_df": encounters_df,
        "procedures_df": procedures_df,
        "conditions_df": conditions_df,
        # "observations_df": observations_df,
        # "patients_df": patients_df,
        # "immunizations_df": immunizations_df,
        # "careplans_df": careplans_df,
        # "allergies_df": allergies_df
    }


def merge_data(*args):

    medications_frames = []
    encounters_frames = []
    procedures_frames = []
    conditions_frames = []
    observations_frames = []
    patients_frames = []
    immunizations_frames = []
    careplans_frames = []
    allergies_frames = []

    for arg in args:
        medications_frames.append(arg["medications_df"])
        encounters_frames.append(arg["encounters_df"])
        procedures_frames.append(arg["procedures_df"])
        conditions_frames.append(arg["conditions_df"])
        # observations_frames.append(arg["observations_df"])
        # patients_frames.append(arg["patients_df"])
        # immunizations_frames.append(arg["immunizations_df"])
        # careplans_frames.append(arg["careplans_df"])
        # allergies_frames.append(arg["allergies_df"])

    medications_df = pd.concat(medications_frames)
    encounters_df = pd.concat(encounters_frames)
    procedures_df = pd.concat(procedures_frames)
    conditions_df = pd.concat(conditions_frames)
    # observations_df = pd.concat(observations_frames)
    # patients_df = pd.concat(patients_frames)
    # immunizations_df = pd.concat(immunizations_frames)
    # careplans_df = pd.concat(careplans_frames)
    # allergies_df = pd.concat(allergies_frames)

    return {
        "medications_df": medications_df,
        "encounters_df": encounters_df,
        "procedures_df": procedures_df,
        "conditions_df": conditions_df,
        # "observations_df": observations_df,
        # "patients_df": patients_df,
        # "immunizations_df": immunizations_df,
        # "careplans_df": careplans_df,
        # "allergies_df": allergies_df
    }


def get_merged_data():

    list_of_paths = [
        'dataset\\synthea_1m_fhir_3_0_May_24\\csv_output_1',
        'dataset\\synthea_1m_fhir_3_0_May_24\\csv_output_3',
        'dataset\\synthea_1m_fhir_3_0_May_24\\csv_output_4',
        'dataset\\synthea_1m_fhir_3_0_May_24\\csv_output_5',
    ]
    data = []
    for path in list_of_paths:
        temp_base_path = Path(__file__).resolve().parents[2].joinpath(path)
        temp_data = get_csv_data(temp_base_path)
        data.append(temp_data)

    return merge_data(*data)
