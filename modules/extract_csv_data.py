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
    # print("pre obtained data: ", len(medications_df.index))

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
    # encounters_df = encounters_df.drop(['DATE'], axis=1)
    encounters_df['DATE'] = encounters_df['DATE'].str[0:4]
    encounters_df = encounters_df.rename(columns={'ID': 'encounter_id', 'PATIENT': 'patient_id', 'CODE': 'encounter_type_code', 'DESCRIPTION': 'encounter_description',
                                                  'REASONCODE': 'encounter_reason_code', 'DATE': 'year',
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
    # # print("patients_path: ", patients_path)
    # patients_df = pd.read_csv(patients_path, low_memory=False)
    # patients_df = patients_df.drop(['BIRTHDATE', 'DEATHDATE', 'SSN', 'DRIVERS', 'PASSPORT', 'PREFIX', 'FIRST', 'LAST',
    #                                 'SUFFIX', 'MAIDEN', 'MARITAL', 'BIRTHPLACE', 'ADDRESS'], axis=1)

    # patients_df = patients_df.query('GENDER == "M" or GENDER == "F"')
    # patients_df = patients_df.rename(columns={
    #     'ID': 'patient_id', 'RACE': 'race', 'ETHNICITY': 'ethnicity', 'GENDER': 'gender'})
    # patients_df = patients_df[['patient_id', 'race', 'ethnicity', 'gender']]

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
        # 'dataset\\synthea_1m_fhir_3_0_May_24\\csv_output_6',
        # 'dataset\\synthea_1m_fhir_3_0_May_24\\csv_output_7',
        # 'dataset\\synthea_1m_fhir_3_0_May_24\\csv_output_8',
        # 'dataset\\synthea_1m_fhir_3_0_May_24\\csv_output_9',
    ]
    data = []
    for path in list_of_paths:
        temp_base_path = Path(__file__).resolve().parents[2].joinpath(path)
        temp_data = get_csv_data(temp_base_path)
        data.append(temp_data)

    return merge_data(*data)


def get_drug_data_from_csv():
    base_path = Path(__file__).resolve().parents[1].joinpath(
        'helpers\\drug_related_info_update.csv')
    drugs_df = pd.read_csv(base_path, low_memory=False)
    return drugs_df


def get_condition_data_from_csv():
    base_path = Path(__file__).resolve().parents[1].joinpath(
        'helpers\\reason_related_info_update.csv')
    condition_df = pd.read_csv(base_path, low_memory=False)
    return condition_df


def get_procedure_data_from_csv():
    base_path = Path(__file__).resolve().parents[1].joinpath(
        'helpers\\procedure_related_info_update.csv')
    procedure_df = pd.read_csv(base_path, low_memory=False)
    return procedure_df


def get_encounter_data_from_csv():
    base_path = Path(__file__).resolve().parents[1].joinpath(
        'helpers\\encounter_related_info_update.csv')
    encounter_df = pd.read_csv(base_path, low_memory=False)
    return encounter_df


def dump_drug_code(data):
    my_set = set()
    medications_df = data['medications_df']
    print("obtained data: ", len(medications_df.index))
    for index, row in medications_df.iterrows():
        # print("adding to set: ", type(
        #     row['medication_code']), row['medication_code'])
        my_set.add(row['medication_code'])

    print("my_drug_set: ", my_set, len(my_set))


def dump_conditions_code(data):
    my_set = set()
    conditions_df = data['conditions_df']
    print("obtained data: ", len(conditions_df.index))
    for index, row in conditions_df.iterrows():
        my_set.add(row['condition_type_code'])

    print("my_conditions_set: ", my_set, len(my_set))


def dump_procedures_code(data):
    my_set = set()
    procedures_df = data['procedures_df']
    print("obtained data: ", len(procedures_df.index))
    for index, row in procedures_df.iterrows():
        my_set.add(row['procedure_type_code'])

    print("my_procedures_set: ", my_set, len(my_set))


def dump_encounters_code(data):
    my_set = set()
    encounters_df = data['encounters_df']
    print("obtained data: ", len(encounters_df.index))
    for index, row in encounters_df.iterrows():
        my_set.add(row['encounter_type_code'])

    print("my_encounters_set: ", my_set, len(my_set))


if __name__ == '__main__':
    data = get_merged_data()
    dump_drug_code(data)
    dump_conditions_code(data)
    dump_procedures_code(data)
    dump_encounters_code(data)
