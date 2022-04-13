

import json
import requests
import csv


def get_drug_info(code):
    # code = "310965"
    URL = f"https://rxnav.nlm.nih.gov/REST/rxcui/{code}/allrelated.json"
    r = requests.get(url=URL, params={})

    # extracting data in json format
    data = r.json()
    # print("##data: ", data)
    concept_group = data["allRelatedGroup"]["conceptGroup"]
    response = {
        "medication_code": code,

        "parent_code": 'XXX',
        "parent_name": 'NNNN',

        "dose_form_code": 'DDDD',
        "dose_form_name": 'EEEE',

        "parent_codes": [],
        "parent_names": [],

        "dose_form_codes": [],
        "dose_form_names": []
    }
    for group in concept_group:

        if group["tty"] == "IN" and ("conceptProperties" in group):
            print("parent property1: ", group)
            response["parent_code"] = group["conceptProperties"][0]["rxcui"]
            response["parent_name"] = group["conceptProperties"][0]["name"]
            response["parent_codes"].append(
                group["conceptProperties"][0]["rxcui"])
            response["parent_names"].append(
                group["conceptProperties"][0]["name"])

        elif group["tty"] == "PIN" and ("conceptProperties" in group):
            print("parent property3: ", group)
            response["parent_code"] = group["conceptProperties"][0]["rxcui"]
            response["parent_name"] = group["conceptProperties"][0]["name"]
            response["parent_codes"].append(
                group["conceptProperties"][0]["rxcui"])
            response["parent_names"].append(
                group["conceptProperties"][0]["name"])

        elif group["tty"] == "MIN" and ("conceptProperties" in group):
            print("multiple parent property2: ", group)
            response["parent_code"] = group["conceptProperties"][0]["rxcui"]
            response["parent_name"] = group["conceptProperties"][0]["name"]
            response["parent_codes"].append(
                group["conceptProperties"][0]["rxcui"])
            response["parent_names"].append(
                group["conceptProperties"][0]["name"])

        if group["tty"] == "DFG" and ("conceptProperties" in group):
            print("DFG property4: ", group)
            response["dose_form_code"] = group["conceptProperties"][0]["rxcui"]
            response["dose_form_name"] = group["conceptProperties"][0]["name"]
            response["dose_form_codes"].append(
                group["conceptProperties"][0]["rxcui"])
            response["dose_form_names"].append(
                group["conceptProperties"][0]["name"])

    # print("pre response: ", response)
    return response


if __name__ == "__main__":
    # 79 procedure codes
    drug_codes = [13995008, 252160004, 274474001, 703423002, 305428000, 312681000, 429609002, 15081005, 74857009,
                  180030006, 52734007, 173160006, 31208007, 399208008, 91602002, 80146002, 313191000, 127783003, 90407005, 432231006,
                  88039007, 384700001, 305340004, 755621000000101, 1015401000000102, 714812005, 169553002, 305425002, 22523008,
                  167995008, 88848003, 387685009, 76601001, 265764009, 29240004, 428191000124101, 228557008, 232717009, 46028000,
                  73761001, 418891003, 180256009, 79733001, 699253003, 11466000, 65546002, 171231001, 609588000, 65588006, 237001001,
                  68254000, 415070008, 698354004, 46706006, 395142003, 65200003, 287664005, 447365002, 177157003, 183450002, 311791003,
                  301807007, 274031008, 76164006, 305433001, 236974004, 18286008, 43075005, 445912000, 433112001, 18946005, 23426006, 387607004,
                  269911007, 117015009, 66348005, 85548006, 112790001, 288086009]

    with open('procedure_related_info.csv', 'w', newline='') as f:
        first_time = True
        for code in drug_codes:
            print("code: ", code)
            response = get_drug_info(code)
            print("response: ", response)
            if first_time:
                w = csv.DictWriter(f, response.keys())
                w.writeheader()
                first_time = False
            w.writerow(response)
