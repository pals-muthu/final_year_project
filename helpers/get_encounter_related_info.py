

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
    # 28 encounter codes
    drug_codes = [305408004, 170837001, 185349003, 32485007, 698314001, 308335008, 183460006, 424441002, 185347001,
                  305411003, 266707007, 394701000, 316744009, 34285007, 170258001, 210098006, 183495009, 56876005, 183452005, 50849002,
                  185345009, 308646001, 183478001, 170838006, 371883000, 184347001, 270427003, 424619006]

    with open('encounter_related_info.csv', 'w', newline='') as f:
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
