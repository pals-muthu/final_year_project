

import json
import requests
import csv


def get_drug_info(code):
    # code = "310965"
    URL = f"https://snowstorm-fhir.snomedtools.org/fhir/CodeSystem/$lookup?system=http://snomed.info/sct&code={code}"
    headers = {
        "User-Agent": "PostmanRuntime/7.28.3",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Content-Type": "application/json"
    }
    r = requests.get(url=URL, headers=headers, verify=False)

    # extracting data in json format
    data = r.json()
   # print("##data: ", data)
    if ("parameter" not in data):
        print("paramater not found in: ", data)
        return {
            "reason_code": code,
            "parent_codes": [],
            "parent_code": 'XXXXX'
        }
    parameters = data["parameter"]
    response = {
        "reason_code": code,
        "parent_codes": [],
        "parent_code": 'XXX'
    }
    for parameter in parameters:

        if parameter["name"] == "property" and ("part" in parameter) and ("valueString" in parameter["part"][0]) and \
                (parameter["part"][0]["valueString"] == "parent"):
            response["parent_codes"].append(parameter["part"][1]["valueCode"])
            response["parent_code"] = parameter["part"][1]["valueCode"]
        elif parameter["name"] == "property" and ("part" in parameter) and ("valueString" in parameter["part"][1]) and \
                (parameter["part"][1]["valueString"] == "parent"):
            response["parent_codes"].append(parameter["part"][0]["valueCode"])
            response["parent_code"] = parameter["part"][0]["valueCode"]

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
