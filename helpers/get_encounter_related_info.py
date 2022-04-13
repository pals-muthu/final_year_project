

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
