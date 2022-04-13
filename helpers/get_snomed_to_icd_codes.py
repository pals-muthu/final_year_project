import json
import requests
import csv


def get_reason_info(code):
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
    # 128 condition codes
    drug_codes = [36971009, 399211009, 239872002, 196416002, 55680006, 90560007, 161621004, 75498004, 201834006, 195967001, 192127007,
                  263102004, 185086009, 301011002, 65363002, 307731004, 40275004, 239720000, 254632001, 200936003, 410429000, 39848009, 422034002,
                  197927001, 72892002, 85116003, 423121009, 58150001, 287185009, 38822007, 43878008, 68496003, 198992004, 446096008, 127013003,
                  24079001, 429007001, 62564004, 283385000, 230265002, 95417003, 110030002, 363406005, 233678006, 109838007, 287182007, 195662009,
                  444814009, 45816000, 403192003, 422968005, 6072007, 10509002, 162573006, 47693006, 79586000, 230690007, 40055000, 82423001,
                  698423002, 403191005, 15777000, 19169002, 46177005, 232353008, 444470001, 403190006, 444448004, 74400008, 429280009, 267253006,
                  367498001, 35999006, 94260004, 33737001, 87433001, 359817006, 241929008, 97331000119101, 90781000119102, 67841000119103, 67811000119102,
                  124171000119105, 368581000119106, 678310001197008, 698754002, 236077008, 15724005, 86849004, 93761005, 239873007, 44054006, 161622006, 283371005]
    # You will need 'wb' mode in Python 2.x
    with open('reason_related_info.csv', 'w', newline='') as f:
        first_time = True
        for code in drug_codes:
            print("code: ", code)
            response = get_reason_info(code)
            print("response: ", response)
            if first_time:
                w = csv.DictWriter(f, response.keys())
                w.writeheader()
                first_time = False
            w.writerow(response)
