from pathlib import Path
import json
import requests
import csv
import pandas as pd
import numpy as np
import ast


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


def put_to_csv(base_path, df, file_name=None):
    if not file_name:
        file_path = Path(base_path).resolve().joinpath('temp.csv')
    else:
        file_path = file_name
    df.to_csv(file_path, sep='\t', encoding='utf-8')
    print("#Done adding to file: ", file_path)


if __name__ == "__main__":
    # 128 condition codes
    drug_codes = [36971009, 399211009, 239872002, 196416002, 55680006, 90560007, 161621004, 75498004, 201834006, 195967001, 192127007, 263102004, 185086009, 301011002, 65363002, 307731004, 40275004, 239720000, 254632001, 200936003, 410429000, 39848009, 422034002, 197927001, 72892002, 85116003, 423121009, 58150001, 287185009, 38822007, 43878008, 68496003, 198992004, 446096008, 127013003, 24079001, 429007001, 62564004, 283385000, 230265002, 95417003, 110030002, 363406005, 233678006, 109838007, 287182007, 195662009, 444814009, 45816000, 403192003, 422968005, 6072007, 10509002, 162573006, 47693006, 79586000, 230690007, 40055000, 82423001, 698423002, 403191005, 15777000, 19169002, 46177005, 232353008, 444470001,
                  403190006, 444448004, 74400008, 429280009, 267253006, 367498001, 35999006, 94260004, 33737001, 87433001, 359817006, 241929008, 97331000119101, 90781000119102, 67841000119103, 67811000119102, 124171000119105, 368581000119106, 67831000119107, 1551000119108, 67821000119109, 275272006, 1501000119109, 69896004, 60951000119105, 157141000119108, 443165006, 16114001, 284551006, 370247008, 26929004, 44465007, 1734006, 64859006, 428251008, 30832001, 70704007, 38341003, 284549007, 22298006, 62106007, 424132000, 233604007, 37849005, 287193009, 65966004, 262574004, 398254007, 425048006, 609496007, 254637007, 53741008, 713197008, 698754002, 236077008, 15724005, 86849004, 93761005, 239873007, 44054006, 161622006, 283371005]

    # with open('reason_related_info.csv', 'w', newline='') as f:
    #     first_time = True
    #     for code in drug_codes:
    #         print("code: ", code)
    #         response = get_reason_info(code)
    #         print("response: ", response)
    #         if first_time:
    #             w = csv.DictWriter(f, response.keys())
    #             w.writeheader()
    #             first_time = False
    #         w.writerow(response)

    # cmerging eason code and parent reason code
    reason_related_info_path = 'reason_related_info.csv'
    reason_related_info_df = pd.read_csv(
        reason_related_info_path, low_memory=False)
    complete_response = []

    def label_race(row):
        # print("new_list1: ", type(row['parent_codes']), row['parent_codes'])
        new_list = ast.literal_eval(
            row['parent_codes']) + [str(row['reason_code'])]
        # print("new_list2: ", new_list)
        return new_list
    reason_related_info_df['compiled_codes'] = reason_related_info_df.apply(
        lambda row: label_race(row), axis=1)
    reason_related_info_df.drop(['parent_codes'], axis=1, inplace=True)
    reason_related_info_df.drop(['parent_code'], axis=1, inplace=True)
    put_to_csv("", reason_related_info_df, "reason_related_info_update.csv")
