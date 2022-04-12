

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
        "dose_form_name": 'EEEE'
    }
    for group in concept_group:

        if group["tty"] == "IN" and ("conceptProperties" in group):
            print("parent property1: ", group)
            response["parent_id"] = group["conceptProperties"][0]["rxcui"]
            response["parent_name"] = group["conceptProperties"][0]["name"]
        elif group["tty"] == "PIN" and ("conceptProperties" in group):
            print("parent property3: ", group)
            response["parent_id"] = group["conceptProperties"][0]["rxcui"]
            response["parent_name"] = group["conceptProperties"][0]["name"]
        elif group["tty"] == "MIN" and ("conceptProperties" in group):
            print("multiple parent property2: ", group)
            response["parent_id"] = group["conceptProperties"][0]["rxcui"]
            response["parent_name"] = group["conceptProperties"][0]["name"]

        if group["tty"] == "DFG" and ("conceptProperties" in group):
            print("DFG property4: ", group)
            response["dose_form_id"] = group["conceptProperties"][0]["rxcui"]
            response["dose_form_name"] = group["conceptProperties"][0]["name"]
    # print("pre response: ", response)
    return response


if __name__ == "__main__":
    # 102 drug codes
    drug_codes = [978950, 389128, 834060, 833036, 849437, 1049630, 1602593, 1856546, 1049636, 1049639, 575020, 831533, 583214,
                  834101, 1536586, 311372, 141918, 1091166, 897122, 646250, 602735, 997488, 833137, 309362, 997501, 1648767, 312961, 1049221,
                  1605257, 1736854, 1803932, 849574, 310965, 998582, 259255, 896188, 1000128, 749762, 1734340, 745679, 568530, 749785, 1000158,
                  904420, 1020137, 197361, 197378, 198405, 106258, 727316, 1373463, 861467, 1359133, 751905, 607015, 312617, 860975, 309043,
                  834357, 748856, 749882, 835900, 849727, 1366342, 727374, 748879, 308056, 757594, 567645, 282464, 313185, 997221, 309097,
                  239981, 807283, 824184, 235389, 996741, 1190795, 106892, 573839, 1367439, 1014676, 1014678, 748962, 608680, 857005, 313782,
                  564666, 596927, 1049544, 858069, 243670, 392151, 617944, 1094108, 308192, 575971, 1111011, 1310197, 665078, 895994]
    with open('drug_related_info.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
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
