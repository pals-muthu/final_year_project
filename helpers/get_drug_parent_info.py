

import json
import requests

id = "310965"
URL = f"https://rxnav.nlm.nih.gov/REST/rxcui/{id}/allrelated.json"
r = requests.get(url=URL, params={})

# extracting data in json format
data = r.json()
print("##data: ", data)
concept_group = data["allRelatedGroup"]["conceptGroup"]
for group in concept_group:
    if group["tty"] == "IN":
        print("parent property: ", group)
        parent_id = group["conceptProperties"][0]["rxcui"]
        parent_name = group["conceptProperties"][0]["name"]
