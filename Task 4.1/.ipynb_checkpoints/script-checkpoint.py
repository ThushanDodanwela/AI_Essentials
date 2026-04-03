import requests
import json

URL = 'http://51.11.113.158:80/api/v1/service/irisdeployment/score'
KEY='7aqnhr9bSdVzduOXK85aK6GwiapWIP3X'

import requests
import json

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {KEY}"
}

# Payload
data = {
    "Inputs": {
        "data": [
            {
                "Id": 0,
                "SepalLengthCm": 5.1,
                "SepalWidthCm": 3.5,
                "PetalLengthCm": 1.4,
                "PetalWidthCm": 0.2,
                "Species": "Iris-setosa"
            }
        ]
    },
    "GlobalParameters": {}
}

# Send request
response = requests.post(URL, json=data, headers=headers)

# Print result
print(response.json())