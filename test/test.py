'''
If not done
docker pull flowiseai/flowise
then
docker run -d --name flowise -p 5000:3000 flowise
or if already started
docker restart flowise
you can open flowise with
localhost:3000
'''

import requests
import json

FLOW_ID = "43db1b2b-1ceb-44a4-a227-0ae90738fe47"  # Substitute with your Flow-ID
FLOWISE_API_URL = f"http://localhost:8000/api/v1/prediction/{FLOW_ID}"

# let's create some example in json
landmark_json = {
    "face_oval": [{"x": 10, "y": 20}],
    "eyes": [{"x": 30, "y": 40}],
    "nose": [{"x": 50, "y": 60}],
    "mouth": [{"x": 70, "y": 80}],
    "chin": [{"x": 90, "y": 100}]
}

response = requests.post(
    FLOWISE_API_URL,
    json={"question": landmark_json}
)

print("\n--- Answer from Flowise ---")

try:
    response_json = response.json()
    if response.status_code == 200:
        print("Status Code:", response.status_code)
        print("Output from LLM:")
        print(response_json.get("text"))  # Verwende .get() um None-Fehler zu vermeiden
        #print("Question:", json.dumps(response_json.get("question"), indent=4)) # Schöneres JSON-Format
        print("Chat ID:", response_json.get("chatId"))
        print("Chat Message ID:", response_json.get("chatMessageId"))
        #print("Is Stream Valid:", response_json.get("isStreamValid"))
        #print("Session ID:", response_json.get("sessionId"))
    else:
        print("Error trying to get an answer!")
        print("Status Code:", response.status_code)
        print("Error Message:", response_json.get("message"))
        print("Stack Trace:", response_json.get("stack"))

except json.JSONDecodeError:
    print("Error: Not valid JSON answer fromFlowise!")
    print("Text:", response.text)

print("--- Ready ---")

