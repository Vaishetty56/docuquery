from google import genai
from docuquery.config import load_settings

s = load_settings()
client = genai.Client(api_key=s.api_key)
for m in client.models.list():
    name = m.name.replace("models/", "")
    actions = m.supported_actions or []
    if "embedContent" in actions or "generateContent" in actions:
        print(name, actions)