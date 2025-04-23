import requests
import os

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("GEMINI_API_KEY environment variable is not set.")
    exit(1)

url = f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
response = requests.get(url)

print("Status Code:", response.status_code)
print("Response:")
print(response.json())
