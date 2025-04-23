import requests
import os

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("ANTHROPIC_API_KEY environment variable is not set.")
    exit(1)

url = "https://api.anthropic.com/v1/models"
headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01"
}

response = requests.get(url, headers=headers)

print("Status Code:", response.status_code)
print("Response:")
print(response.json())
