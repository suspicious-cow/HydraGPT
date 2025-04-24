import threading
import json
import os
import time
import requests
from datetime import datetime, timedelta

CACHE_PATH = os.path.join(os.path.dirname(__file__), "hf_provider_model_cache.json")
CACHE_TTL_HOURS = 24

# Function to fetch and cache provider/model pairs
def fetch_and_cache_hf_provider_models():
    url = "https://huggingface.co/api/models?inference_provider=all&pipeline_tag=text-generation&limit=50"
    response = requests.get(url)
    models = response.json()
    pairs = []
    for model in models:
        model_id = model['id']
        info_url = f"https://huggingface.co/api/models/{model_id}?expand=inferenceProviderMapping"
        info_resp = requests.get(info_url)
        mapping = info_resp.json().get('inferenceProviderMapping', {})
        for provider in mapping.keys():
            pairs.append({"provider": provider, "model": model_id})
        time.sleep(0.5)
    with open(CACHE_PATH, "w") as f:
        json.dump({"fetched_at": datetime.utcnow().isoformat(), "pairs": pairs}, f)

# Function to load from cache (if fresh)
def load_hf_provider_model_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            data = json.load(f)
            fetched_at = datetime.fromisoformat(data.get("fetched_at", "1970-01-01T00:00:00"))
            if datetime.utcnow() - fetched_at < timedelta(hours=CACHE_TTL_HOURS):
                return data["pairs"]
    return None

# On import, start background thread if cache is missing or stale
def ensure_hf_provider_model_cache():
    pairs = load_hf_provider_model_cache()
    if pairs is None:
        threading.Thread(target=fetch_and_cache_hf_provider_models, daemon=True).start()
    return pairs

if __name__ == "__main__":
    print("Checking for cached Hugging Face provider/model pairs...")
    pairs = ensure_hf_provider_model_cache()
    if pairs:
        print("Loaded from cache:")
        for pair in pairs:
            print(f"{pair['provider']} / {pair['model']}")
    else:
        print("Cache is being built in the background. Please re-run this script in a minute.")
