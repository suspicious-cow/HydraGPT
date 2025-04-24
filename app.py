import os
import streamlit as st
import requests
import json
import threading
import time
from datetime import datetime, timedelta

# Provider configuration
PROVIDERS = {
    "OpenAI": {
        "env": "OPENAI_API_KEY",
        "api_url": "https://api.openai.com/v1/chat/completions"
    },
    "Gemini": {
        "env": "GEMINI_API_KEY",
        "api_url": "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"
    },
    "Anthropic": {
        "env": "ANTHROPIC_API_KEY",
        "api_url": "https://api.anthropic.com/v1/messages"
    },
    "Grok": {
        "env": "XAI_API_KEY",
        "api_url": "https://api.x.ai/v1/chat/completions"
    },
    "HuggingFace": {
        "env": "HF_TOKEN",
        "api_url": "https://router.huggingface.co"
    }
}

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "hydragpt_config.json")
HF_CACHE_PATH = os.path.join(os.path.dirname(__file__), "hf_provider_model_cache.json")
HF_CACHE_TTL_HOURS = 24

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {
        'selected_models': {
            'OpenAI': 'gpt-4.1-mini',
            'Gemini': 'gemini-2.0-flash',
            'Anthropic': 'claude-3-7-sonnet-20250219',
            'Grok': 'grok-3-latest',
            'HuggingFace': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'HuggingFaceProvider': 'hf-inference'
        }
    }

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f)

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
    with open(HF_CACHE_PATH, "w") as f:
        json.dump({"fetched_at": datetime.utcnow().isoformat(), "pairs": pairs}, f)

def load_hf_provider_model_cache():
    if os.path.exists(HF_CACHE_PATH):
        with open(HF_CACHE_PATH, "r") as f:
            data = json.load(f)
            fetched_at = datetime.fromisoformat(data.get("fetched_at", "1970-01-01T00:00:00"))
            if datetime.utcnow() - fetched_at < timedelta(hours=HF_CACHE_TTL_HOURS):
                return data["pairs"]
    return None

def ensure_hf_provider_model_cache():
    pairs = load_hf_provider_model_cache()
    if pairs is None:
        threading.Thread(target=fetch_and_cache_hf_provider_models, daemon=True).start()
    return pairs

# Load config at startup
if 'config' not in st.session_state:
    st.session_state['config'] = load_config()

st.set_page_config(page_title="HydraGPT Chat", page_icon="🤖")
st.title("HydraGPT Chat")

# Sidebar for provider selection
provider = st.sidebar.selectbox("Select Provider", list(PROVIDERS.keys()))

# --- Settings Section ---
with st.sidebar.expander("⚙️ Settings", expanded=False):
    st.markdown("### Model Selection")
    # Fetch and cache models per provider
    def get_openai_models(api_key):
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get("https://api.openai.com/v1/models", headers=headers)
            response.raise_for_status()
            # Filter for chat/completions models
            return [m['id'] for m in response.json()['data'] if 'gpt' in m['id']]
        except Exception as e:
            return ["gpt-4.1-mini"]

    def get_gemini_models(api_key):
        try:
            url = f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
            response = requests.get(url)
            response.raise_for_status()
            return [m['name'].split('/')[-1] for m in response.json().get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
        except Exception as e:
            return ["gemini-2.0-flash"]

    def get_anthropic_models(api_key):
        try:
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
            response = requests.get("https://api.anthropic.com/v1/models", headers=headers)
            response.raise_for_status()
            return [m['id'] for m in response.json().get('data', [])]
        except Exception as e:
            return ["claude-3-7-sonnet-20250219"]

    def get_grok_models(api_key):
        # Grok does not have a public list endpoint; hardcode known models
        return ["grok-3-latest"]

    def get_hf_models(hf_token):
        try:
            headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
            response = requests.get("https://huggingface.co/api/models?filter=text-generation&sort=downloads&direction=-1&limit=50&full=true", headers=headers)
            response.raise_for_status()
            # Use model['id'] for the model id (not modelId)
            return [m['id'] for m in response.json() if m.get('id') and '/' in m['id']]
        except Exception as e:
            return ["meta-llama/Meta-Llama-3-8B-Instruct"]

    def get_hf_provider_model_pairs(hf_token):
        pairs = ensure_hf_provider_model_cache()
        if pairs:
            return [(pair["provider"], pair["model"]) for pair in pairs]
        return [("hf-inference", "meta-llama/Meta-Llama-3-8B-Instruct")]

    # Get API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    grok_key = os.getenv("XAI_API_KEY")
    hf_token = os.getenv("HF_TOKEN")

    # Model selection dropdowns
    config = st.session_state['config']
    selected_models = config['selected_models']

    openai_models = get_openai_models(openai_key) if openai_key else ["gpt-4.1-mini"]
    gemini_models = get_gemini_models(gemini_key) if gemini_key else ["gemini-2.0-flash"]
    anthropic_models = get_anthropic_models(anthropic_key) if anthropic_key else ["claude-3-7-sonnet-20250219"]
    grok_models = get_grok_models(grok_key)
    hf_models = get_hf_models(hf_token) if hf_token else ["meta-llama/Meta-Llama-3-8B-Instruct"]
    hf_provider_model_pairs = get_hf_provider_model_pairs(hf_token) if hf_token else [("hf-inference", "meta-llama/Meta-Llama-3-8B-Instruct")]
    # Sort alphabetically by provider
    hf_provider_model_pairs = sorted(hf_provider_model_pairs, key=lambda x: x[0])
    hf_combo_labels = [f"{provider} / {model}" for provider, model in hf_provider_model_pairs]
    default_combo = f"{selected_models.get('HuggingFaceProvider', 'hf-inference')} / {selected_models.get('HuggingFace', 'meta-llama/Meta-Llama-3-8B-Instruct')}"
    selected_combo = st.selectbox("Hugging Face Provider/Model", hf_combo_labels, index=hf_combo_labels.index(default_combo) if default_combo in hf_combo_labels else 0)
    selected_provider, selected_model = hf_provider_model_pairs[hf_combo_labels.index(selected_combo)]
    selected_models['HuggingFaceProvider'] = selected_provider
    selected_models['HuggingFace'] = selected_model

    selected_models['OpenAI'] = st.selectbox("OpenAI Model", openai_models, index=openai_models.index(selected_models['OpenAI']) if selected_models['OpenAI'] in openai_models else 0)
    selected_models['Gemini'] = st.selectbox("Gemini Model", gemini_models, index=gemini_models.index(selected_models['Gemini']) if selected_models['Gemini'] in gemini_models else 0)
    selected_models['Anthropic'] = st.selectbox("Anthropic Model", anthropic_models, index=anthropic_models.index(selected_models['Anthropic']) if selected_models['Anthropic'] in anthropic_models else 0)
    selected_models['Grok'] = st.selectbox("Grok Model", grok_models, index=grok_models.index(selected_models['Grok']) if selected_models['Grok'] in grok_models else 0)

    if st.button("Save Model Settings"):
        config['selected_models'] = selected_models
        save_config(config)
        st.success("Model settings saved!")
        st.session_state['config'] = config

# Store chat history in session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display chat history
for msg in st.session_state['messages']:
    if msg['role'] == 'user':
        st.markdown(f"**You:** {msg['content']}")
    else:
        provider_label = msg.get('provider', provider)
        st.markdown(f"**{provider_label}:** {msg['content']}")

# Always show chat input at the end
prompt = st.chat_input("Enter your prompt:")

def call_openai(api_key, prompt):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": st.session_state['config']['selected_models']['OpenAI'],
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(PROVIDERS["OpenAI"]["api_url"], headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"OpenAI Error: {str(e)}\nResponse: {getattr(e, 'response', None)}"

def call_gemini(api_key, prompt):
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models/{st.session_state['config']['selected_models']['Gemini']}:generateContent?key={api_key}"
        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ]
        }
        response = requests.post(url, json=data)
        if response.status_code != 200:
            return f"Gemini Error: {response.status_code} {response.reason}\nResponse: {response.text}"
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Gemini Exception: {str(e)}"

def call_anthropic(api_key, prompt):
    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": st.session_state['config']['selected_models']['Anthropic'],
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(PROVIDERS["Anthropic"]["api_url"], headers=headers, json=data)
        response.raise_for_status()
        return response.json()['content'][0]['text']
    except Exception as e:
        return f"Anthropic Error: {str(e)}\nResponse: {getattr(e, 'response', None)}"

def call_grok(api_key, prompt, system_message="You are a helpful assistant."):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": st.session_state['config']['selected_models']['Grok'],
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "temperature": 0
        }
        response = requests.post(PROVIDERS["Grok"]["api_url"], headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Grok Error: {str(e)}\nResponse: {getattr(e, 'response', None)}"

def call_huggingface(api_key, prompt, provider, model_id, system_message="You are a helpful assistant."):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # Use the selected provider/model in the router endpoint
        url = f"https://router.huggingface.co/{provider}/v1/chat/completions"
        data = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": False
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Hugging Face Error: {str(e)}\nResponse: {getattr(e, 'response', None)}"

if prompt:
    api_key = os.getenv(PROVIDERS[provider]["env"])
    if not api_key:
        st.error(f"API key for {provider} not found. Please set the {PROVIDERS[provider]['env']} environment variable.")
    else:
        st.session_state['messages'].append({"role": "user", "content": prompt})
        if provider == "OpenAI":
            response = call_openai(api_key, prompt)
        elif provider == "Gemini":
            response = call_gemini(api_key, prompt)
        elif provider == "Anthropic":
            response = call_anthropic(api_key, prompt)
        elif provider == "Grok":
            response = call_grok(api_key, prompt)
        elif provider == "HuggingFace":
            hf_model = st.session_state['config']['selected_models']['HuggingFace']
            hf_provider = st.session_state['config']['selected_models']['HuggingFaceProvider']
            response = call_huggingface(api_key, prompt, hf_provider, hf_model)
        else:
            response = "Provider not supported."
        st.session_state['messages'].append({"role": "assistant", "content": response, "provider": provider})
        st.rerun()
