import os
import streamlit as st
import requests
import json
import threading
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

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
    }
}

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "hydragpt_config.json")

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {
        'selected_models': {
            'OpenAI': 'gpt-4.1-mini',
            'Gemini': 'gemini-2.0-flash',
            'Anthropic': 'claude-3-7-sonnet-20250219',
            'Grok': 'grok-3-latest'
        }
    }

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f)

# Load config at startup
if 'config' not in st.session_state:
    st.session_state['config'] = load_config()

st.set_page_config(page_title="HydraGPT", page_icon="🐉")
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

    # Get API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    grok_key = os.getenv("XAI_API_KEY")

    # Model selection dropdowns (no Hugging Face)
    config = st.session_state['config']
    selected_models = config['selected_models']

    openai_models = get_openai_models(openai_key) if openai_key else ["gpt-4.1-mini"]
    gemini_models = get_gemini_models(gemini_key) if gemini_key else ["gemini-2.0-flash"]
    anthropic_models = get_anthropic_models(anthropic_key) if anthropic_key else ["claude-3-7-sonnet-20250219"]
    grok_models = get_grok_models(grok_key)

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
        else:
            response = "Provider not supported."
        st.session_state['messages'].append({"role": "assistant", "content": response, "provider": provider})
        st.rerun()

st.sidebar.image("hydra_heads.png", use_container_width=True, caption=None)
