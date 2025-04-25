import os
import streamlit as st
import requests
import json
import threading
import time
from datetime import datetime, timedelta
import warnings
from config import PROVIDERS, load_config, save_config
from llm_providers import call_openai, call_gemini, call_anthropic, call_grok

warnings.filterwarnings("ignore")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "hydragpt_config.json")

# Load config at startup
if 'config' not in st.session_state:
    st.session_state['config'] = load_config()

st.set_page_config(page_title="HydraGPT", page_icon="🐉")
st.title("HydraGPT Chat")

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

# Only call st.chat_input once, and do not repeat it below
prompt = st.chat_input("Enter your prompt:", key="main_chat_input")

# Provider checkboxes below the prompt input
selected_providers = []
st.markdown("<div style='margin-top: 1em; margin-bottom: 0.5em;'><b>Select Providers to Compare:</b></div>", unsafe_allow_html=True)
cols = st.columns(len(PROVIDERS))
for idx, prov in enumerate(PROVIDERS.keys()):
    if cols[idx].checkbox(prov, value=True, key=f"prov_{prov}"):
        selected_providers.append(prov)

if prompt and selected_providers:
    responses = {}
    for prov in selected_providers:
        api_key = os.getenv(PROVIDERS[prov]["env"])
        if not api_key:
            responses[prov] = f"API key for {prov} not found. Please set the {PROVIDERS[prov]['env']} environment variable."
        else:
            if prov == "OpenAI":
                responses[prov] = call_openai(api_key, prompt)
            elif prov == "Gemini":
                responses[prov] = call_gemini(api_key, prompt)
            elif prov == "Anthropic":
                responses[prov] = call_anthropic(api_key, prompt)
            elif prov == "Grok":
                responses[prov] = call_grok(api_key, prompt)
            else:
                responses[prov] = "Provider not supported."
    # Display results side-by-side
    cols = st.columns(len(selected_providers))
    for idx, prov in enumerate(selected_providers):
        with cols[idx]:
            st.markdown(f"### {prov}")
            st.write(responses[prov])

st.sidebar.image("hydra_heads.png", use_container_width=True, caption=None)
