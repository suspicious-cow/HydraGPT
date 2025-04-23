import os
import streamlit as st
import requests

# Provider configuration
PROVIDERS = {
    "OpenAI": {
        "env": "OPENAI_API_KEY",
        "api_url": "https://api.openai.com/v1/chat/completions"
    },
    "Gemini": {
        "env": "GEMINI_API_KEY",
        "api_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
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

st.set_page_config(page_title="HydraGPT Chat", page_icon="🤖")
st.title("HydraGPT Chat")

# Sidebar for provider selection
provider = st.sidebar.selectbox("Select Provider", list(PROVIDERS.keys()))

# Store chat history in session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display chat history
for msg in st.session_state['messages']:
    if msg['role'] == 'user':
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**{provider}:** {msg['content']}")

# Always show chat input at the end
prompt = st.chat_input("Enter your prompt:")

def call_openai(api_key, prompt):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(PROVIDERS["OpenAI"]["api_url"], headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"OpenAI Error: {str(e)}\nResponse: {getattr(e, 'response', None)}"

def call_gemini(api_key, prompt):
    try:
        url = f"{PROVIDERS['Gemini']['api_url']}?key={api_key}"
        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ]
        }
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Gemini Error: {str(e)}\nResponse: {getattr(e, 'response', None)}"

def call_anthropic(api_key, prompt):
    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(PROVIDERS["Anthropic"]["api_url"], headers=headers, json=data)
        response.raise_for_status()
        return response.json()['content'][0]['text']
    except Exception as e:
        return f"Anthropic Error: {str(e)}\nResponse: {getattr(e, 'response', None)}"

def call_grok(api_key, prompt):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "grok-1",
            "messages": [{"role": "user", "content": prompt}]
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
        st.session_state['messages'].append({"role": "assistant", "content": response})
        st.rerun()
