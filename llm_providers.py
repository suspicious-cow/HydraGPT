import os
import requests
import streamlit as st
from config import PROVIDERS

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
