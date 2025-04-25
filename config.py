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

import os
import json

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