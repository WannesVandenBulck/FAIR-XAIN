import yaml
from openai import OpenAI
import anthropic


# --- Load API keys from YAML ---
with open("config/keys.yaml", "r") as f:
    KEYS = yaml.safe_load(f)

# --- Create clients ---
openai_client = OpenAI(api_key=KEYS.get("openai_key"))
anthropic_client = anthropic.Anthropic(api_key=KEYS.get("anthropic_key"))


def generate_text(messages, provider="openai", model=None, temperature=0):
    """
    Generate text using different LLM providers.
    
    Args:
        messages: List of {"role": ..., "content": ...}
        provider: "openai", "anthropic", "ollama", etc.
        model: Model name (provider-specific)
        temperature: Sampling temperature (0-1)
    
    Returns:
        Generated text string
    """
    if provider == "openai":
        model = model or "gpt-4o"
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    elif provider == "anthropic":
        model = model or "claude-3-opus-20240229"
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=4096,
            messages=messages,
            temperature=temperature
        )
        return response.content[0].text
    
    elif provider == "ollama":
        import requests
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model or "llama2",
                "messages": messages,
                "temperature": temperature,
                "stream": False
            }
        )
        return response.json()["message"]["content"]
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
