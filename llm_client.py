"""HTTP client for the local LLM server. Supports both raw prompt and chat endpoints."""
from typing import Dict, Any, Optional, List
import requests


class LLHTTPClient:
    def __init__(self, host: str = "localhost", port: int = 8080, timeout: int = 300):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def send_prompt(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3,
                     no_repeat_ngram_size: int = 3, repetition_penalty: float = 1.2) -> Dict[str, Any]:
        """Send a raw prompt to /v1/generate."""
        url = f"{self.base}/v1/generate"
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repetition_penalty": repetition_penalty,
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512,
             temperature: float = 0.3) -> str:
        """Send chat messages to /v1/chat. Returns the assistant's response text."""
        url = f"{self.base}/v1/chat"
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("content", "") or data.get("generated_text", "")


if __name__ == "__main__":
    client = LLHTTPClient()
    print(client.send_prompt("Hello from local LLM client", max_tokens=16))
