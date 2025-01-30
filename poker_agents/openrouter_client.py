import requests
import json
import os
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class OpenRouterClient:
    def __init__(self, model_name: str = "anthropic/claude-2"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        self.model = model_name
        self.base_url = "https://openrouter.ai/api/v1"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "http://localhost"),
            "X-Title": "Poker Agent"
        }

    def get_available_models(self):
        """Fetch available models from OpenRouter"""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
            # Extract model data from response
            return [
                {
                    "id": model["id"],
                    "name": model.get("name", model["id"]),
                    "context_length": model.get("context_length", 4096)
                }
                for model in data.get("data", [])
            ]
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            return []

    def get_completion(self, messages: List[Dict]) -> str:
        """Get completion from OpenRouter"""
        try:
            response = requests.post(
                url=f"{self.base_url}/chat/completions",
                headers=self.headers,
                data=json.dumps({
                    "model": self.model,
                    "messages": messages
                })
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Failed to get completion: {e}")
            raise