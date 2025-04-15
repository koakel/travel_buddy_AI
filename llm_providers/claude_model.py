import requests
from typing import Dict, Any, List

from llm_providers.base_model import BaseLLMModel


class ClaudeModel(BaseLLMModel):
    """Anthropic Claude API implementation of LLM provider"""

    def __init__(self, api_key: str, api_base: str = "https://api.anthropic.com"):
        """
        Initialize Claude model
        
        Args:
            api_key: Anthropic API key
            api_base: Claude API base URL
        """
        super().__init__(api_key, api_base)
        self.model_name = "claude-3-haiku-20240307"  # Default model
        self.api_version = "v1"  # API version
        
    def generate_response(self,
                          user_query: str,
                          retrieved_info: str,
                          temperature: float = 0.7,
                          max_tokens: int = 800) -> str:
        """
        Generate response using Claude API
        
        Args:
            user_query: Original user question/query
            retrieved_info: Information retrieved from database
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        prompt = self.create_prompt(
            user_query=user_query,
            retrieved_info=retrieved_info
        )

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                f"{self.api_base}/messages",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract generated text from Claude response format
            generated_text = result["content"][0]["text"]
            return generated_text
            
        except Exception as e:
            print(f"API调用错误: {e}")
            # Fallback response in case of error
            return "抱歉，我暂时无法为您提供信息。请稍后再试或联系客服。" 