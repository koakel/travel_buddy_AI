import requests
from typing import Dict, Any, List

from llm_providers.base_model import BaseLLMModel


class OpenAIModel(BaseLLMModel):
    """OpenAI API implementation of LLM provider"""

    def __init__(self, api_key: str, api_base: str = "https://api.openai.com/v1"):
        """
        Initialize OpenAI model
        
        Args:
            api_key: OpenAI API key
            api_base: OpenAI API base URL (can be changed for Azure OpenAI)
        """
        super().__init__(api_key, api_base)
        self.model_name = "gpt-3.5-turbo"  # Default model, can be updated to gpt-4 etc.
        
    def generate_response(self,
                          user_query: str,
                          retrieved_info: str,
                          temperature: float = 0.7,
                          max_tokens: int = 800) -> str:
        """
        Generate response using OpenAI API
        
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
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", 
                 "content": "你是一位专业的旅伴智能助手，擅长提供旅游咨询和帮助，特别关注老年人和行动不便人士的需求。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()

            # Extract generated text from OpenAI response format
            generated_text = result["choices"][0]["message"]["content"]
            return generated_text

        except Exception as e:
            print(f"API调用错误: {e}")
            # Fallback response in case of error
            return "抱歉，我暂时无法为您提供信息。请稍后再试或联系客服。" 