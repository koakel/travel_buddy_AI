import requests
import json
from typing import Dict, Any, List

from llm_providers.base_model import BaseLLMModel


class BaiduModel(BaseLLMModel):
    """Baidu Qianfan API implementation of LLM provider"""

    def __init__(self, api_key: str, api_secret: str, api_base: str = "https://aip.baidubce.com"):
        """
        Initialize Baidu model
        
        Args:
            api_key: Baidu API key
            api_secret: Baidu API secret
            api_base: Baidu API base URL
        """
        super().__init__(api_key, api_base)
        self.api_secret = api_secret
        self.model_name = "ERNIE-Bot-4"  # Default model
        self.access_token = None
        
    def _get_access_token(self) -> str:
        """
        Get Baidu API access token
        
        Returns:
            Access token string
        """
        if self.access_token is not None:
            return self.access_token
            
        url = f"{self.api_base}/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.api_secret
        }
        
        response = requests.post(url, params=params)
        response.raise_for_status()
        result = response.json()
        
        self.access_token = result["access_token"]
        return self.access_token
        
    def generate_response(self,
                          user_query: str,
                          retrieved_info: str,
                          temperature: float = 0.7,
                          max_tokens: int = 800) -> str:
        """
        Generate response using Baidu API
        
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
        
        access_token = self._get_access_token()
        url = f"{self.api_base}/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{self.model_name.lower()}?access_token={access_token}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "top_p": 0.8,
            "max_output_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract generated text from Baidu response format
            generated_text = result["result"]
            return generated_text
            
        except Exception as e:
            print(f"API调用错误: {e}")
            # Fallback response in case of error
            return "抱歉，我暂时无法为您提供信息。请稍后再试或联系客服。" 