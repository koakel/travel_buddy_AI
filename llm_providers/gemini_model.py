import json
from typing import Dict, Any, List
from google import genai
from llm_providers.base_model import BaseLLMModel
from google.genai import types

class GeminiModel(BaseLLMModel):
    """Google Gemini API implementation of LLM provider"""

    def __init__(self, api_key: str, api_base: str = "https://generativelanguage.googleapis.com"):
        """
        Initialize Gemini model
        
        Args:
            api_key: Gemini API key
            api_base: Gemini API base URL
        """
        super().__init__(api_key, api_base)
        self.model_name = "gemini-2.5-pro-exp-03-25"  # Default model
        # Initialize the Gemini client
        self.client = genai.Client(api_key=api_key)
        
    def generate_response(self,
                          user_query: str,
                          retrieved_info: str,
                          temperature: float = 0.7,
                          max_tokens: int = 800) -> str:
        """
        Generate response using Gemini API
        
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
        
        try:
            # Use the official Gemini client to generate content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    top_k=40,
                    top_p=0.95
                )
            )
            
            # Extract generated text from response
            generated_text = response.text
            return generated_text
            
        except Exception as e:
            print(f"API调用错误: {e}")
            # Fallback response in case of error
            return "抱歉，我暂时无法为您提供信息。请稍后再试或联系客服。" 