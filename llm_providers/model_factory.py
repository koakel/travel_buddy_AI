import os
from typing import Dict, Any, Optional, List

from llm_providers.base_model import BaseLLMModel
from llm_providers.deepseek_model import DeepSeekModel
from llm_providers.openai_model import OpenAIModel
from llm_providers.claude_model import ClaudeModel
from llm_providers.gemini_model import GeminiModel
from llm_providers.baidu_model import BaiduModel


# Import future model implementations here
# from llm_providers.qwen_model import QwenModel
# from llm_providers.gemini_model import GeminiModel
# from llm_providers.claude_model import ClaudeModel
# from llm_providers.baidu_model import BaiduModel


class ModelFactory:
    """Factory class to create LLM provider instances"""
    
    @staticmethod
    def get_model(provider: str, api_key: str, api_base: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> BaseLLMModel:
        """
        Get an LLM provider instance based on the specified provider and configuration
        
        Args:
            provider: The name of the LLM provider (e.g., "deepseek", "openai")
            api_key: The API key for the LLM provider
            api_base: The API base URL for the LLM provider (optional)
            config: Additional configuration parameters for the LLM provider (optional)
            
        Returns:
            An instance of the specified LLM provider
            
        Raises:
            ValueError: If the specified provider is not supported
        """
        # Initialize config dict if None
        if config is None:
            config = {}
        
        # Add api_key and api_base to config
        config["api_key"] = api_key
        if api_base:
            config["api_base"] = api_base
            
        # Use the existing create_model method
        return ModelFactory.create_model(provider, config)
    
    @staticmethod
    def create_model(provider: str, config: Dict[str, Any]) -> BaseLLMModel:
        """
        Create an LLM provider instance based on the specified provider and configuration
        
        Args:
            provider: The name of the LLM provider (e.g., "deepseek", "openai")
            config: Configuration parameters for the LLM provider
            
        Returns:
            An instance of the specified LLM provider
            
        Raises:
            ValueError: If the specified provider is not supported
        """
        provider = provider.lower()
        
        if provider == "deepseek":
            api_key = config.get("api_key", os.environ.get("DEEPSEEK_API_KEY", ""))
            api_base = config.get("api_base", "https://api.deepseek.com")
            return DeepSeekModel(api_key=api_key, api_base=api_base)
            
        elif provider == "openai":
            api_key = config.get("api_key", os.environ.get("OPENAI_API_KEY", ""))
            api_base = config.get("api_base", "https://api.openai.com/v1")
            model = OpenAIModel(api_key=api_key, api_base=api_base)
            
            # Optional: Update model parameters if provided
            if "model_name" in config:
                model.model_name = config["model_name"]
                
            return model
            
        elif provider == "claude":
            api_key = config.get("api_key", os.environ.get("ANTHROPIC_API_KEY", ""))
            api_base = config.get("api_base", "https://api.anthropic.com")
            model = ClaudeModel(api_key=api_key, api_base=api_base)
            
            # Optional: Update model parameters if provided
            if "model_name" in config:
                model.model_name = config["model_name"]
                
            return model
            
        elif provider == "gemini":
            api_key = config.get("api_key", os.environ.get("GEMINI_API_KEY", ""))
            api_base = config.get("api_base", "https://generativelanguage.googleapis.com")
            model = GeminiModel(api_key=api_key, api_base=api_base)
            
            # Optional: Update model parameters if provided
            if "model_name" in config:
                model.model_name = config["model_name"]
                
            return model
            
        elif provider == "baidu":
            api_key = config.get("api_key", os.environ.get("BAIDU_API_KEY", ""))
            api_secret = config.get("api_secret", os.environ.get("BAIDU_API_SECRET", ""))
            api_base = config.get("api_base", "https://aip.baidubce.com")
            model = BaiduModel(api_key=api_key, api_secret=api_secret, api_base=api_base)
            
            # Optional: Update model parameters if provided
            if "model_name" in config:
                model.model_name = config["model_name"]
                
            return model
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}") 