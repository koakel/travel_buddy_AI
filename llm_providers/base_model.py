from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json

class BaseLLMModel(ABC):
    """Base abstract class for LLM provider implementations"""

    # 通用提示词模板，可以在子类中覆盖
    PROMPT_TEMPLATE = """你是一位专注于服务中老年人的旅游伴侣，以下是你的角色特点：
    1. 你擅长给中老年人提供他们所需的解惑、体谅和支持，比如了解老年人出行的不便、体谅老年人对电子产品的不适应。请注意在回答问题时请尽量体现你对中老年用户的关怀，比如涉及景点等情景时要介绍老年人优惠政策和无障碍设施。
    2. 你非常熟悉大理及周边地区的各类特色景点、饮食、住宿和深度体验项目，能够从体验角度给出细节建议。
    3. 你的回答应尽量控制在200字内，确保在适配老年人习惯的大字号的移动端上能够方便阅读。
    4. 下面根据用户需求提供了一些景点或住宿的信息，请你酌情向用户进行介绍和推荐。
    
    重要：你的回答必须严格按照以下JSON格式输出：
    ```json
    {{
        "needs_card": "yes/no",
        "content": "你的回复内容"
    }}
    ```
    
    其中：
    - needs_card: 只能是"yes"或"no"，表示这个回答是否应该显示为卡片格式
      - 如果用户在询问具体景点、住宿、餐厅等可以展示为卡片的信息，设置为"yes"
      - 如果用户在询问的是一般性问题、闲聊、建议、天气等不需要卡片展示的内容，设置为"no"
    - content: 你对用户问题的回答内容
    
    回答类型判断要点：
    - 卡片回复（needs_card="yes"）：用于具体景点信息、特定住宿问题、清晰可选项的推荐
    - 文本回复（needs_card="no"）：用于一般问候、天气咨询、开放性建议、闲聊、用户询问你是谁等

    接下来请你根据以上要求开展与用户的对话：
    - 用户提问：{user_query}
    - 检索信息：{retrieved_info}
    - 请以JSON格式回复:
    """

    def __init__(self, api_key: str, api_base: Optional[str] = None):
        """
        Initialize the LLM provider
        
        Args:
            api_key: API key for the provider
            api_base: Base URL for API (optional)
        """
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = None  # Will be set by subclasses

    @abstractmethod
    def generate_response(self,
                          user_query: str,
                          retrieved_info: str,
                          temperature: float = 0.7,
                          max_tokens: int = 800) -> str:
        """
        Generate a response using the LLM provider
        
        Args:
            user_query: Original user question/query
            retrieved_info: Information retrieved from database
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        pass
    
    def create_prompt(self,
                      user_query: str,
                      retrieved_info: str) -> str:
        """
        Create a prompt for the LLM provider using the template
        
        Args:
            user_query: Original user question/query
            retrieved_info: Information retrieved from database
            
        Returns:
            Prompt text
        """
        # 使用模板格式化提示词
        return self.PROMPT_TEMPLATE.format(
            user_query=user_query,
            retrieved_info=retrieved_info
        ) 
    
    def _is_valid_json(self, string_to_test: str) -> bool:
        """
        测试字符串是否为有效的JSON格式
        
        Args:
            string_to_test: 要测试的字符串
            
        Returns:
            布尔值，表示是否为有效JSON
        """
        try:
            json.loads(string_to_test)
            return True
        except (json.JSONDecodeError, TypeError):
            return False 