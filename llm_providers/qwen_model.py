import requests
from typing import Dict, Any, List

from llm_providers.base_model import BaseLLMModel


class QwenModel(BaseLLMModel):
    """Qwen (通义千问) API implementation of LLM provider"""

    def __init__(self, api_key: str, api_base: str = "https://dashscope.aliyuncs.com/api/v1"):
        """
        Initialize Qwen model
        
        Args:
            api_key: Qwen API key
            api_base: Qwen API base URL
        """
        super().__init__(api_key, api_base)
        self.model_name = "qwen-max"  # Default model, can be changed to other models like qwen-plus
        
    def create_prompt(self,
                      personality: Dict[str, float],
                      user_profile: Dict[str, str],
                      user_intent: str,
                      user_entities: List[Dict[str, Any]],
                      user_emotion: str,
                      retrieved_info: str) -> str:
        """
        Create prompt for Qwen model
        
        Args:
            personality: Personality parameters
            user_profile: User profile information
            user_intent: User intent
            user_entities: Entities mentioned by user
            user_emotion: User emotion
            retrieved_info: Information retrieved from database
            
        Returns:
            Formatted prompt for Qwen
        """
        # Format entities into readable string
        entities_str = ", ".join(
            [f"{entity['type']}: {entity['value']}" for entity in user_entities]) if user_entities else "无具体实体信息"

        prompt = f"""你是一位旅游智能助手，根据以下信息与用户互动交流：

                # 你的性格特征：
                - 友善度: {personality.get('friendliness', 0.7)}（高表示亲切和蔼）
                - 专业度: {personality.get('professionalism', 0.8)}（高表示精准、专业）
                - 耐心度: {personality.get('patience', 0.9)}（高表示耐心细致）
                - 主动度: {personality.get('initiative', 0.6)}（高表示积极推荐）
                - 共情度: {personality.get('empathy', 0.8)}（高表示温暖且善于倾听）

                # 用户画像：
                - 年龄：{user_profile.get('age', '未知')}
                - 性别：{user_profile.get('gender', '未知')}
                - 行动能力：{user_profile.get('mobility_status', '正常')}

                # 用户的意图和具体需求：
                - 用户的核心意图是：{user_intent}
                - 用户提到的具体实体信息是：{entities_str}

                # 用户当前情绪状态：
                - {user_emotion}

                # 查询数据库后获得的景点信息：
                {retrieved_info}

                # 回复要求：
                - 请用符合你的人格参数的语气，以简洁、温暖、耐心的风格回复用户；
                - 针对用户的情绪做简单关怀（如果是负面情绪，请表达同理心并适当安抚）；
                - 基于用户的画像信息（如年龄和行动能力），主动推荐最适合用户体验且评价良好的1个或2个景点；
                - 推荐时附带简单理由（例如适合老人行动、景色优美、服务周到等）；
                - 回复应简洁明了，避免使用过于复杂的术语，最好控制在250字以内；
                - 如果是老年用户，请使用更大的字号，语言更简单直白。

                现在，请根据以上信息生成回复：
                """
        return prompt

    def generate_response(self,
                          personality: Dict[str, float],
                          user_profile: Dict[str, str],
                          user_intent: str,
                          user_entities: List[Dict[str, Any]],
                          user_emotion: str,
                          retrieved_info: str,
                          temperature: float = 0.7,
                          max_tokens: int = 800) -> str:
        """
        Generate response using Qwen API
        
        Args:
            personality: Personality parameters
            user_profile: User profile information
            user_intent: User intent
            user_entities: Entities mentioned by user
            user_emotion: User emotion
            retrieved_info: Information retrieved from database
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        prompt = self.create_prompt(
            personality=personality,
            user_profile=user_profile,
            user_intent=user_intent,
            user_entities=user_entities,
            user_emotion=user_emotion,
            retrieved_info=retrieved_info
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "input": {
                "messages": [
                    {"role": "system", 
                     "content": "你是一位专业的旅伴智能助手，擅长提供旅游咨询和帮助，特别关注老年人和行动不便人士的需求。"},
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        }

        try:
            response = requests.post(
                f"{self.api_base}/services/aigc/text-generation/generation",  # Qwen API endpoint
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()

            # Extract generated text from Qwen response format
            generated_text = result["output"]["text"]
            return generated_text

        except Exception as e:
            print(f"API调用错误: {e}")
            # Fallback response in case of error
            return "抱歉，我暂时无法为您提供信息。请稍后再试或联系客服。" 