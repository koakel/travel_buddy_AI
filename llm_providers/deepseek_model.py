import requests
from typing import Dict, Any, List
import json
import re

from llm_providers.base_model import BaseLLMModel


class DeepSeekModel(BaseLLMModel):
    """DeepSeek API implementation of LLM provider"""

    def __init__(self, api_key: str, api_base: str = "https://api.deepseek.com/v1"):
        """
        Initialize DeepSeek model
        
        Args:
            api_key: DeepSeek API key
            api_base: DeepSeek API base URL
        """
        super().__init__(api_key, api_base)
        self.model_name = "deepseek-chat"  # Default model, can be updated
        

    def generate_response(self,
                          user_query: str,
                          retrieved_info: str,
                          temperature: float = 0.7,
                          max_tokens: int = 800) -> str:
        """
        Generate response using DeepSeek API
        
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
                 "content": "你是一位专业的旅伴智能助手，擅长提供旅游咨询和帮助，特别关注老年人和行动不便人士的需求。请始终以JSON格式返回你的回答。"},
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

            # Extract generated text from DeepSeek response format
            generated_text = result["choices"][0]["message"]["content"]
            
            
            # 尝试提取并解析JSON
            try:
                # 尝试多种方式提取JSON
                json_object = None
                
                # 1. 尝试直接解析整个响应
                if self._is_valid_json(generated_text):
                    json_object = json.loads(generated_text)
                else:
                    # 2. 尝试找到并提取JSON代码块
                    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', generated_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1).strip()
                        if self._is_valid_json(json_str):
                            json_object = json.loads(json_str)
                    
                    # 3. 尝试使用正则表达式匹配花括号内容
                    if not json_object:
                        bracket_match = re.search(r'({.*})', generated_text, re.DOTALL)
                        if bracket_match:
                            json_str = bracket_match.group(1).strip()
                            if self._is_valid_json(json_str):
                                json_object = json.loads(json_str)
                
                # 如果成功解析JSON
                if json_object:
                    # 确保关键字段存在
                    if "needs_card" not in json_object:
                        json_object["needs_card"] = "no"
                    if "content" not in json_object:
                        json_object["content"] = generated_text
                    
                    return json.dumps(json_object, ensure_ascii=False)
                else:
                    # 所有解析方法都失败了，使用默认结构
                    raise json.JSONDecodeError("无法提取有效JSON", generated_text, 0)
                    
            except (json.JSONDecodeError, AttributeError) as e:
                # 不记录为"错误"，而是记录为"处理情况"，避免误导
                print(f"JSON处理情况: 使用默认格式")
                
                # 如果解析失败，检查是否包含旧标记
                if "[CARD]" in generated_text:
                    default_response = {
                        "needs_card": "yes",
                        "content": generated_text.replace("[CARD]", "").strip()
                    }
                else:
                    # 普通文本响应
                    default_response = {
                        "needs_card": "no", 
                        "content": generated_text
                    }
                
                return json.dumps(default_response, ensure_ascii=False)

        except Exception as e:
            print(f"API调用处理: {str(e)}")
            # Fallback response in case of error
            default_response = {
                "needs_card": "no",
                "content": "抱歉，我暂时无法为您提供信息。请稍后再试或联系客服。"
            }
            return json.dumps(default_response, ensure_ascii=False)
            