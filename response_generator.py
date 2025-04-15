import os
import json
import requests
import re
from typing import Dict, Any, List, Optional


class DeepSeekResponseGenerator:
    """使用DeepSeek API生成旅伴响应的类"""

    def __init__(self, api_key: str, api_base: str = "https://api.deepseek.com/v1"):
        """
        初始化DeepSeek响应生成器

        Args:
            api_key: DeepSeek API密钥
            api_base: DeepSeek API基础URL
        """
        self.api_key = api_key
        self.api_base = api_base
        self.model = "deepseek-chat"  # 根据DeepSeek提供的模型名称进行调整
        
        # 基础提示词模板，与BaseLLMModel保持一致
        self.PROMPT_TEMPLATE = """你是一位专注于服务中老年人的旅游伴侣，以下是你的角色特点：
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

    def create_prompt(self,
                      user_query: str,
                      retrieved_info: str,
                      personality: Dict[str, float] = None,
                      user_profile: Dict[str, str] = None,
                      user_intent: str = None,
                      user_entities: List[Dict[str, Any]] = None,
                      user_emotion: str = None) -> str:
        """
        创建发送给DeepSeek模型的提示文本

        Args:
            user_query: 用户查询
            retrieved_info: 从数据库检索到的信息
            personality: 智能体性格参数（可选）
            user_profile: 用户个人信息（可选）
            user_intent: 用户意图（可选）
            user_entities: 用户提到的实体（可选）
            user_emotion: 用户情绪状态（可选）

        Returns:
            构建好的提示文本
        """
        # 使用基础模板但添加额外上下文信息（如果提供）
        additional_context = ""
        
        # 添加用户画像信息（如果有）
        if user_profile:
            additional_context += f"\n# 用户画像：\n"
            additional_context += f"- 年龄：{user_profile.get('age', '未知')}\n"
            additional_context += f"- 性别：{user_profile.get('gender', '未知')}\n"
            additional_context += f"- 行动能力：{user_profile.get('mobility_status', '正常')}\n"
        
        # 添加性格参数（如果有）
        if personality:
            additional_context += f"\n# 你的性格特征：\n"
            if 'friendliness' in personality:
                additional_context += f"- 友善度: {personality.get('friendliness', 0.7)}（高表示亲切和蔼）\n"
            if 'professionalism' in personality:
                additional_context += f"- 专业度: {personality.get('professionalism', 0.8)}（高表示精准、专业）\n"
            if 'patience' in personality:
                additional_context += f"- 耐心度: {personality.get('patience', 0.9)}（高表示耐心细致）\n"
            if 'initiative' in personality:
                additional_context += f"- 主动度: {personality.get('initiative', 0.6)}（高表示积极推荐）\n"
            if 'empathy' in personality:
                additional_context += f"- 共情度: {personality.get('empathy', 0.8)}（高表示温暖且善于倾听）\n"
        
        # 添加用户意图和实体信息（如果有）
        if user_intent or user_entities:
            additional_context += f"\n# 用户的意图和具体需求：\n"
            if user_intent:
                additional_context += f"- 用户的核心意图是：{user_intent}\n"
            if user_entities:
                entities_str = ", ".join(
                    [f"{entity['type']}: {entity['value']}" for entity in user_entities]) if user_entities else "无具体实体信息"
                additional_context += f"- 用户提到的具体实体信息是：{entities_str}\n"
        
        # 添加用户情绪信息（如果有）
        if user_emotion:
            additional_context += f"\n# 用户当前情绪状态：\n- {user_emotion}\n"
        
        # 将基础模板与额外上下文结合
        prompt = self.PROMPT_TEMPLATE.format(
            user_query=user_query,
            retrieved_info=retrieved_info
        )
        
        # 在适当位置插入额外上下文（在用户提问和检索信息之间）
        if additional_context:
            prompt_parts = prompt.split("接下来请你根据以上要求开展与用户的对话：")
            if len(prompt_parts) == 2:
                prompt = prompt_parts[0] + additional_context + "\n接下来请你根据以上要求开展与用户的对话：" + prompt_parts[1]
            else:
                # 如果分割失败，直接在末尾添加额外上下文
                prompt += additional_context
        
        return prompt

    def generate_response(self,
                          user_query: str,
                          retrieved_info: str,
                          personality: Dict[str, float] = None,
                          user_profile: Dict[str, str] = None,
                          user_intent: str = None,
                          user_entities: List[Dict[str, Any]] = None,
                          user_emotion: str = None,
                          temperature: float = 0.7,
                          max_tokens: int = 800) -> str:
        """
        生成回复

        Args:
            user_query: 用户输入文本
            retrieved_info: 从数据库检索到的信息
            personality: 智能体性格参数
            user_profile: 用户个人信息
            user_intent: 用户意图
            user_entities: 用户提到的实体
            user_emotion: 用户情绪状态
            temperature: 生成多样性参数
            max_tokens: 最大生成令牌数

        Returns:
            生成的回复文本
        """
        prompt = self.create_prompt(
            user_query=user_query,
            retrieved_info=retrieved_info,
            personality=personality,
            user_profile=user_profile,
            user_intent=user_intent,
            user_entities=user_entities,
            user_emotion=user_emotion
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
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

            # 根据DeepSeek API的返回格式提取生成的文本
            generated_text = result["choices"][0]["message"]["content"]
            
            # 尝试提取并解析JSON
            try:
                # 使用正则表达式提取JSON部分（兼容可能的格式问题）
                json_match = re.search(r'```json\s*(.*?)\s*```', generated_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # 如果没有找到json代码块包装，直接尝试解析整个文本
                    json_str = generated_text
                
                # 清理可能的干扰字符（如果模型输出了额外的引导文本）
                json_str = json_str.strip()
                if json_str.startswith('```') and json_str.endswith('```'):
                    json_str = json_str[3:-3].strip()
                
                # 解析JSON
                parsed = json.loads(json_str)
                return json.dumps(parsed, ensure_ascii=False)
                
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"JSON解析错误: {e}，将按文本处理")
                # 如果解析失败，使用默认结构
                default_response = {
                    "needs_card": "no",
                    "content": generated_text
                }
                return json.dumps(default_response, ensure_ascii=False)

        except Exception as e:
            print(f"API调用错误: {e}")
            # 发生错误时返回备用回复
            default_response = {
                "needs_card": "no",
                "content": "抱歉，我暂时无法为您提供信息。请稍后再试或联系客服。"
            }
            return json.dumps(default_response, ensure_ascii=False)


# 使用示例
def main():
    # 从环境变量获取API密钥，确保安全性
    api_key = os.environ.get("DEEPSEEK_API_KEY", "your_api_key_here")

    # 创建响应生成器实例
    generator = DeepSeekResponseGenerator(api_key=api_key)

    # 示例输入
    personality = {
        "friendliness": 0.9,
        "professionalism": 0.8,
        "patience": 0.9,
        "initiative": 0.7,
        "empathy": 0.8
    }

    user_profile = {
        "age": "65岁",
        "gender": "女",
        "mobility_status": "行动略微缓慢，需要偶尔休息"
    }

    user_intent = "查询适合老年人的大理景点"

    user_entities = [
        {"type": "地点", "value": "大理"},
        {"type": "需求", "value": "适合老年人"},
        {"type": "偏好", "value": "避免爬山"}
    ]

    user_emotion = "略显担忧，担心旅途劳累"

    retrieved_info = """
    1. 大理古城：
       - 位置：大理市古城区
       - 特点：平坦的石板路，历史悠久，文化氛围浓厚
       - 适合度：★★★★★（非常适合老年人，地势平坦）
       - 门票：免费
       - 设施：有休息区，公共卫生间，轮椅通道
    
    2. 洱海公园：
       - 位置：大理市北部
       - 特点：湖景优美，有专门的观景平台和座椅
       - 适合度：★★★★☆（非常适合，有休息设施）
       - 门票：免费
       - 设施：有休息区，餐厅，无障碍通道
    
    3. 崇圣寺三塔：
       - 位置：大理市北郊
       - 特点：历史文化遗址，环境清幽
       - 适合度：★★★☆☆（地势较为平坦，但需步行）
       - 门票：121元
       - 设施：有休息区，电瓶车服务（另收费）
        """

    # 用户查询
    user_query = "我妈妈腿脚不方便，想去大理玩，有什么推荐的地方吗？"

    # 生成响应
    response = generator.generate_response(
        user_query=user_query,
        retrieved_info=retrieved_info,
        personality=personality,
        user_profile=user_profile,
        user_intent=user_intent,
        user_entities=user_entities,
        user_emotion=user_emotion
    )

    print("用户查询:", user_query)
    print("生成的回复：")
    print(response)


if __name__ == "__main__":
    main()