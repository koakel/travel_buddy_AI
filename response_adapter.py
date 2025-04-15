from typing import Dict, List, Any, Optional
from response_generator import DeepSeekResponseGenerator
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEFAULT_PERSONALITY, RESPONSE_CONFIG


class ResponseAdapter:
    """
    响应适配器：连接各模块并生成最终响应
    """

    def __init__(self):
        """初始化响应适配器"""
        self.generator = DeepSeekResponseGenerator(
            api_key=DEEPSEEK_API_KEY,
            api_base=DEEPSEEK_API_BASE
        )
        self.default_personality = DEFAULT_PERSONALITY

    def adapt_personality(self,
                          intent_type: str,
                          emotion: str,
                          user_age: Optional[str] = None) -> Dict[str, float]:
        """
        根据用户意图、情绪和年龄调整智能体人格

        Args:
            intent_type: 意图类型(如询问、投诉、闲聊等)
            emotion: 情绪状态
            user_age: 用户年龄

        Returns:
            调整后的人格参数
        """
        personality = self.default_personality.copy()

        # 根据意图调整
        if intent_type == "inquiry":  # 询问信息
            personality["professionalism"] = 0.9
        elif intent_type == "complaint":  # 投诉
            personality["empathy"] = 0.9
            personality["patience"] = 1.0
        elif intent_type == "casual_chat":  # 闲聊
            personality["friendliness"] = 1.0
            personality["initiative"] = 0.8

        # 根据情绪调整
        if "愤怒" in emotion or "失望" in emotion or "担忧" in emotion:
            personality["empathy"] = min(1.0, personality["empathy"] + 0.1)
            personality["patience"] = min(1.0, personality["patience"] + 0.1)
        elif "开心" in emotion or "兴奋" in emotion:
            personality["friendliness"] = min(1.0, personality["friendliness"] + 0.1)

        # 根据年龄调整
        if user_age and user_age.isdigit():
            age = int(user_age)
            if age > 65:
                personality["patience"] = min(1.0, personality["patience"] + 0.1)

        return personality

    def generate_final_response(self,
                                user_input: str,
                                intent_data: Dict[str, Any],
                                emotion_data: Dict[str, Any],
                                user_data: Dict[str, Any],
                                retrieved_info: str) -> str:
        """
        生成最终响应

        Args:
            user_input: 用户输入
            intent_data: 意图和实体识别结果
            emotion_data: 情感分析结果
            user_data: 用户信息
            retrieved_info: 数据库检索结果

        Returns:
            生成的最终响应
        """
        # 提取意图和实体
        user_intent = intent_data.get("intent", "未识别")
        user_entities = intent_data.get("entities", [])

        # 提取情绪
        emotion = emotion_data.get("emotion", "中性")

        # 提取用户profile
        user_profile = {
            "age": user_data.get("age", "未知"),
            "gender": user_data.get("gender", "未知"),
            "mobility_status": user_data.get("mobility_status", "正常")
        }

        # 根据意图和情绪调整性格
        personality = self.adapt_personality(
            intent_type=intent_data.get("intent_type", "inquiry"),
            emotion=emotion,
            user_age=user_data.get("age")
        )

        # 调用响应生成器生成回复
        response = self.generator.generate_response(
            user_query=user_input,
            retrieved_info=retrieved_info,
            personality=personality,
            user_profile=user_profile,
            user_intent=user_intent,
            user_entities=user_entities,
            user_emotion=emotion,
            temperature=RESPONSE_CONFIG["temperature"],
            max_tokens=RESPONSE_CONFIG["max_tokens"]
        )

        return response


# 使用示例
def example_usage():
    adapter = ResponseAdapter()

    # 模拟其他模块的输出
    intent_data = {
        "intent": "查询景点信息",
        "intent_type": "inquiry",
        "entities": [
            {"type": "location", "value": "大理"},
            {"type": "preference", "value": "适合老人"}
        ]
    }

    emotion_data = {
        "emotion": "略显担忧",
        "confidence": 0.85
    }

    user_data = {
        "age": "68",
        "gender": "女",
        "mobility_status": "需要拐杖辅助"
    }

    retrieved_info = """
    1. 大理古城：平坦石板路，适合慢步游览，有多处休息场所，免费入城
    2. 喜洲古镇：民居古朴，游览线路短，适合老年人，设有专门休息区
    3. 洱海游船：坐船欣赏风景，不需要行走，适合行动不便老人
    """

    user_input = "我妈妈腿脚不方便，想去大理玩，有什么推荐的地方吗？"

    response = adapter.generate_final_response(
        user_input=user_input,
        intent_data=intent_data,
        emotion_data=emotion_data,
        user_data=user_data,
        retrieved_info=retrieved_info
    )

    print(f"用户输入: {user_input}")
    print(f"生成回复: {response}")


if __name__ == "__main__":
    example_usage()