# 配置文件
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# DeepSeek API配置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

# 智能体默认性格配置
DEFAULT_PERSONALITY = {
    "friendliness": 0.9,  # 非常友善
    "professionalism": 0.8,  # 相当专业
    "patience": 0.9,  # 非常耐心
    "initiative": 0.7,  # 中等主动
    "empathy": 0.8,  # 高度共情
}

# 响应生成配置
RESPONSE_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 800,
    "model": "deepseek-chat",
}

# 安全配置
SAFETY_THRESHOLDS = {
    "offensive_content": 0.7,
    "harmful_content": 0.6,
}