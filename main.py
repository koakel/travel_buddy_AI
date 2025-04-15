import sys
import os
import re
import argparse
import json
from typing import Dict, Any, List, Optional
from database_retrieval import DataRetrievalModule
from llm_providers.model_factory import ModelFactory
from auto_test import AutoTester


class TravelCompanionAI:
    """漫游旅伴AI主类：整合数据检索与响应生成"""

    def __init__(self, provider: str = "deepseek", api_key: Optional[str] = None,
                 api_base: Optional[str] = None, model_name: Optional[str] = None,
                 skip_interactive: bool = False):
        """
        初始化旅伴AI系统

        Args:
            provider: LLM提供商名称 ('deepseek', 'openai', 'qianwen', 'gemini', 'claude', 'baidu')
            api_key: API密钥，默认从环境变量获取
            api_base: 自定义API基础URL
            model_name: 模型名称，特定于所选提供商
            skip_interactive: 是否跳过交互式用户信息收集（用于自动化测试）
        """
        # 规范化提供商名称
        self.provider = provider.lower()
        
        # 获取API密钥
        self.api_key = api_key or os.environ.get(f"{self.provider.upper()}_API_KEY")
        if not self.api_key:
            print(f"错误: 未设置{self.provider.upper()}_API_KEY，请设置环境变量或直接传入")
            sys.exit(1)

        # 初始化模块
        print("正在初始化数据检索模块...")
        self.retrieval = DataRetrievalModule()

        print(f"正在初始化{self.provider}响应生成模块...")
        # 创建模型配置
        model_config = {}
        if model_name:
            model_config["model_name"] = model_name
            
        try:
            # 使用工厂类获取指定的LLM提供商
            self.generator = ModelFactory.get_model(
                provider=self.provider,
                api_key=self.api_key,
                api_base=api_base,
                config=model_config
            )
            print(f"成功初始化{self.provider}模型，使用的具体模型为: {self.generator.model_name}")
        except ValueError as e:
            print(f"错误: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"初始化{self.provider}响应生成模块时发生错误: {e}")
            sys.exit(1)

        # 默认用户画像
        self.user_profile = {
            "age": "未知",
            "gender": "未知",
            "mobility_status": "正常"
        }
        
        # 设置自动化测试标志
        self.skip_interactive = skip_interactive

        print("系统初始化完成，准备开始对话...")

    def simple_intent_recognition(self, text: str) -> Dict[str, Any]:
        """
        简单的意图识别（暂时替代模块1）

        Args:
            text: 用户输入文本

        Returns:
            识别结果
        """
        # 简单的关键词匹配
        keywords = {
            "询问景点": ["景点", "好玩", "玩什么", "游玩", "参观", "景区", "打卡", "必去"],
            "询问住宿": ["住宿", "酒店", "民宿", "客栈", "住哪", "住在哪", "住宿推荐"],
            "询问交通": ["交通", "怎么去", "怎么到", "路线", "公交", "打车", "导航"],
            "询问美食": ["美食", "吃什么", "餐厅", "好吃的", "特色菜", "小吃"],
            "闲聊": ["你好", "你是谁", "聊聊", "陪我", "无聊"],
            "投诉": ["不满", "投诉", "差评", "退款", "糟糕", "不好", "差", "坏"]
        }

        # 默认意图
        intent = {
            "intent": "未识别意图",
            "intent_type": "inquiry",
            "entities": []
        }

        # 规则匹配意图
        for intent_name, words in keywords.items():
            if any(word in text for word in words):
                intent["intent"] = intent_name
                break

        # 简单提取地点实体
        locations = ["大理", "古城", "洱海", "苍山", "崇圣寺", "喜洲", "双廊", "三塔"]
        for loc in locations:
            if loc in text:
                intent["entities"].append({"type": "location", "value": loc})

        # 简单提取偏好实体
        preferences = ["适合老人", "无障碍", "不爬山", "轮椅", "行动不便", "平缓", "休息"]
        for pref in preferences:
            if pref in text:
                intent["entities"].append({"type": "preference", "value": pref})

        return intent

    def simple_emotion_analysis(self, text: str) -> Dict[str, Any]:
        """
        简单的情感分析（暂时替代模块2）

        Args:
            text: 用户输入文本

        Returns:
            情感分析结果
        """
        # 情绪关键词映射
        emotion_keywords = {
            "担忧": ["担心", "害怕", "焦虑", "不安", "怕", "困难", "问题"],
            "愤怒": ["生气", "愤怒", "讨厌", "烦", "不满", "投诉"],
            "开心": ["开心", "高兴", "快乐", "满意", "期待", "喜欢"],
            "困惑": ["不懂", "疑惑", "为什么", "怎么", "如何"],
            "中性": []  # 默认
        }

        # 判断情绪
        for emotion, words in emotion_keywords.items():
            if any(word in text for word in words):
                return {"emotion": emotion, "confidence": 0.8}

        # 默认为中性
        return {"emotion": "中性", "confidence": 0.6}

    def collect_user_info(self) -> None:
        """收集用户基本信息"""
        print("\n" + "=" * 50)
        print("欢迎使用【漫游旅伴AI】，很高兴能为您提供服务！")
        print("为了给您提供更贴心的陪伴服务，需要了解您的一些基本信息。")
        print("=" * 50)

        # 收集年龄
        while True:
            age_input = input("\n请问您今年多大年纪了？(直接输入数字或描述如'六十多'): ")
            if age_input:
                if age_input.isdigit():
                    self.user_profile["age"] = age_input
                    break
                else:
                    # 简单处理文字年龄
                    age_mapping = {
                        "五十": "50", "六十": "60", "七十": "70", "八十": "80",
                        "50多": "55", "60多": "65", "70多": "75", "80多": "85"
                    }
                    for key, value in age_mapping.items():
                        if key in age_input:
                            self.user_profile["age"] = value
                            break
                    if self.user_profile["age"] != "未知":
                        break
                    print("抱歉，无法理解您的年龄输入，请尝试直接输入数字。")
            else:
                print("您选择不提供年龄信息，这也没关系。")
                break

        # 收集性别
        while True:
            gender_input = input("\n请问您的性别是？(男/女/其他): ")
            if gender_input:
                if any(keyword in gender_input for keyword in ["男", "先生", "male"]):
                    self.user_profile["gender"] = "男"
                    break
                elif any(keyword in gender_input for keyword in ["女", "小姐", "女士", "female"]):
                    self.user_profile["gender"] = "女"
                    break
                else:
                    self.user_profile["gender"] = "其他"
                    break
            else:
                print("您选择不提供性别信息，这也没关系。")
                break

        # 收集行动能力
        print("\n为了更好地推荐适合您的体验，请告诉我您的行动能力情况：")
        print("1. 完全自由活动，行动无障碍")
        print("2. 可以自由活动，但步行速度稍慢")
        print("3. 行动需要借助拐杖或他人搀扶")
        print("4. 使用轮椅出行")

        mobility_input = input("请选择适合您的选项编号(1-4)，或直接描述您的情况: ")
        if mobility_input:
            if mobility_input == "1":
                self.user_profile["mobility_status"] = "行动自如"
            elif mobility_input == "2":
                self.user_profile["mobility_status"] = "步行稍慢"
            elif mobility_input == "3":
                self.user_profile["mobility_status"] = "需要拐杖或搀扶"
            elif mobility_input == "4":
                self.user_profile["mobility_status"] = "使用轮椅"
            else:
                self.user_profile["mobility_status"] = mobility_input

        print("\n谢谢您提供的信息！现在我可以更好地为您服务了。")
        print(f"\n您的信息: 年龄-{self.user_profile['age']}, "
              f"性别-{self.user_profile['gender']}, "
              f"行动能力-{self.user_profile['mobility_status']}")

        input("\n按回车键继续...")

    def process_query(self, user_input: str) -> str:
        """
        处理用户输入并生成回复

        Args:
            user_input: 用户输入文本

        Returns:
            生成的回复
        """
        # 数据检索 - 第一阶段，检查是否有特定数据（景点或商家）
        preliminary_data = self.retrieval.process_query(user_input, self.user_profile)

        # 检查是否已经有明确的景点或商家数据
        if preliminary_data["type"] in ["spot", "merchant"]:
            # 找到了特定的景点或商家数据，直接返回卡片
            return preliminary_data["content"]
            
        # 准备检索到的信息文本供LLM使用
        retrieved_info = "我没有找到与您查询相关的具体景点或住宿信息，但我会尽力回答您的问题。"
        if preliminary_data["type"] != "text":
            # 从初步检索中获取内容作为上下文
            retrieved_info = self.retrieval.retrieve(user_input, top_k=3)

        # 生成LLM回复
        llm_response = self.generator.generate_response(
            user_query=user_input,
            retrieved_info=retrieved_info,
            temperature=0.7,
            max_tokens=500
        )

        # 将LLM生成的内容传递给数据检索模块处理
        final_data = self.retrieval.process_query(user_input, self.user_profile, llm_content=llm_response)
        
        # 根据响应类型返回不同格式
        if final_data["type"] == "text":
            # 如果是纯文本类型，直接返回内容，不使用卡片格式
            return final_data["content"]
        else:
            # 其他类型返回格式化内容
            return final_data["content"]

    def run(self) -> None:
        """主运行循环"""
        # 收集用户信息
        if not self.skip_interactive:
            self.collect_user_info()

            # 显示主菜单
            print("\n" + "=" * 50)
            print("【漫游旅伴AI】为您服务！您可以：")
            print("1. 聊聊大理的旅游景点")
            print("2. 咨询适合您的住宿推荐")
            print("3. 了解大理的交通或美食")
            print("4. 和我随便聊聊")
            print("输入'退出'结束对话")
            print("=" * 50)

        # 自动测试模式下不执行交互循环
        if self.skip_interactive:
            return

        # 主交互循环
        while True:
            print("\n")
            user_input = input("您想了解什么？> ")

            if user_input.lower() in ["退出", "exit", "quit", "bye"]:
                print("\n感谢您使用【漫游旅伴AI】，祝您旅途愉快！再见👋")
                break

            if not user_input.strip():
                continue

            # 处理用户输入
            print("\n正在思考...")
            response = self.process_query(user_input)

            # 显示回复
            print("\n【旅伴回复】:")
            print(response)
            print("\n" + "-" * 50)


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        # Create default configuration
        default_config = {
            "active_provider": "deepseek",
            "providers": {
                "deepseek": {
                    "api_key": "",
                    "model_name": "deepseek-chat",
                    "api_base": "https://api.deepseek.com/v1"
                }
            },
            "common_settings": {
                "temperature": 0.7,
                "max_tokens": 800,
                "data_dir": "data"
            }
        }
        
        # Save default configuration
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
            
        print(f"创建了默认配置文件: {config_path}")
        return default_config
    
    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config


def process_query(user_query: str, retrieval_module: DataRetrievalModule, 
                  config: Dict[str, Any]) -> str:
    """
    Process a user query
    
    Args:
        user_query: User query string
        retrieval_module: Database retrieval module
        config: Configuration dictionary
        
    Returns:
        Response to the user query
    """
    # Retrieve information from the database
    retrieved_info = retrieval_module.retrieve(user_query, top_k=3)
    
    # Get active provider configuration
    active_provider = config.get("active_provider", "deepseek")
    provider_config = config.get("providers", {}).get(active_provider, {})
    
    # Merge provider config with common settings
    model_config = {**config.get("common_settings", {}), **provider_config}
    
    # Create LLM model instance
    model = ModelFactory.get_model(
        provider=active_provider,
        api_key=provider_config.get("api_key", ""),
        api_base=provider_config.get("api_base", ""),
        config={"model_name": provider_config.get("model_name", "")}
    )
    
    # Generate response
    response = model.generate_response(
        user_query=user_query,
        retrieved_info=retrieved_info,
        temperature=model_config.get("temperature", 0.7),
        max_tokens=model_config.get("max_tokens", 800)
    )
    
    # 尝试解析JSON响应
    try:
        # 检查是否可能为JSON格式
        if '{' in response and '}' in response:
            # 清理可能的Markdown代码块
            json_str = response.strip()
            if "```json" in json_str:
                # 提取JSON部分
                start = json_str.find("```json") + 7
                end = json_str.rfind("```")
                if end > start:
                    json_str = json_str[start:end].strip()
            
            # 解析JSON
            response_json = json.loads(json_str)
            needs_card = response_json.get("needs_card", "no").lower() == "yes"
            content = response_json.get("content", "")
            
            # 如果需要卡片格式，则转换为卡片
            if needs_card:
                card_data = retrieval_module.format_llm_card(user_query, content, card_type="llm")
                return card_data
            else:
                # 否则直接返回内容
                return content
                
        # 处理旧格式的响应
        elif response.startswith("[CARD]"):
            # 移除标记
            clean_response = response[6:].strip()
            # 将回复转换为卡片格式
            card_data = retrieval_module.format_llm_card(user_query, clean_response, card_type="llm")
            return card_data
        elif response.startswith("[TEXT]"):
            # 移除标记返回纯文本
            return response[6:].strip()
        else:
            # 没有标记的情况，保持原样返回
            return response
            
    except (json.JSONDecodeError, AttributeError) as e:
        # 解析失败但仍需要返回有用的回复
        if response.startswith("[CARD]"):
            # 移除标记
            clean_response = response[6:].strip()
            # 将回复转换为卡片格式
            card_data = retrieval_module.format_llm_card(user_query, clean_response, card_type="llm")
            return card_data
        elif response.startswith("[TEXT]"):
            # 移除标记返回纯文本
            return response[6:].strip()
        else:
            # 没有标记的情况，保持原样返回
            return response


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="TravelBuddy AI 旅游助手")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--provider", type=str, help="指定使用的LLM提供商")
    parser.add_argument("--test", action="store_true", help="运行自动测试模式")
    parser.add_argument("--test-file", type=str, default="test_data/AI旅伴-测试集(1).xlsx", help="测试用例文件路径")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override active provider if specified in command line
    if args.provider:
        if args.provider in config.get("providers", {}):
            config["active_provider"] = args.provider
            print(f"使用命令行指定的提供商: {args.provider}")
        else:
            print(f"警告: 命令行指定的提供商 '{args.provider}' 不在配置文件中，使用默认提供商")
    
    active_provider = config.get("active_provider", "deepseek")
    provider_config = config.get("providers", {}).get(active_provider, {})
    
    # Initialize data retrieval module
    data_dir = config.get("common_settings", {}).get("data_dir", "data")
    retrieval_module = DataRetrievalModule(data_dir=data_dir)
    
    # Check if running in test mode
    if args.test:
        # Run automated testing
        tester = AutoTester(
            ai_class=TravelCompanionAI,
            provider=active_provider,
            test_file_path=args.test_file,
            api_key=provider_config.get("api_key", ""),
        )
        tester.load_test_cases()
        tester.run_tests()
        # 保存结果
        tester.save_results()
        
        # 使用LLM进行评估
        tester.run_evaluations()
        
        # 保存评估结果
        tester.save_evaluations()
        
        # 生成评估提示词（可选，用于手动验证）
        tester.generate_evaluation_prompts()
        
        print("\n自动测试和评估完成！")
        return
    
    # Interactive mode
    print("欢迎使用 TravelBuddy AI 旅游助手！输入 'exit' 或 'quit' 退出。")
    
    while True:
        # Get user input
        user_query = input("\n请输入您的问题: ")
        
        # Check if user wants to exit
        if user_query.lower() in ["exit", "quit", "退出"]:
            print("感谢使用 TravelBuddy AI，再见！")
            break
        
        # Process query
        try:
            response = process_query(user_query, retrieval_module, config)
            print(f"\n{response}")
        except Exception as e:
            print(f"处理查询时出错: {e}")


if __name__ == "__main__":
    main()
