# 漫游旅伴AI

一个智能旅游助手系统，专注于为老年人和行动不便人士提供旅游咨询服务。

## 项目特点

- 支持多种LLM提供商（DeepSeek、OpenAI、通义千问、Gemini、Claude、文心一言等）
- 专为老年人和行动不便人士设计的旅游推荐
- 情感感知和个性化回复
- 支持景点和住宿信息检索
- 支持Excel和CSV数据源
- 使用FAISS进行高效的向量检索
- 提供FastAPI接口，支持微信小程序对接
- 自动化测试工具，支持批量测试和评估

## 安装说明

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

如果您计划使用特定LLM提供商的SDK，请根据需要取消注释requirements.txt中的相应行并重新运行上述命令。

### 2. 准备API密钥

您需要至少一个大模型API提供商的API密钥。支持以下提供商：

- DeepSeek
- OpenAI
- 通义千问（Qwen/Qianwen）
- Google Gemini
- Anthropic Claude
- 百度文心一言（Baidu/Wenxin）

可以通过环境变量或命令行参数提供API密钥。

## 使用方法

### 命令行交互模式

#### 1. 通过环境变量设置API密钥

```bash
# 设置DeepSeek API密钥
export DEEPSEEK_API_KEY=your_deepseek_api_key

# 或者设置OpenAI API密钥
export OPENAI_API_KEY=your_openai_api_key

# 或者设置通义千问API密钥
export QIANWEN_API_KEY=your_qianwen_api_key

# 或者设置Google Gemini API密钥
export GEMINI_API_KEY=your_gemini_api_key

# 或者设置Anthropic Claude API密钥
export CLAUDE_API_KEY=your_claude_api_key

# 或者设置百度文心一言API密钥（需要API_KEY:SECRET_KEY格式）
export BAIDU_API_KEY=your_api_key:your_secret_key
```

#### 2. 运行命令行程序

```bash
# 使用默认DeepSeek模型
python main.py

# 使用OpenAI模型
python main.py --provider openai

# 使用通义千问模型
python main.py --provider qianwen

# 使用Google Gemini模型
python main.py --provider gemini

# 使用Anthropic Claude模型
python main.py --provider claude

# 使用百度文心一言模型
python main.py --provider baidu

# 指定自定义API密钥
python main.py --provider openai --api-key your_openai_api_key

# 指定模型名称
python main.py --provider openai --model gpt-4
```

### FastAPI 服务模式（用于微信小程序对接）

#### 1. 启动API服务

```bash
# 使用默认配置启动API服务
python api.py

# 通过环境变量配置服务
export LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY=your_deepseek_api_key
python api.py
```

服务启动后，可通过以下URL访问：
- API文档: `http://localhost:8000/`
- 健康检查: `http://localhost:8000/health`
- 查询接口: `http://localhost:8000/query` (POST)

#### 2. API接口使用示例

使用`/query`接口（POST请求）:

```json
// 请求体
{
  "user_input": "大理古城有什么好玩的地方？",
  "user_profile": {
    "age": "68",
    "gender": "男",
    "mobility_status": "行动不便，需要拐杖"
  }
}
```

```json
// 响应体
{
  "response": "【大理古城】\n地址：云南省大理白族自治州大理市古城区\n门票：免费（部分景点需单独购票）\n开放时间：全天开放\n参考游览时间：3-4小时\n联系电话：联系方式未提供\n\n景点简介：大理古城是云南省著名的历史文化名城，曾是南诏国和大理国的都城，有着悠久的历史...\n\n特色看点：五华楼、大理文庙、洋人街等，城内充满白族建筑特色...\n\n【AI旅伴点评】\n\n大理古城非常适合您游览，古城内街道平坦，适合行动不便的游客。建议您从北门进入，那里坡度较缓。特别提醒您，古城对老年人有免费政策，只需出示老年证。如果走累了，古城内有很多休息点和咖啡馆可以歇脚。\n\n----------------------------------------\n上述信息由大理旅伴AI提供，仅供参考，请以实际情况为准。"
}
```

### 命令行选项

- `--provider`: 指定LLM提供商，可选值：`deepseek`, `openai`, `qianwen`, `gemini`, `claude`, `baidu`
- `--api-key`: 指定API密钥
- `--api-base`: 指定自定义API基础URL（如私有部署或自定义端点）
- `--model`: 指定模型名称（根据不同提供商选择不同的模型）

## 自动化测试

系统提供自动化测试工具，可以从Excel文件读取测试案例，并使用LLM自动评估回复质量。

### 运行自动测试

```bash
# 使用默认配置运行测试
python auto_test.py

# 指定LLM提供商和API密钥
python auto_test.py deepseek your_api_key
```

测试结果会保存在以下文件中：
- `test_results.json`: 包含所有测试案例和AI回复
- `evaluation_results.json`: 包含LLM对每个回复的评估结果
- `evaluation_prompts.txt`: 包含用于评估的提示词，可用于手动验证

## 各提供商默认模型

| 提供商 | 默认模型 | 可选模型示例 |
|--------|---------|------------|
| DeepSeek | deepseek-chat | deepseek-chat, deepseek-coder |
| OpenAI | gpt-3.5-turbo | gpt-4, gpt-4-turbo |
| 通义千问 | qwen-max | qwen-plus, qwen-turbo |
| Google Gemini | gemini-1.5-pro | gemini-1.5-flash |
| Anthropic Claude | claude-3-opus-20240229 | claude-3-sonnet, claude-3-haiku |
| 百度文心一言 | ERNIE-4.0-8K | ERNIE-4.0-8K-0205, ERNIE-Speed |

## 文件结构

```
.
├── main.py                      # 主程序
├── api.py                       # FastAPI服务
├── auto_test.py                 # 自动化测试工具
├── database_retrieval.py        # 数据检索模块
├── llm_providers/
│   ├── __init__.py              # 包初始化文件
│   ├── base_model.py            # 基础模型接口
│   ├── model_factory.py         # 模型工厂类
│   ├── deepseek_model.py        # DeepSeek模型实现
│   ├── openai_model.py          # OpenAI模型实现
│   ├── qwen_model.py            # 通义千问模型实现
│   ├── gemini_model.py          # Google Gemini模型实现
│   ├── claude_model.py          # Anthropic Claude模型实现
│   └── baidu_model.py           # 百度文心一言模型实现
├── data/                        # 数据文件目录
│   ├── 大理景点整理.xlsx         # 景点数据（Excel格式）
│   ├── 大理景点整理.csv          # 景点数据（CSV格式）
│   ├── 意向合作商家.xlsx         # 商家数据（Excel格式）
│   └── 意向合作商家.csv          # 商家数据（CSV格式）
├── test_data/                   # 测试数据目录
│   └── AI旅伴-测试集(1).xlsx     # 测试案例
└── requirements.txt             # 依赖列表
```

## 添加新的模型支持

要添加新的LLM提供商支持，请执行以下步骤：

1. 在`llm_providers/`目录下创建新的模型实现文件，如`new_model.py`
2. 实现继承自`BaseLLMModel`的模型类
3. 在`model_factory.py`的`MODEL_MAPPING`中添加新的模型映射
4. 在`llm_providers/__init__.py`中导入并添加到`__all__`列表

示例代码：

```python
# llm_providers/new_model.py
from llm_providers.base_model import BaseLLMModel

class NewModel(BaseLLMModel):
    # 实现必要的方法
    pass

# 在model_factory.py中添加
MODEL_MAPPING = {
    "deepseek": DeepSeekModel,
    "openai": OpenAIModel,
    # ... 其他模型
    "new": NewModel,  # 新添加的模型
}
```

## 系统特性

- **多模态支持**: 兼容多种LLM提供商，可随时切换
- **卡片化界面**: 所有回复采用信息卡片格式，清晰直观
- **向量检索优化**: 使用FAISS进行高效的向量相似度检索
- **多格式数据源**: 支持Excel和CSV数据格式
- **微信小程序集成**: 提供API接口，方便微信小程序调用
- **老年人适配**: 专门针对老年人用户体验优化
- **自动化测试**: 内置测试和评估工具

## 环境要求

- Python 3.8+
- 必要的Python包（详见requirements.txt）
- 互联网连接（用于API调用）

## 部署建议

在生产环境中部署时，建议：
1. 使用Nginx等反向代理服务器
2. 配置SSL证书确保通信安全
3. 使用环境变量管理API密钥，避免硬编码
4. 使用进程管理工具（如Supervisor或PM2）确保服务稳定运行

## 许可证

[MIT License](LICENSE) 