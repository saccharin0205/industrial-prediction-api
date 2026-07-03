# 工业 Agent 分析系统

> 集 **ML 成分预测 + RAG 知识检索 + 工艺标准查询** 于一体的 LLM Agent 智能分析服务。

## 项目简介

将高纯铝工艺预测模型封装为 **Agent 服务**。用户用自然语言提问，LLM 自主决定调用哪个工具、传什么参数，完成预测分析、知识检索、标准查询等任务。

模拟工业场景中「操作工用自然语言提问 → AI 自动预测 + 查规范 + 出报告」的完整链路。

## 架构

```
用户自然语言问题
       ↓
  FastAPI /chat
       ↓
  ┌─ Agent 循环 ─────────────────────────────┐
  │                                           │
  │  LLM (DeepSeek) 自主决定调用哪些工具：     │
  │                                           │
  │  ├── predict_purity   → ML 模型预测成分    │
  │  ├── search_knowledge → RAG 检索工艺知识库 │
  │  └── query_standard   → 查询工艺标准参数   │
  │                                           │
  │  拿到结果 → LLM 生成分析报告 → 返回用户    │
  └───────────────────────────────────────────┘
```

## 技术栈

| 层级 | 技术 |
|------|------|
| **Web 框架** | FastAPI + Pydantic |
| **ML 模型** | scikit-learn / XGBoost（多模型 Stacking，R²=0.91~0.94） |
| **LLM** | DeepSeek API（Tool Calling + 分析生成） |
| **RAG** | TF-IDF + jieba 分词 + LLM 查询重写 + 引用溯源 |
| **Agent** | 多工具自主调度（predict + search + query），并行调用 |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API Key
setx DEEPSEEK_API_KEY "your-key"

# 3. 构建知识库（首次运行）
python rag_01_build.py

# 4. 启动 Agent 服务
python -m uvicorn agent_server:app --reload --port 8000

# 5. 打开接口文档
# http://localhost:8000/docs
```

## 接口说明

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务状态 |
| `/health` | GET | 健康检查（含知识库状态） |
| `/chat` | POST | **Agent 对话**：LLM 自主选择工具 |
| `/analyze` | POST | **快捷分析**：一步完成预测+标准判断 |

### /chat 请求示例

```json
{
  "message": "Fe=0.08, Si=0.02, Cu=0.01, 温度720°C，预测成分并判断合不合格"
}
```

### /chat 响应示例

```json
{
  "回答": "## 成分预测结果\n| Al纯度 | Fe | Si | Cu | 判定 |\n| 99.9981% | 0.00012% | 0.0002% | 0.00011% | 全部合格 ✅ |",
  "工具调用记录": [
    {"tool": "predict_purity", "args": {"fe": 0.08, "si": 0.02, "cu": 0.01, "temperature": 720}},
    {"tool": "query_standard", "args": {"keyword": "高纯铝成分标准"}}
  ],
  "对话轮数": 2,
  "时间戳": "2026-07-03T21:48:05"
}
```

### 更多对话示例

```
用户："三层液电解法和偏析法有什么区别？"
Agent → 自动调 search_knowledge → RAG 检索知识库 → 返回对比表格（原理/能耗/纯度/杂质）

用户："Fe=0.12, 温度740°C，预测成分，并告诉我Fe超标怎么办"
Agent → 同时调 predict_purity + query_standard + search_knowledge
     → 返回预测结果 + 超标判断 + 异常处理规程
```

## 项目结构

```
industrial-prediction-api/
├── agent_server.py       # Agent 服务（Tool Calling + RAG + FastAPI）
├── tool_01_simple.py     # Tool Calling 入门：计算器工具
├── tool_02_predict.py    # Tool Calling 进阶：真实 predict 工具
├── tool_03_multitool.py  # Tool Calling 高级：多工具 Agent 循环
├── rag_01_build.py       # RAG 知识库构建（jieba + TF-IDF）
├── rag_02_query.py       # RAG 查询（LLM 查询重写 + 引用回答）
├── knowledge_base.pkl    # 序列化知识库（13 chunks × 311 维）
├── requirements.txt      # Python 依赖
├── README.md            # 项目说明
└── .gitignore           # 排除模型、数据、密钥
```

## 设计要点

- **Agent 自主决策**：LLM 根据用户意图自动选择工具，支持并行调用多个工具
- **Tool Calling**：高纯铝预测模型封装为标准 Tool，LLM 自动填参、调用、解读
- **RAG 检索增强**：工艺知识库 + LLM 查询重写，解决中文同义词问题（"采样"≈"取样"）
- **引用溯源**：RAG 回答带来源标注 [参考1]、[参考2]，避免 LLM 编造
- **错误降级**：LLM/工具调用失败不中断服务，返回降级结果
- **安全**：API Key 环境变量注入，模型和数据不入库
- **自动文档**：FastAPI Swagger 交互式文档
