# 工业数据预测与分析 API

基于 ML 模型与大模型（LLM）分析的工业数据预测服务。

## 项目简介

将工艺参数预测模型封装为 Web API，支持单一接口同时返回 **数值预测** 和 **AI 文字分析**，模拟工业场景中"质量预测 + 工艺建议"的完整链路。

## 架构

```
用户 → POST /predict        → ML 模型 → 预测成分值
     → POST /full-analysis  → ML 模型 → 预测成分值
                            → LLM 分析 → 超标判断 + 工艺建议
                            → 合并返回完整报告
```

## 技术栈

- **Web 框架**：FastAPI + Pydantic
- **机器学习**：scikit-learn / XGBoost（多模型 Stacking）
- **大模型**：DeepSeek API（LLM 分析层）
- **运行环境**：Python 3.8+

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API Key
setx DEEPSEEK_API_KEY "your-key"

# 3. 启动服务
python -m uvicorn api_server:app --reload --port 8000

# 4. 打开接口文档
# http://localhost:8000/docs
```

## 接口说明

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务状态 |
| `/health` | GET | 健康检查 |
| `/predict` | POST | ML 模型预测成分 |
| `/full-analysis` | POST | ML 预测 + LLM 分析报告 |

### 请求示例

```json
{
  "fe": 0.08,
  "si": 0.02,
  "cu": 0.01,
  "temperature": 720
}
```

### 响应示例（/full-analysis）

```json
{
  "输入": {"Fe": 0.08, "Si": 0.02, "Cu": 0.01, "温度": 720},
  "ML预测": {
    "Al纯度": "99.9977%",
    "Fe": "0.00012%",
    "Si": "0.00020%",
    "Cu": "0.00011%"
  },
  "AI分析": {
    "Fe超标": false,
    "Si超标": false,
    "Cu超标": false,
    "风险等级": "正常",
    "分析": "ML预测杂质含量均低于标准限值，样品符合高纯铝要求。"
  }
}
```

## 项目结构

```
├── api_server.py        # FastAPI 服务主文件
├── requirements.txt     # Python 依赖
├── README.md           # 项目说明
└── .gitignore          # 排除模型、数据、密钥
```

## 设计要点

- **输入校验**：Pydantic 自动校验参数范围和类型
- **错误处理**：LLM 调用失败时优雅降级，不中断服务
- **安全**：API Key 通过环境变量注入，不写入代码
- **自动文档**：FastAPI 自动生成 Swagger 交互式文档
