"""
工业 Agent 分析系统 — 完整版

一个 FastAPI 接口，LLM 自主决定：
  - 调 predict_purity（预测成分）
  - 调 search_knowledge（RAG 检索工艺知识）
  - 调 query_standard（查工艺标准）

这是你简历的核心项目。

启动：
  python -m uvicorn agent_server:app --reload --port 8000
  http://localhost:8000/docs
"""
import os
import sys
import json
import pickle
import logging
from typing import Optional
from datetime import datetime

# ---------- 清除代理 + 编码 ----------
for key in list(os.environ.keys()):
    if key.lower().endswith('_proxy') or key.lower() == 'all_proxy':
        del os.environ[key]

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---------- 引入 ML 模型 ----------
# 尝试加载真实模型，失败则使用 mock（Docker 环境）
try:
    sys.path.insert(0, r"E:\小论文\Dataset construction")
    from predict import predict
    _PREDICT_MODE = "真实模型"
except (ImportError, FileNotFoundError):
    from predict_mock import predict
    _PREDICT_MODE = "Mock演示模式"
from openai import OpenAI
import httpx
import numpy as np
import jieba
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------- 日志 ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------- 客户端（禁用代理） ----------
http_client = httpx.Client(proxy=None, trust_env=False)
llm = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    http_client=http_client
)

# ---------- 应用 ----------
app = FastAPI(
    title="工业 Agent 分析系统",
    description="集 ML 预测 + RAG 知识检索 + 工艺标准查询于一体的智能分析服务",
    version="2.0.0"
)

# ============================================================
# 加载知识库
# ============================================================
_kb = None

def get_kb():
    global _kb
    if _kb is None:
        kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.pkl")
        if os.path.exists(kb_path):
            with open(kb_path, "rb") as f:
                _kb = pickle.load(f)
            logger.info(f"知识库已加载：{len(_kb['chunks'])} 个 chunk")
        else:
            logger.warning("知识库文件不存在，RAG 检索不可用")
            _kb = {"chunks": [], "sources": [], "embeddings": None, "vectorizer": None}
    return _kb


# ============================================================
# 工具 1：成分预测
# ============================================================
def tool_predict_purity(args: dict) -> str:
    """调用真实 ML 模型预测成分"""
    try:
        params = {
            "Fe": args["fe"], "Si": args["si"],
            "Cu": args["cu"], "temperature": args["temperature"]
        }
        if "cell" in args and args["cell"] is not None:
            params["cell"] = args["cell"]
        if "previous_al" in args and args["previous_al"] is not None:
            params["previous_Al"] = args["previous_al"]

        result = predict(params)
        return json.dumps({
            "success": True,
            "Al纯度(%)": round(result["Al"], 4),
            "Fe预测含量(%)": round(result["Fe"], 5),
            "Si预测含量(%)": round(result["Si"], 5),
            "Cu预测含量(%)": round(result["Cu"], 5),
            "标准": "Fe≤0.005%, Si≤0.005%, Cu≤0.005%"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# ============================================================
# 工具 2：RAG 知识检索
# ============================================================
def tool_search_knowledge(args: dict) -> str:
    """从知识库检索相关工艺知识"""
    query = args.get("query", "")
    top_k = args.get("top_k", 3)
    kb = get_kb()

    if kb["vectorizer"] is None:
        return json.dumps({"success": False, "error": "知识库未加载"}, ensure_ascii=False)

    try:
        # 查询重写
        expanded = llm.chat.completions.create(
            model="deepseek-chat", temperature=0.0,
            messages=[
                {"role": "system", "content": "将以下问题改写为一串搜索关键词（空格分隔，包含同义词）。只返回关键词。"},
                {"role": "user", "content": query}
            ]
        ).choices[0].message.content.strip()

        # TF-IDF 检索
        query_vec = kb["vectorizer"].transform([" ".join(jieba.cut(expanded))]).toarray()
        similarities = cosine_similarity(query_vec, kb["embeddings"])[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # 过滤无关结果
                results.append({
                    "title": kb["sources"][idx]["title"],
                    "source": kb["sources"][idx]["source"],
                    "content": kb["chunks"][idx],
                    "score": round(float(similarities[idx]), 4)
                })

        return json.dumps({
            "success": True,
            "query": query,
            "expanded_keywords": expanded,
            "results": results
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# ============================================================
# 工具 3：工艺标准查询
# ============================================================
PROCESS_STANDARDS = {
    "Fe上限": "高纯铝中 Fe ≤ 0.005%",
    "Si上限": "高纯铝中 Si ≤ 0.005%",
    "Cu上限": "高纯铝中 Cu ≤ 0.005%",
    "温度范围": "电解温度 700-760°C，推荐 720-740°C",
    "Al纯度等级": "4N(99.99%) > 4N5(99.995%) > 5N(99.999%) > 6N(99.9999%)",
    "采样规范": "每 2h 取一次样，≥100g，氩气保护冷却，每批次 3 个平行样",
    "槽维护": "每 30 天清理阳极泥，每 90 天更换内衬",
    "异常处理": "Fe/Si 超标 50% 以上：降温至 700°C + 增加搅拌。排查原料/电极/工具",
}


def tool_query_standard(args: dict) -> str:
    """查询工艺标准"""
    keyword = args.get("keyword", "")
    matches = {}
    for k, v in PROCESS_STANDARDS.items():
        if keyword in k or keyword in v:
            matches[k] = v
    if not matches:
        return json.dumps({"success": True, "found": False, "message": f"未找到'{keyword}'相关标准"}, ensure_ascii=False)
    return json.dumps({"success": True, "found": True, "keyword": keyword, "results": matches}, ensure_ascii=False)


# ============================================================
# 工具清单
# ============================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_purity",
            "description": "预测高纯铝样品成分（Al纯度、Fe、Si、Cu含量）。当用户提供原料配比、温度参数并询问成分预测、质量判断、是否超标时使用。严禁编造数据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "fe": {"type": "number", "description": "Fe含量(%)，0-1"},
                    "si": {"type": "number", "description": "Si含量(%)，0-1"},
                    "cu": {"type": "number", "description": "Cu含量(%)，0-1"},
                    "temperature": {"type": "number", "description": "工艺温度(°C)，0-2000"},
                    "cell": {"type": "integer", "description": "电解槽号（可选）"},
                    "previous_al": {"type": "number", "description": "前一天Al纯度（可选）"}
                },
                "required": ["fe", "si", "cu", "temperature"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "从工艺知识库检索信息。当用户问工艺知识、标准规范、操作方法、技术原理等非预测类问题时使用。如'三层液电解法是什么'、'偏析法原理'、'硼化处理'等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询语句"},
                    "top_k": {"type": "integer", "description": "返回结果数量，默认3"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_standard",
            "description": "快速查询工艺标准参数。当用户问具体标准数值时使用，如'Fe标准是多少'、'温度范围'、'采样规范'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "查询关键词，如'Fe上限'、'温度'、'采样'、'异常'"}
                },
                "required": ["keyword"]
            }
        }
    },
]

TOOL_EXECUTORS = {
    "predict_purity": tool_predict_purity,
    "search_knowledge": tool_search_knowledge,
    "query_standard": tool_query_standard,
}


# ============================================================
# Agent 主循环
# ============================================================
def agent_loop(user_message: str, max_turns: int = 5) -> dict:
    """
    Agent 循环：
    1. 发送用户消息 + 工具列表给 LLM
    2. LLM 决定：直接回答 or 调工具
    3. 调工具 → 结果发回 LLM → 重复
    4. 直到 LLM 给出最终回答
    """
    messages = [
        {"role": "system", "content": (
            "你是高纯铝工艺智能分析专家，负责成分预测、质量判断和工艺知识解答。\n"
            "规则：\n"
            "1. 成分预测 → 必须用 predict_purity，严禁编造数字\n"
            "2. 工艺知识/原理/方法 → 必须用 search_knowledge 检索知识库\n"
            "3. 标准数值查询 → 用 query_standard\n"
            "4. 拿到工具结果后，用表格、分点清晰呈现\n"
            "5. 若信息足够，直接给出最终回答"
        )},
        {"role": "user", "content": user_message}
    ]

    tools_called = []
    turn = 0

    while turn < max_turns:
        turn += 1

        response = llm.chat.completions.create(
            model="deepseek-chat", temperature=0.0,
            messages=messages, tools=TOOLS
        )

        msg = response.choices[0].message

        # 直接回答 → 结束
        if not msg.tool_calls:
            return {
                "answer": msg.content,
                "tools_called": tools_called,
                "turns": turn
            }

        # 执行工具
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            logger.info(f"Agent 调用工具：{name}({json.dumps(args, ensure_ascii=False)})")

            executor = TOOL_EXECUTORS.get(name)
            result = executor(args) if executor else json.dumps({"error": f"未知工具: {name}"})

            tools_called.append({"tool": name, "args": args})

            messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return {"answer": "分析超时，请简化问题后重试", "tools_called": tools_called, "turns": turn}


# ============================================================
# Pydantic 模型
# ============================================================
class ChatRequest(BaseModel):
    message: str = Field(..., description="用户问题，如'Fe=0.08, Si=0.02, 温度720, 预测一下'")
    max_turns: int = Field(5, ge=1, le=10, description="最大工具调用轮数")


class ChatResponse(BaseModel):
    回答: str
    工具调用记录: list
    对话轮数: int
    时间戳: str


# ============================================================
# 接口
# ============================================================
@app.get("/")
def home():
    return {
        "service": "工业 Agent 分析系统",
        "version": "2.0.0",
        "features": ["ML 成分预测", "RAG 知识检索", "工艺标准查询", "LLM 智能分析"],
        "docs": "/docs"
    }


@app.get("/health")
def health():
    kb = get_kb()
    return {
        "status": "ok",
        "knowledge_base": f"{len(kb['chunks'])} chunks" if kb["vectorizer"] else "未加载",
        "deepseek_api": "已配置" if os.getenv("DEEPSEEK_API_KEY") else "未配置"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Agent 对话接口 — 核心功能

    LLM 自主选择调用以下工具：
    - **predict_purity**：ML 模型预测成分
    - **search_knowledge**：RAG 检索工艺知识库
    - **query_standard**：查询工艺标准参数

    示例问题：
    - "Fe=0.08, Si=0.02, Cu=0.01, 温度720，帮我预测一下"
    - "三层液电解法和偏析法有什么区别？"
    - "高纯铝的采样规范是什么？"
    """
    try:
        result = agent_loop(request.message, request.max_turns)
    except Exception as e:
        logger.error(f"Agent 执行失败: {e}")
        raise HTTPException(status_code=500, detail=f"Agent 执行失败: {str(e)}")

    return ChatResponse(
        回答=result["answer"],
        工具调用记录=result["tools_called"],
        对话轮数=result["turns"],
        时间戳=datetime.now().isoformat()
    )


@app.post("/analyze")
def analyze(fe: float, si: float, cu: float, temperature: float, cell: Optional[int] = None):
    """
    快捷分析接口 — 一步完成预测 + 标准判断

    自动调 predict_purity + query_standard，返回综合报告。
    适合不想手动写问题的场景。
    """
    message = (
        f"原料配比：Fe={fe}%, Si={si}%, Cu={cu}%, 温度={temperature}°C"
        + (f", 槽号={cell}" if cell else "")
        + "。请预测成分含量，并判断是否超标、给出工艺建议。"
    )
    return chat(ChatRequest(message=message))


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
