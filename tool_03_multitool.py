"""
Tool Calling 第三步：多工具 Agent + 多轮对话

这次 LLM 手里有三个工具：
  1. predict_purity  — 预测成分
  2. query_standard  — 查工艺标准/规范
  3. calculate       — 数学计算

LLM 要根据用户问题，自己选工具、自己决定调用顺序。

关键变化：
  - 之前：一个工具，调一次就结束
  - 现在：多个工具，可能需要调多次（Agent Loop）

运行方式：
  python tool_03_multitool.py
"""
import os
import sys
import json

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, r"E:\小论文\Dataset construction")
from predict import predict
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ============================================================
# 工具 1：高纯铝成分预测
# ============================================================
def run_predict(args: dict) -> str:
    """执行真实预测"""
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
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# ============================================================
# 工具 2：工艺标准查询（模拟知识库）
# ============================================================
# 实际项目里这会是数据库或向量检索，现在用字典模拟
PROCESS_STANDARDS = {
    "Fe上限": "高纯铝中 Fe 含量不得超过 0.005%",
    "Si上限": "高纯铝中 Si 含量不得超过 0.005%",
    "Cu上限": "高纯铝中 Cu 含量不得超过 0.005%",
    "温度范围": "电解温度一般控制在 700-760°C，推荐 720-740°C",
    "Al纯度等级": "4N(99.99%) > 3N5(99.95%) > 3N(99.9%)，高纯铝要求≥4N",
    "采样规范": "每 2 小时从出铝口取一次样，样品重量≥100g，需在氩气保护下冷却",
    "槽维护": "电解槽每 30 天清理一次阳极泥，每 90 天更换内衬",
    "异常处理": "当 Fe/Si 任一超标 50% 以上时，需立即降温至 700°C 并增加搅拌",
    "铝液温度": "铝液温度是电解过程中的关键工艺参数，影响杂质溶解度和分离效率。温度过高会增加杂质溶解度，降低纯度；温度过低会导致铝液粘度增加，不利于杂质沉降分离。",
}


def run_query_standard(args: dict) -> str:
    """查询工艺标准"""
    keyword = args.get("keyword", "")
    # 模糊匹配
    matches = {}
    for k, v in PROCESS_STANDARDS.items():
        if keyword in k or keyword in v:
            matches[k] = v

    if not matches:
        return json.dumps({
            "success": True,
            "found": False,
            "message": f"未找到与'{keyword}'相关的工艺标准"
        }, ensure_ascii=False)

    return json.dumps({
        "success": True,
        "found": True,
        "keyword": keyword,
        "results": matches
    }, ensure_ascii=False)


# ============================================================
# 工具 3：计算器
# ============================================================
def run_calculate(args: dict) -> str:
    """安全计算"""
    try:
        expression = args.get("expression", "")
        # 安全：只允许数字和基本运算符
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return json.dumps({"success": False, "error": "表达式包含不允许的字符"})
        result = eval(expression)
        return json.dumps({"success": True, "expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# ============================================================
# 工具清单 — LLM 的"工具箱"
# ============================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_purity",
            "description": (
                "预测高纯铝样品成分含量（Al纯度、Fe、Si、Cu）。"
                "当用户问样品成分预测、某配比是否合格、质量判断时使用。"
                "调用后得到真实预测数据，不要自己编造数字。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fe": {"type": "number", "description": "Fe 含量(%)，范围 0-1"},
                    "si": {"type": "number", "description": "Si 含量(%)，范围 0-1"},
                    "cu": {"type": "number", "description": "Cu 含量(%)，范围 0-1"},
                    "temperature": {"type": "number", "description": "工艺温度(°C)，范围 0-2000"},
                    "cell": {"type": "integer", "description": "电解槽号（可选），1-100"},
                    "previous_al": {"type": "number", "description": "前一天 Al 纯度（可选）"}
                },
                "required": ["fe", "si", "cu", "temperature"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_standard",
            "description": (
                "查询高纯铝工艺标准、操作规范、质量要求。"
                "当用户问'标准是什么'、'温度多少合适'、'怎么采样'、"
                "'超标怎么办'、'Al纯度等级'等工艺知识问题时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "查询关键词，如'温度'、'Fe上限'、'采样'、'异常处理'"}
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "数学计算。用于换算、统计等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    },
]

# 工具名 → 执行函数映射
TOOL_EXECUTORS = {
    "predict_purity": run_predict,
    "query_standard": run_query_standard,
    "calculate": run_calculate,
}


# ============================================================
# Agent 主循环 — 支持多轮多工具调用
# ============================================================
def agent_chat(user_question: str, max_turns: int = 5):
    """
    Agent 对话循环：
    1. 发给 LLM → 看它要不要调工具
    2. 如果要 → 执行 → 发回结果 → 重复（可能再调另一个工具）
    3. 如果不要 → 结束，返回最终回答
    """
    print(f"\n{'='*60}")
    print(f"👤 用户：{user_question}")
    print(f"{'='*60}")

    messages = [
        {"role": "system", "content": (
            "你是高纯铝电解工艺分析专家，负责分析样品质量并提供工艺建议。"
            "规则：\n"
            "1. 当需要预测成分时，必须调用 predict_purity，严禁编造数据\n"
            "2. 当需要查工艺标准/规范时，必须调用 query_standard\n"
            "3. 当需要计算时，必须调用 calculate\n"
            "4. 拿到工具结果后，用表格、分点等方式清晰呈现\n"
            "5. 如果信息足够回答用户问题，直接回答，不要画蛇添足"
        )},
        {"role": "user", "content": user_question}
    ]

    turn = 0
    tools_called = []

    while turn < max_turns:
        turn += 1

        response = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.0,
            messages=messages,
            tools=TOOLS
        )

        msg = response.choices[0].message

        # LLM 决定直接回答 → 结束
        if not msg.tool_calls:
            print(f"\n💬 Agent 最终回答（共 {turn} 轮）：\n{msg.content}")
            return {
                "answer": msg.content,
                "tools_called": tools_called,
                "turns": turn
            }

        # LLM 要调工具 → 执行
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            print(f"\n🔧 第 {turn} 轮 · 调用工具：{name}")
            print(f"   📥 参数：{json.dumps(args, ensure_ascii=False)}")

            # 执行
            executor = TOOL_EXECUTORS.get(name)
            if executor:
                result = executor(args)
            else:
                result = json.dumps({"error": f"未知工具：{name}"})

            print(f"   📤 结果：{result[:150]}{'...' if len(result) > 150 else ''}")
            tools_called.append({"tool": name, "args": args})

            # 把工具调用和结果加入对话历史
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    # 达到最大轮数
    print(f"\n⚠️ 达到最大轮数 {max_turns}，强制结束")
    return {"answer": "分析超时", "tools_called": tools_called, "turns": turn}


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("多工具 Agent — 自动选择工具演示")
    print(f"可用工具：{list(TOOL_EXECUTORS.keys())}")
    print("=" * 60)

    # 测试1：需要 predict + 可能查标准
    agent_chat("Fe=0.08, Si=0.02, Cu=0.01, 温度720°C，帮我预测成分并判断合不合格")

    # 测试2：只查标准，不调 predict
    agent_chat("高纯铝的采样规范是什么？")

    # 测试3：需要 predict + 查异常处理 → 两个工具！
    agent_chat("Fe=0.12, Si=0.02, Cu=0.01, 温度740°C，预测一下，如果Fe超标了怎么办？")

    # 测试4：不需要任何工具
    agent_chat("什么是电解铝？一句话解释")
