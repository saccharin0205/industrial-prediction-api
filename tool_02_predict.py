"""
Tool Calling 第二步：把高纯铝预测模型封装成 LLM 工具

这次不用计算器了，用你真实的 predict() 函数。
LLM 能：
  1. 理解用户说的原料配比
  2. 自己决定调 predict_purity 工具
  3. 拿到预测结果后判断合不合格、给出工艺建议

运行方式：
  python tool_02_predict.py
"""
import os
import sys
import json

# 修复 Windows 中文编码
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 引入你的真实预测函数
sys.path.insert(0, r"E:\小论文\Dataset construction")
from predict import predict
from openai import OpenAI

# ---------- 1. 客户端 ----------
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ---------- 2. 定义 predict 工具 ----------
# 这就是你的模型"说明书"，LLM 看了就知道怎么用
predict_tool = {
    "type": "function",
    "function": {
        "name": "predict_purity",
        "description": (
            "预测高纯铝样品的成分含量。"
            "输入原料中 Fe/Si/Cu 的配比和工艺温度，"
            "返回 Al 纯度(%) 以及 Fe、Si、Cu 的预测含量(%)。"
            "当用户问'这批料合不合格'、'帮我预测成分'、'这个配比怎么样'时，使用此工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fe": {
                    "type": "number",
                    "description": "Fe 原料配比 (%)，范围 0-1"
                },
                "si": {
                    "type": "number",
                    "description": "Si 原料配比 (%)，范围 0-1"
                },
                "cu": {
                    "type": "number",
                    "description": "Cu 原料配比 (%)，范围 0-1"
                },
                "temperature": {
                    "type": "number",
                    "description": "工艺温度 (°C)，范围 0-2000"
                },
                "cell": {
                    "type": "integer",
                    "description": "电解槽号（可选），范围 1-100"
                },
                "previous_al": {
                    "type": "number",
                    "description": "前一天 Al 纯度百分比（可选），如 99.85"
                }
            },
            "required": ["fe", "si", "cu", "temperature"]
        }
    }
}


# ---------- 3. 工具执行函数 ----------
def run_predict(args: dict) -> str:
    """执行真实预测，返回 JSON 字符串给 LLM"""
    try:
        # 把 LLM 给的参数名映射到 predict() 需要的格式
        params = {
            "Fe": args["fe"],
            "Si": args["si"],
            "Cu": args["cu"],
            "temperature": args["temperature"]
        }
        if "cell" in args and args["cell"] is not None:
            params["cell"] = args["cell"]
        if "previous_al" in args and args["previous_al"] is not None:
            params["previous_Al"] = args["previous_al"]

        result = predict(params)

        # 返回结构化结果，LLM 能理解
        return json.dumps({
            "success": True,
            "Al纯度(%)": round(result["Al"], 4),
            "Fe预测含量(%)": round(result["Fe"], 5),
            "Si预测含量(%)": round(result["Si"], 5),
            "Cu预测含量(%)": round(result["Cu"], 5),
            "标准": {
                "Fe上限": "0.005%",
                "Si上限": "0.005%",
                "Cu上限": "0.005%"
            }
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# ---------- 4. 完整的 Tool Calling 对话 ----------
def ask(user_question: str):
    """用户问一句话，LLM 自己决定调不调 predict"""
    print(f"\n{'='*60}")
    print(f"用户：{user_question}")
    print(f"{'='*60}")

    messages = [
        {"role": "system", "content": (
            "你是高纯铝电解工艺分析专家。"
            "当用户问关于铝样品成分预测、质量判断、配比分析的问题时，"
            "你必须使用 predict_purity 工具获取真实预测数据，"
            "严禁自己编造数据。"
            "拿到数据后，对比标准判断是否超标，给出工艺建议。"
            "高纯铝标准：Fe≤0.005%, Si≤0.005%, Cu≤0.005%。"
        )},
        {"role": "user", "content": user_question}
    ]

    # 第一轮：发给 LLM，看它要不要调工具
    response = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0.0,
        messages=messages,
        tools=[predict_tool]
    )

    msg = response.choices[0].message

    # 如果 LLM 要调工具
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        print(f"\n🔧 LLM 决定调用工具：{tool_name}")
        print(f"📥 参数：{json.dumps(tool_args, ensure_ascii=False)}")

        # 执行
        tool_result = run_predict(tool_args)
        print(f"📤 预测结果：{tool_result}")

        # 把工具结果发回 LLM
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result
        })

        # 第二轮：LLM 基于结果生成分析
        response2 = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.3,
            messages=messages
        )
        final_answer = response2.choices[0].message.content
        print(f"\n📋 LLM 分析报告：\n{final_answer}")

    else:
        # 不需要工具，直接回答
        print(f"\n💬 LLM 直接回答：{msg.content}")


# ---------- 5. 测试 ----------
if __name__ == "__main__":
    print("=" * 60)
    print("高纯铝预测 Agent — Tool Calling 演示")
    print("=" * 60)

    # 测试1：明确的预测请求
    ask("帮我预测一下：Fe=0.08%, Si=0.02%, Cu=0.01%, 温度 720°C，这批料合不合格？")

    # 测试2：自然语言描述
    ask("如果铁配比加到 0.15%，硅 0.03%，铜 0.02%，温度还是 720，铝纯度会降到多少？")

    # 测试3：多条件对比
    ask("槽号 5 的温度 750°C、Fe 0.06、Si 0.01、Cu 0.01，帮我看看预测结果")

    # 测试4：不需要工具的问题
    ask("高纯铝的标准是什么？")
