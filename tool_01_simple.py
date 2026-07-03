"""
Tool Calling 第一步：看懂 LLM 怎么"决定调函数"

这个例子用一个简单的计算器工具，让你看清楚 Tool Calling 的完整流程：
  1. 定义工具（告诉 LLM 有哪些函数可用）
  2. 用户提问
  3. LLM 判断：直接回答 or 调工具
  4. 如果是调工具 → 你的代码执行函数 → 把结果发回 LLM
  5. LLM 基于结果生成最终回答

运行方式：
  python tool_01_simple.py
"""
import os
import sys
import json
from openai import OpenAI

# 修复 Windows 中文编码问题
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---------- 1. 创建客户端 ----------
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ---------- 2. 定义工具 ==========
# 这是 Tool Calling 的核心：用 JSON Schema 描述你的函数
# LLM 通过这个"说明书"知道：
#   - 这个工具叫什么名字
#   - 什么时候该用它
#   - 它需要什么参数、参数类型是什么
calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "执行数学计算。当用户问数学题、需要算数字时使用。支持加减乘除、幂运算等。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '3.14 * 2.5 + 100' 或 '2**10'"
                }
            },
            "required": ["expression"]
        }
    }
}

# ---------- 3. 你的实际函数 ----------
def calculate(expression: str) -> str:
    """真正执行计算的函数"""
    try:
        # ⚠️ eval 只用于演示！生产环境要用 ast.literal_eval 或 safe_eval
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算出错：{str(e)}"


# ---------- 4. 一次完整的 Tool Calling 对话 ----------
def tool_calling_demo(user_question: str):
    """演示一次完整的 Tool Calling 流程"""
    print(f"\n{'='*60}")
    print(f"用户：{user_question}")
    print(f"{'='*60}")

    # 第 1 轮：发送用户问题 + 工具定义
    response = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "你是一个有用的助手。如果用户问数学题，请使用 calculate 工具计算，不要自己心算。"},
            {"role": "user", "content": user_question}
        ],
        tools=[calculator_tool]  # ← 关键！把工具列表传给 LLM
    )

    # 看 LLM 返回了什么
    msg = response.choices[0].message

    print(f"\nLLM 的决定：")
    print(f"  直接回复文字？ {msg.content is not None}")
    print(f"  要调工具？       {msg.tool_calls is not None}")

    # 情况 A：LLM 决定调工具
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        print(f"\n  要调的工具：{tool_call.function.name}")
        print(f"  参数：{tool_call.function.arguments}")

        # 执行工具
        args = json.loads(tool_call.function.arguments)
        tool_result = calculate(args["expression"])
        print(f"  工具执行结果：{tool_result}")

        # 把工具结果发回 LLM，让它生成最终回答
        response2 = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": user_question},
                # 下面两行是 Tool Calling 的关键：
                # 把"LLM 说我要调什么工具"和"工具返回了什么"都发给 LLM
                {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                {"role": "tool", "tool_call_id": tool_call.id, "content": tool_result},
            ]
        )
        final_answer = response2.choices[0].message.content
        print(f"\n  LLM 最终回答：{final_answer}")

    # 情况 B：LLM 直接回答（不需要工具）
    else:
        print(f"\n  LLM 直接回答：{msg.content}")


# ---------- 5. 测试几个不同的问题 ----------
if __name__ == "__main__":
    # 这个会触发工具调用
    tool_calling_demo("帮我算一下 156 * 23 + 789 等于多少")

    # 这个也会触发工具调用
    tool_calling_demo("3.14 乘以 5 的平方是多少")

    # 这个不需要工具，LLM 直接回答
    tool_calling_demo("Python 是什么？")
