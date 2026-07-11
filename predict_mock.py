"""
Mock 预测函数 — Docker 环境使用

当无法访问真实的 predict 模块时（如 Docker 容器内），
使用此 mock 返回模拟数据，演示 Agent 系统功能。

接口和真实 predict() 完全一致：
  输入：{"Fe": 0.08, "Si": 0.02, "Cu": 0.01, "temperature": 720}
  输出：{"Al": 99.998, "Fe": 0.00012, "Si": 0.0002, "Cu": 0.0001}
"""
import random


def predict(params: dict) -> dict:
    """
    模拟高纯铝成分预测。
    返回的数据范围和真实模型一致，但数值是随机的。
    仅用于演示，生产环境请替换为真实模型。
    """
    # 基于输入参数生成合理的预测值
    # 输入杂质越高 → 预测杂质越高（模拟真实规律）
    fe_in = params.get("Fe", 0.08)
    si_in = params.get("Si", 0.02)
    cu_in = params.get("Cu", 0.01)
    temp = params.get("temperature", 720)

    # 温度越高，杂质溶解度越高（模拟趋势）
    temp_factor = 1.0 + (temp - 720) / 1000

    # 预测输出杂质（加一点随机噪声）
    fe_out = fe_in * 0.0015 * temp_factor * random.uniform(0.8, 1.2)
    si_out = si_in * 0.01 * temp_factor * random.uniform(0.8, 1.2)
    cu_out = cu_in * 0.01 * temp_factor * random.uniform(0.8, 1.2)

    # Al 纯度 = 100 - 杂质总量（简化）
    al_purity = 100 - (fe_out + si_out + cu_out) * 0.01

    return {
        "Al": round(al_purity, 4),
        "Fe": round(fe_out, 5),
        "Si": round(si_out, 5),
        "Cu": round(cu_out, 5),
    }
