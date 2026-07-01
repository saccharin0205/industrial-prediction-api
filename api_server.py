
"""
高纯铝成分预测与分析 API 服务
工业数据处理与模型服务化项目
"""
import os
import sys
import json
import logging
from typing import Optional

sys.path.insert(0, r"E:\小论文\Dataset construction")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from predict import predict
from openai import OpenAI

# ---------- 日志 ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- 应用 ----------
app = FastAPI(
    title="高纯铝成分预测与分析 API",
    description="基于 ML 模型 + LLM 分析的工业数据预测服务",
    version="1.0.0"
)

# ---------- 客户端 ----------
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ---------- 模型 ----------
class SampleInput(BaseModel):
    fe: float = Field(..., ge=0, le=1, description="Fe 含量 (%)")
    si: float = Field(..., ge=0, le=1, description="Si 含量 (%)")
    cu: float = Field(..., ge=0, le=1, description="Cu 含量 (%)")
    temperature: float = Field(..., ge=0, le=2000, description="工艺温度 (°C)")
    cell: Optional[int] = Field(None, ge=1, le=100, description="槽号")
    previous_al: Optional[float] = Field(None, ge=90, le=100, description="前一天 Al 纯度 (%)")

class PredictResponse(BaseModel):
    Al纯度: str
    Fe含量: str
    Si含量: str
    Cu含量: str

class FullAnalysisResponse(BaseModel):
    输入: dict
    ML预测: dict
    AI分析: dict


# ---------- 通用方法 ----------
def _build_params(sample: SampleInput) -> dict:
    params = {
        "Fe": sample.fe, "Si": sample.si,
        "Cu": sample.cu, "temperature": sample.temperature
    }
    if sample.cell is not None:
        params["cell"] = sample.cell
    if sample.previous_al is not None:
        params["previous_Al"] = sample.previous_al
    return params


# ---------- 接口 ----------
@app.get("/")
def home():
    return {"service": "高纯铝成分预测与分析 API", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_api(sample: SampleInput):
    """仅 ML 模型预测"""
    try:
        result = predict(_build_params(sample))
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型预测失败: {str(e)}")

    return {
        "Al纯度": f"{result['Al']:.4f}%",
        "Fe含量": f"{result['Fe']:.5f}%",
        "Si含量": f"{result['Si']:.5f}%",
        "Cu含量": f"{result['Cu']:.5f}%",
    }


@app.post("/full-analysis", response_model=FullAnalysisResponse)
def full_analysis(sample: SampleInput):
    """ML 预测 + LLM 分析"""
    # 1. ML 预测
    try:
        ml_result = predict(_build_params(sample))
    except Exception as e:
        logger.error(f"ML 预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型预测失败: {str(e)}")

    # 2. LLM 分析
    prompt = (
        f"分析以下高纯铝样品：\n"
        f"输入：Fe {sample.fe}%, Si {sample.si}%, Cu {sample.cu}%, 温度 {sample.temperature}°C\n"
        f"ML预测：Al {ml_result['Al']:.4f}%, Fe {ml_result['Fe']:.5f}%, "
        f"Si {ml_result['Si']:.5f}%, Cu {ml_result['Cu']:.5f}%\n"
        f"标准：Fe≤0.005%, Si≤0.005%, Cu≤0.005% 为合格\n"
        f'返回JSON：{{"Fe超标": true/false, "Si超标": true/false, "Cu超标": true/false, '
        f'"风险等级": "正常/关注/异常", "分析": "一句话"}}'
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.1,
            messages=[
                {"role": "system", "content": "你是高纯铝冶金分析专家。只返回JSON，不要其他文字。"},
                {"role": "user", "content": prompt}
            ]
        )
        llm = json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"LLM 分析失败: {e}")
        llm = {"Fe超标": None, "Si超标": None, "Cu超标": None, "风险等级": "分析不可用", "分析": str(e)}

    return {
        "输入": {"Fe": sample.fe, "Si": sample.si, "Cu": sample.cu, "温度": sample.temperature},
        "ML预测": {
            "Al纯度": f"{ml_result['Al']:.4f}%",
            "Fe": f"{ml_result['Fe']:.5f}%",
            "Si": f"{ml_result['Si']:.5f}%",
            "Cu": f"{ml_result['Cu']:.5f}%",
        },
        "AI分析": llm
    }