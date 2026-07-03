"""
RAG 第一步：构建知识库（TF-IDF + jieba 中文分词）

流程：
  1. 文档 → 切分（chunking）
  2. jieba 分词 → TF-IDF 向量化
  3. 向量 + 原文保存到本地

检索时配合 LLM 查询重写（rag_02_query.py），解决 TF-IDF 同义词问题。

运行方式：
  python rag_01_build.py
"""
import os
import sys
import pickle

# 清除可能残留的代理设置
for key in list(os.environ.keys()):
    if key.lower().endswith('_proxy') or key.lower() == 'all_proxy':
        del os.environ[key]

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# 知识库文档
# ============================================================
DOCUMENTS = [
    {
        "title": "高纯铝质量标准",
        "content": (
            "高纯铝（High Purity Aluminum）指纯度达到 4N（99.99%）及以上的金属铝。"
            "主要杂质元素控制标准：Fe ≤ 0.005%、Si ≤ 0.005%、Cu ≤ 0.005%。"
            "纯度等级划分：4N(99.99%)、4N5(99.995%)、5N(99.999%)、6N(99.9999%)。"
            "其中 4N 级为工业高纯铝入门标准，5N 以上主要用于半导体、超导、光学等领域。"
            "检测方法：GD-MS（辉光放电质谱）和 ICP-MS 是高纯铝成分分析的主流手段。"
        ),
        "source": "工艺标准手册"
    },
    {
        "title": "电解工艺参数规范",
        "content": (
            "高纯铝电解精炼的关键工艺参数："
            "电解温度一般控制在 700-760°C，推荐温度 720-740°C。"
            "温度过高（>760°C）会增加杂质溶解度，降低铝纯度；"
            "温度过低（<700°C）会导致铝液粘度增加，不利于杂质沉降分离。"
            "铝液温度是电解过程中的关键工艺参数，影响杂质溶解度和分离效率。"
            "电解质成分需定期检测和调整，保持 NaF/AlF3 分子比在 2.3-2.7 之间。"
        ),
        "source": "工艺标准手册"
    },
    {
        "title": "采样与检测规范",
        "content": (
            "高纯铝生产过程采样规范："
            "每 2 小时从出铝口取一次样，样品重量 ≥ 100g。"
            "取样后需在氩气保护下冷却，防止表面氧化影响成分分析准确性。"
            "每批次至少取 3 个平行样，分析结果取平均值。"
            "槽维护周期：电解槽每 30 天清理一次阳极泥，每 90 天更换内衬。"
        ),
        "source": "采样操作规程"
    },
    {
        "title": "三层液电解精炼法",
        "content": (
            "三层液电解精炼法（Three-layer Electrolysis）是制备高纯铝的经典方法。"
            "原理：利用密度差异将体系分为三层——下层为阳极铝-铜合金熔体，"
            "中层为氟化物电解质，上层为阴极精铝液。"
            "电解时 Al3+ 在阴极优先放电析出，电位比铝正的杂质（Cu、Fe、Si、Zn、Ti）"
            "残留阳极合金中，电位比铝负的杂质（Na、Ca、Mg）滞留电解质中。"
            "可将 99.8% 原铝提纯至 99.996%（4N6）。"
            "缺点：吨铝电耗 > 15,000 kWh，每吨产品产生约 30-40 kg 氟化物废弃物，"
            "能耗高、污染大。对 Cu、Mg、Ba、F 等杂质去除困难。"
        ),
        "source": "文献综述"
    },
    {
        "title": "偏析法提纯技术",
        "content": (
            "偏析法（Segregation/Fractional Crystallization）利用杂质元素在铝熔体"
            "凝固过程中固液两相间平衡分配系数（K0 = C固/C液）的差异进行分离。"
            "K0 < 1 的杂质（Si、Fe、Cu、Zn）富集于液相随残液排出；"
            "K0 > 1 的杂质（Ti、V、Cr）和 K0≈1 的杂质（Zn、Mn）难以去除。"
            "优势：吨铝电耗 < 3,500 kWh，几乎无废弃物产生，环保节能。"
            "平均比三层液法每吨省电 6,000 度。"
            "上海交大发明了添加 Si 元素调控 V 元素分配系数的方法，"
            "解决了偏析法无法去除 V 的难题，制备出 5N5-6N 超高纯铝锭，"
            "全元素综合提纯效率 > 75%。"
        ),
        "source": "上海交大精铝技术"
    },
    {
        "title": "熔盐电解精炼新工艺",
        "content": (
            "熔盐电解精炼法以废旧铝合金为可溶性阳极，在低温氯化物熔盐中电解提纯。"
            "电解质体系：56 mol% LiCl - 36 mol% KCl - 8 mol% NaCl + 10 mol% AlCl3。"
            "最优电解参数：温度 550°C，AlCl3 浓度 10 mol%，电流密度 ≤ 0.15 A/cm²。"
            "电流效率可达 92.5%，可获得 99.8%-99.9% 的工业级纯铝。"
            "该工艺绿色环保、成本较低、可处理废杂铝（Upcycling）。"
            "电化学行为：Al 在 -1.6V 优先溶解，Si 和 Fe 在 > -0.9V 才开始溶解，"
            "证明了电化学选择性提纯的可行性。"
        ),
        "source": "Journal of Cleaner Production"
    },
    {
        "title": "异常工况处理规程",
        "content": (
            "高纯铝电解过程异常工况处理："
            "当 Fe 或 Si 任一超标 50% 以上时，需立即降温至 700°C 并增加搅拌。"
            "Fe 超标排查：检查原料（氧化铝、氟化盐）Fe 含量是否偏高，"
            "检查阳极钢爪、阴极钢棒是否有腐蚀脱落，检查工具是否带入铁杂质。"
            "Si 超标排查：检查原料中 SiO2 含量，检查炉衬是否有侵蚀脱落。"
            "Cu 超标排查：检查阳极导电铜排是否腐蚀，检查加料工具是否含铜。"
            "轻微超标时可降低电流密度、延长电解时间；"
            "严重超标时建议将该槽铝液单独存放，回炉重炼或降级使用。"
        ),
        "source": "异常处理规程"
    },
    {
        "title": "硼化处理与导电性能优化",
        "content": (
            "电解原铝液熔体处理技术："
            "氟化盐精炼可使电阻率降至 27.46 nΩ·m，抗拉强度提升至 58.29 MPa，"
            "伸长率提升至 45.24%。"
            "硼化处理除 V：B 添加量为 0.08%（质量分数）时，V 元素去除率达 94.2%。"
            "精炼剂+硼化剂复合处理：电阻率平均下降 7.54%，"
            "抗拉强度平均提升 35.60%，伸长率平均提升 64.84%。"
        ),
        "source": "中国冶金 2024"
    },
]


# ============================================================
# 文本切分
# ============================================================
def split_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list:
    """滑动窗口切分"""
    chunks = []
    if len(text) <= chunk_size:
        return [text]
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


# ============================================================
# 构建知识库
# ============================================================
def build_knowledge_base():
    """主流程：切分 → jieba分词 → TF-IDF向量化 → 保存"""
    print("=" * 60)
    print("构建高纯铝工艺知识库（TF-IDF + jieba）")
    print("=" * 60)

    # 1. 切分
    all_chunks = []
    all_sources = []

    for doc in DOCUMENTS:
        chunks = split_text(doc["content"], chunk_size=200, overlap=50)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_sources.append({
                "title": doc["title"],
                "source": doc["source"],
            })

    print(f"切分完成：{len(DOCUMENTS)} 篇文档 → {len(all_chunks)} 个 chunk")

    # 2. jieba 分词 + TF-IDF 向量化
    print("jieba 分词中...")
    processed_chunks = [" ".join(jieba.cut(c)) for c in all_chunks]
    vectorizer = TfidfVectorizer(max_features=512)
    embeddings = vectorizer.fit_transform(processed_chunks).toarray()

    print(f"向量化完成：{embeddings.shape[0]} 个向量 × {embeddings.shape[1]} 维")
    print(f"词汇表大小：{len(vectorizer.vocabulary_)}")

    # 3. 保存
    output = {
        "chunks": all_chunks,
        "sources": all_sources,
        "embeddings": embeddings,
        "vectorizer": vectorizer,
    }
    with open("knowledge_base.pkl", "wb") as f:
        pickle.dump(output, f)

    print(f"\n✅ 知识库构建完成！")
    print(f"   Chunk 数：{len(all_chunks)}")
    print(f"   向量维度：{embeddings.shape[1]}")
    print(f"   保存位置：knowledge_base.pkl")
    print(f"   文件大小：{os.path.getsize('knowledge_base.pkl') / 1024:.1f} KB")


if __name__ == "__main__":
    build_knowledge_base()
