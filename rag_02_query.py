"""
RAG 第二步：TF-IDF 检索 + LLM 查询重写 + 引用回答

核心技巧：LLM Query Expansion（查询扩展）
  用户问："采样时需要注意什么？"
  → LLM 扩展成："高纯铝采样规范 取样频率 取样注意事项 样品重量 氩气保护"
  → 扩展后的关键词做 TF-IDF 检索 → 命中率大幅提升

这解决了 TF-IDF 的"同义词不匹配"问题（"采样" vs "取样"），
同时不需要下载任何模型，只靠 LLM。

运行方式：
  python rag_02_query.py
"""
import os
import sys
import pickle

# 清除代理
for key in list(os.environ.keys()):
    if key.lower().endswith('_proxy') or key.lower() == 'all_proxy':
        del os.environ[key]

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import jieba
import numpy as np
import httpx
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# 创建 OpenAI 客户端时强制禁用代理
http_client = httpx.Client(proxy=None, trust_env=False)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    http_client=http_client
)


# ============================================================
# 加载
# ============================================================
def load_knowledge_base(path: str = "knowledge_base.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# 查询重写 — RAG 进阶技巧
# ============================================================
def expand_query(user_query: str) -> str:
    """
    让 LLM 把口语化问题扩展成"关键词密集"的搜索字符串。

    例子：
      "采样时需要注意什么？"
      → "高纯铝 采样 取样 规范 频率 样品重量 氩气保护 注意事项"

    TF-IDF 用这些关键词去匹配，命中率比原问题高得多。
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0.0,
        messages=[
            {"role": "system", "content": (
                "你是一个搜索查询优化器。将用户问题改写为一串关键词，"
                "用空格分隔。包含同义词和相关的专业术语。"
                "只返回关键词，不要任何解释。"
            )},
            {"role": "user", "content": f"为以下问题生成搜索关键词：{user_query}"}
        ]
    )
    expanded = response.choices[0].message.content.strip()
    return expanded


# ============================================================
# TF-IDF 检索
# ============================================================
def retrieve(query: str, kb: dict, top_k: int = 3) -> list:
    """
    检索流程：
    1. LLM 重写查询 → 关键词扩展
    2. jieba 分词
    3. TF-IDF 向量化 → 余弦相似度 → top-k
    """
    # 1. 查询扩展
    expanded = expand_query(query)
    print(f"   🔄 查询扩展：{expanded}")

    # 2. jieba 分词 + TF-IDF
    vectorizer = kb["vectorizer"]
    kb_embeddings = kb["embeddings"]

    query_vec = vectorizer.transform([" ".join(jieba.cut(expanded))]).toarray()
    similarities = cosine_similarity(query_vec, kb_embeddings)[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "chunk": kb["chunks"][idx],
            "title": kb["sources"][idx]["title"],
            "source": kb["sources"][idx]["source"],
            "score": float(similarities[idx])
        })

    return results


# ============================================================
# RAG 查询
# ============================================================
def ask(query: str, kb: dict = None, top_k: int = 3):
    """完整的 RAG：查询重写 → TF-IDF 检索 → LLM 生成引用回答"""
    if kb is None:
        kb = load_knowledge_base()

    print(f"\n{'='*60}")
    print(f"🔍 用户问题：{query}")
    print(f"{'='*60}")

    # 1. 检索（含查询重写）
    results = retrieve(query, kb, top_k=top_k)

    print(f"\n📚 检索到 {len(results)} 个相关文段：")
    for i, r in enumerate(results):
        preview = r['chunk'][:100].replace('\n', ' ')
        print(f"  [{i+1}] {r['title']}（{r['source']}）相似度: {r['score']:.4f}")
        print(f"      {preview}...")

    # 2. 拼参考资料
    references = ""
    for i, r in enumerate(results):
        references += f"\n[参考{i+1}] 来源：{r['title']}（{r['source']}）\n内容：{r['chunk']}\n"

    # 3. LLM 生成带引用回答
    response = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0.2,
        messages=[
            {"role": "system", "content": (
                "你是高纯铝工艺专家。请根据提供的参考资料回答用户问题。\n"
                "规则：\n"
                "1. 只能基于参考资料内容回答，不要使用你自己的知识\n"
                "2. 回答中引用参考资料的编号，如 [参考1]、[参考2]\n"
                "3. 如果参考资料不足以回答问题，请明确说明\n"
                "4. 使用表格、分点等方式让回答清晰易读"
            )},
            {"role": "user", "content": (
                f"参考资料：\n{references}\n\n"
                f"用户问题：{query}\n\n"
                f"请基于以上参考资料回答，并注明引用来源。"
            )}
        ]
    )

    answer = response.choices[0].message.content
    print(f"\n📋 RAG 回答（带引用）：\n{answer}")

    # 4. 来源
    print(f"\n{'─'*60}")
    print("📎 引用来源：")
    seen = set()
    for r in results:
        if r['title'] not in seen:
            print(f"  • {r['title']} — {r['source']}")
            seen.add(r['title'])

    return answer


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("加载知识库...")
    kb = load_knowledge_base()
    print(f"知识库：{len(kb['chunks'])} 个 chunk，TF-IDF 向量维度 {kb['embeddings'].shape[1]}")

    ask("高纯铝的铁含量标准是多少？超标了怎么办？", kb)
    ask("三层液电解法和偏析法有什么区别？哪种更好？", kb)
    ask("采样时需要注意什么？多久取一次样？", kb)
    ask("硼化处理的作用是什么？去除什么元素？", kb)
