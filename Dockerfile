# ============================================================
# 工业 Agent 分析系统 — Docker 镜像
#
# 构建：docker build -t industrial-agent .
# 运行：docker run -p 8000:8000 -e DEEPSEEK_API_KEY=你的key industrial-agent
# 打开：http://localhost:8000/docs
# ============================================================

# 1. 基础镜像（Python 3.8，和你的开发环境一致）
FROM python:3.8-slim

# 2. 设置工作目录（容器里的 /app 文件夹）
WORKDIR /app

# 3. 先复制依赖文件，利用 Docker 缓存——改代码不需要重装依赖
COPY requirements.txt .

# 4. 安装依赖（用清华镜像，国内快）
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 复制所有代码
COPY . .

# 6. 暴露端口
EXPOSE 8000

# 7. 知识库预构建 + 启动服务
CMD python rag_01_build.py && python -m uvicorn agent_server:app --host 0.0.0.0 --port 8000
