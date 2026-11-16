# 企业知识库问答系统 (Enterprise RAG Knowledge Base QA System)

《生成式大模型：原理和应用》模块二项目产出：**基于 FastAPI + ChromaDB + MarkItDown + 本地 Embedding 模型 + OpenAI 兼容大模型接口的企业知识库检索增强问答系统**。

系统完整实现“**多格式文档上传与解析 → 数据清洗与真实 Overlap 语义切分 → 本地 Embedding 向量化 → Chroma 持久化与文件 MD5 增量索引 → 向量/BM25 RRF 混合检索 → LLM Rerank 重排 → 距离阈值路由与溯源生成/拒答**”的工程闭环。

---

## 一、 项目目录结构

```text
.
├── .env.example                 # 环境变量配置模板（绝不包含真实密钥）
├── pyproject.toml               # uv 项目依赖与 Python 3.14 环境定义
├── run.sh                       # 一键启动脚本
├── Dockerfile                   # Docker 容器镜像构建文件
├── docker-compose.yml           # Docker Compose 容器编排配置
├── main.py                      # FastAPI 服务入口
├── models/                      # 本地嵌入模型存放目录（通过 EMBEDDING_MODEL 环境变量选择）
├── app/
│   ├── config.py                # 全局配置中心
│   ├── schemas.py               # Pydantic 接口请求/响应模型
│   ├── parser.py                # 基于 MarkItDown 的多格式解析器与文本清洗
│   ├── chunker.py               # 支持滑动窗口 overlap 的语义切分器与元数据构造
│   ├── indexer.py               # Chroma 持久化、本地向量化与文件 MD5 增量索引管理
│   ├── retriever.py             # 检索器（纯向量检索、bm25s+rjieba 混合检索、LLM Rerank 重排）
│   ├── generator.py             # 距离阈值分流、LRU 问答缓存与溯源问答生成
│   ├── api.py                   # FastAPI 路由与问答日志记录
│   └── static/index.html        # Web 交互演示工作台（含引用点击定位、关键词高亮与一键演示）
├── data/
│   ├── docs/                    # 4 份脱敏企业制度与 FAQ 文档（覆盖 MD/DOCX/PDF/TXT）
│   ├── chroma_db/               # Chroma 本地持久化向量库目录
│   ├── index_state.json         # 文件 MD5 增量索引状态与历史构建日志
│   └── eval_dataset.json        # 30 题标准评估数据集（明确答案12题 + 跨段综合10题 + 无答案8题）
├── eval/
│   ├── eval_retrieval.py        # 第一阶段：纯检索评估与距离阈值分布校准脚本
│   ├── eval_e2e.py              # 第二阶段：端到端评估与 LLM-as-Judge 逐条声明支持率评测脚本
│   └── utils_json.py            # JSON 数组优先解析与正则兜底工具
├── outputs/                     # 真实运行产出的检索评估、端到端评估与在线问答日志
└── docs/
    ├── 系统设计文档.md            # 系统设计文档
    ├── 技术报告.md               # 技术报告
    ├── 分工表与展示PPT.md        # 小组分工表与 5 分钟展示答辩大纲
    └── 展示PPT.pptx              # 5 分钟现场答辩演示 PowerPoint 幻灯片文件
```

---

## 二、 环境安装与配置

本项目全程使用 **`uv`** 管理 Python 虚拟环境与依赖（运行于 **CPython 3.14.10**）。

### 1. 安装依赖
在项目根目录下执行：
```bash
uv sync
```

### 2. 准备本地向量模型（统一存放于 `models/` 目录）
项目统一从 `models/` 目录下加载本地嵌入模型，并在 `.env` 中通过 `EMBEDDING_MODEL` 指定使用的模型目录名：
```bash
uv run hf download Qwen/Qwen3-Embedding-0.6B --local-dir ./models/Qwen3-Embedding-0.6B
```

### 3. 配置 `.env` 环境变量
复制 `.env.example` 为 `.env`，并填入 OpenAI 兼容接口的 `OPENAI_API_KEY` 与 `OPENAI_BASE_URL`（兼容 OpenAI、阿里云百炼、DeepSeek、vLLM 等服务）：
```bash
cp .env.example .env
```
`.env` 关键配置项说明：
```ini
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=deepseek-v4.1-flash-1017
EMBEDDING_MODEL=Qwen3-Embedding-0.6B
```

---

## 三、 启动服务与 Web 演示界面

### 方式一：本地 `uv` 一键启动
可通过启动脚本或 `uv run` 直接启动：
```bash
./run.sh
# 或
uv run uvicorn app.api:app --host 127.0.0.1 --port 8000
```

### 方式二：Docker Compose 容器化启动
```bash
docker compose up --build -d
```

- **Web 演示工作台**：浏览器访问 `http://127.0.0.1:8000`
- **Swagger 接口文档**：浏览器访问 `http://127.0.0.1:8000/docs`
- **演示 PPT 文件**：[`docs/展示PPT.pptx`](docs/展示PPT.pptx)

---

## 四、 API 调用示例（上传文档、重建/增量索引、问答）

### 1. 健康检查 (`GET /health`)
```bash
curl -s http://127.0.0.1:8000/health | jq
```

### 2. 查看文档列表 (`GET /documents`)
```bash
curl -s http://127.0.0.1:8000/documents | jq
```

### 3. 上传新文档 (`POST /documents/upload`)
支持上传 `.pdf`、`.docx`、`.txt`、`.md` 格式文档（单文件上限 20 MB）：
```bash
curl -X POST http://127.0.0.1:8000/documents/upload \
  -F "file=@data/docs/员工手册与考勤休假制度.md" | jq
```

### 4. 建索引 / 增量索引 / 全量重建索引 (`POST /index`)
- **基于文件 MD5 的增量索引**（默认跳过未变更文件，避免重复入库）：
```bash
curl -X POST http://127.0.0.1:8000/index \
  -H "Content-Type: application/json" \
  -d '{"mode": "incremental", "chunk_size": 400, "overlap": 80}' | jq
```
- **全量重建索引**（清空旧向量集合并按最新切分参数重新入库）：
```bash
curl -X POST http://127.0.0.1:8000/index \
  -H "Content-Type: application/json" \
  -d '{"mode": "rebuild", "chunk_size": 400, "overlap": 80}' | jq
```

### 5. 检索增强问答 (`POST /ask`)
支持通过 `mode` 参数在 `baseline`（基线）、`threshold_tuned`（阈值校准）、`hybrid`（BM25+向量混合检索）、`rerank`（LLM重排）、`optimized`（优化全开）之间自由切换：
```bash
# 示例 A：明确答案 / 专有编号查询
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "调用问答 API 返回错误码 ERR-4029 代表什么意思？默认单租户并发 QPS 上限是多少？",
    "mode": "optimized",
    "top_k": 4,
    "recall_k": 8
  }' | jq

# 示例 B：无答案题触发拒答路径 (path = "refusal")
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "公司员工股票期权（ESOP）的归属期是几年？离职后行权回购价格如何计算？",
    "mode": "optimized",
    "top_k": 4,
    "recall_k": 8
  }' | jq
```

### 6. 问答缓存统计与清理 (`GET /cache/stats`, `POST /cache/clear`)
```bash
curl -s http://127.0.0.1:8000/cache/stats | jq
curl -X POST http://127.0.0.1:8000/cache/clear | jq
```

---

## 五、 评估命令（复现全部实验指标）

### 1. 第一阶段：纯检索评估与距离阈值分布校准（不调用 LLM）
评测不同 `chunk_size/overlap` 参数消融、纯向量检索 vs BM25+向量 RRF 混合检索的 `Hit@K`、`Recall@K`、`Precision@K`、`MRR@K`（22 道有答案题），并单独统计 8 道无答案题的拒答准确率与 `dists[0]` 分布：
```bash
uv run python -m eval.eval_retrieval
```
结果自动输出至：`outputs/retrieval_eval_results.json`。

### 2. 第二阶段：端到端评估与 LLM-as-Judge 评测
固定生成模型 `deepseek-v4.1-flash-1017` 与温度 `0.1`，按控制变量法依次运行 `baseline`、`threshold_tuned`、`hybrid`、`optimized` 四种模式，评测答案正确性、忠实度（逐条声明支持率）、拒答准确率、平均/P95 延迟与 Token 消耗（严格区分生成 Token、Rerank Token 与裁判 Token），并导出 10 题人工抽检核验表：
```bash
uv run python -m eval.eval_e2e
```
结果自动输出至：`outputs/e2e_eval_results.json`。
