# Mini Mall 商城 · 智能客服 RAG

一个从零手写的商城后端（FastAPI + 异步 SQLAlchemy + MySQL），
在它之上扩展出一套**可本地完整运行**的智能客服知识库问答系统（RAG）。

```
技术栈：Python 3.13 · FastAPI · SQLAlchemy 2.0(异步) · MySQL
        LangChain 1.x · Chroma · BGE(bge-small-zh + bge-reranker-base)
```

---

## 一、这个项目对应哪些考核点

| 考核要求 | 本项目中的落点 | 位置 |
|---|---|---|
| Python + FastAPI 开发 | 完整分层后端，16 个接口，统一响应信封 / 异常处理 | `api/` `core/` |
| **PDF / Word 文件解析** | pypdf + python-docx，含**表格转 Markdown**、PDF 视觉换行还原 | `rag/loader.py` |
| **多轮会话记忆** | 近期原话 + 滚动摘要双层记忆，游标保证幂等 | `services/chat_service.py` |
| **向量检索** | BGE 本地向量模型 + Chroma 持久化，余弦距离 | `rag/embedder.py` `rag/vectorstore.py` |
| 项目可本地完整运行 | 四项启动自检 + 环境自检脚本 + 优雅降级 | `main.py` `scripts/check_env.py` |
| 熟练运用 LangChain 生态 | `RecursiveCharacterTextSplitter`、`ChatOpenAI`、`Chroma`、`Embeddings` 接口 | `rag/` |
| **检索重排** | BGE CrossEncoder 交叉编码器精排，失败自动降级 | `rag/reranker.py` |
| **上下文压缩** | 句子级向量筛选 + 邻居句保留（防断章取义） | `rag/compressor.py` |
| **Prompt 工程** | 4 个版本可切换，每版注明改动理由 | `rag/prompts.py` |
| **针对幻觉开展调优对比** | 22 题评估集 + LLM 裁判 + 6 组配置矩阵，指标是**幻觉率** | `eval/` |
| **输出完整优化文档** | 含实测数据、控制变量实验、踩坑记录 | [docs/RAG优化文档.md](docs/RAG优化文档.md) |

---

## 二、系统架构

### 2.1 两条流水线

整个系统就两件事：**把文档存进去**、**把答案取出来**。

```
【写入侧】离线，一次
  上传 → 解析 → 切分 → 向量化 → 双写（MySQL 存内容 / Chroma 存向量）

【查询侧】在线，每次提问
  提问 ─┬→ 装载会话记忆（摘要 + 近 3 轮原话）
        │
        ├→ 向量召回 recall_k=20        「求不漏」
        │       ↓
        ├→ 拒答判断（最高分 < 阈值？）  「不该答就别答」
        │       ↓
        ├→ 重排精排 → top_k=5          「求准」
        │       ↓
        ├→ 上下文压缩                   「求省」
        │       ↓
        └→ 组装 Prompt → 生成 → 埋点记录
```

> 检索链路的**顺序**是踩过坑才定下来的，见 §六「坑 4」。
> 一句话总结：**阈值管"要不要答"，重排管"先答哪个"**，两件事不能混。

### 2.2 分层依赖（单向，不可反向）

```
api  →  services  →  rag
 ↑          ↑         ↑
契约       业务      RAG 能力
```

`rag/` 里**不会出现 `import services`**，也不认识 FastAPI 和数据库。
好处是 RAG 能力可以脱离 Web 单独跑脚本、单独做评估——
后面做批量对比实验时，这一点省了非常多事。

---

## 三、实测数据

不是"理论上应该能提升"，是实际跑出来的数字。复现命令都附在后面。

### 3.1 相似度阈值标定（决定"什么内容有资格进 Prompt"）

用 12 个「知识库里有答案」+ 12 个「知识库里没答案」的问题，
测出两组的最高相似度分布：

| 分组 | 最高相似度区间 |
|---|---|
| A 组（有答案） | 0.5537 ~ 0.6815 |
| B 组（没答案） | 0.2482 ~ 0.4830 |

两组之间是一道 **0.4830 ~ 0.5537 的空隙**，完全没有重叠。
阈值取 `0.50` 落在缝隙里：**可回答问题全部保留，不可回答问题全部拦截**。

> 最初按"看起来合理"定的 0.25 **等于没设**——像"客服电话是多少"
> 这种知识库里根本没有的问题，最高分也有 0.445，照样被送进 Prompt。
>
> 复现：`.venv\Scripts\python.exe scripts\probe_threshold.py --csv`

### 3.2 切分粒度 → 检索与重排效果

固定其它变量，只改 `chunk_size`，14 个带标准答案的问题：

| chunk_size | 块数 | 向量检索 top1 | 重排后 top1 | 重排增益 |
|---|---|---|---|---|
| 200 | 38 | 12/14 | 11/14 | **−1** |
| 300 | 23 | 9/14 | 11/14 | +2 |
| **450** | 14 | 11/14 | **13/14** | +2 |
| 600 | 11 | 9/14 | 12/14 | +3 |

两个反直觉结论：

1. **块切得越碎不等于越好。** 200 字时向量检索最准（12/14），
   但重排后**反而变差**（11/14）——小块丢了上下文，重排失去判断依据。
2. **块太大时重排"增益"最大（+3），但绝对命中率上不去。**
   重排能纠正排序，补不回"关键信息被一大段无关文字稀释"的损失。

最终取 `chunk_size=450`。
> 诚实说明：14 题样本偏小，450 与 600 的差距（13 vs 12）未必统计显著。

> 复现：`.venv\Scripts\python.exe scripts\experiment_chunk.py --sizes 200 300 450 600`

### 3.3 Prompt 版本与幻觉率对比

<!-- EVAL_RESULT_START -->
⏳ 待填：需要可用的 `LLM_API_KEY`，运行 `eval\run_eval.py` 后回填
（6 组配置的幻觉率 / 引用合法性 / 延迟对比表）。
<!-- EVAL_RESULT_END -->

---

## 四、快速开始

### 1. 安装依赖

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

首次较慢（要下载 torch，约 200MB）。

### 2. 配置 `.env`

```ini
# 数据库
MYSQL_HOST=192.168.18.10
MYSQL_PASSWORD=123456
MYSQL_DB=mini_mall

# ★ 大模型：必填。不填时上传文档照常可用，但一提问会明确告诉你缺什么
LLM_API_KEY=你的密钥
LLM_BASE_URL=http://192.168.18.17:3000/v1
LLM_MODEL=deepseek-v4-flash

# ★ 模型下载源：国内必须配，否则 BGE 模型下不下来
HF_ENDPOINT=https://hf-mirror.com
```

> **换模型只需改两行。** 代码用 `ChatOpenAI` 走 **OpenAI 兼容协议**，
> 所以 DeepSeek 官方、局域网网关、通义、智谱、本机 Ollama 都能直接用，
> **代码一行都不用动**。这正是选兼容协议的价值。
>
> `LLM_API_KEY` 留空时，提问会返回
> 「还没有配置大模型 API Key，请在 .env 里加上 LLM_API_KEY=…」，
> 而不是丢一个看不懂的堆栈——这是刻意做的降级。

### 3. 建表

```powershell
.venv\Scripts\python.exe init_db.py
```

新建 4 张 RAG 表：`knowledge_docs`、`knowledge_chunks`、`chat_sessions`、`chat_messages`。
存量库上重复执行是安全的。

### 4. 生成演示文档（可选但推荐）

```powershell
.venv\Scripts\python.exe scripts\make_sample_docs.py
```

在 `data/sample/` 下生成 4 份电商客服文档，覆盖 PDF / DOCX / TXT / MD 四种格式。
内容刻意埋了跨文档交叉信息、表格、例外条款和"查不到答案"的话题。

### 5. 启动

```powershell
.venv\Scripts\python.exe main.py
```

- 聊天页面 http://127.0.0.1:8600/
- 接口文档 http://127.0.0.1:8600/docs

启动日志会做**四项自检**（数据库 / 大模型配置 / 向量模型预热 / 重排提示），
没配好的地方在这里就直接点出来，不用等讲到一半才发现。

### 6. 跑评估

```powershell
.venv\Scripts\python.exe eval\run_eval.py          # 6 组配置全跑
.venv\Scripts\python.exe eval\run_eval.py --dry-run # 只看题目，不调接口
.venv\Scripts\python.exe eval\run_eval.py --only v2-防幻觉
```

---

## 五、验收步骤（照着走一遍即可确认）

```powershell
# ① 环境自检：依赖、数据库、模型、大模型连通性
.venv\Scripts\python.exe scripts\check_env.py

# ② 启动服务
.venv\Scripts\python.exe main.py
```

然后按顺序验证：

| # | 操作 | 预期结果 |
|---|---|---|
| 1 | 看启动日志 | 四项自检都有明确结论（✓ 或 明确提示） |
| 2 | 打开 http://127.0.0.1:8600/ | 出现聊天页，左侧侧边栏显示知识库概况 |
| 3 | 依次上传 `data/sample/` 下 4 个文件 | 4 份都变成「已就绪」，共 14 个切片 |
| 4 | 再上传同一个文件 | 提示「该文件已存在，未重复入库」 |
| 5 | 点某份文档的「查看切片」 | 能看到切片内容——**用来确认切分没切坏** |
| 6 | `POST /api/knowledge/retrieve-debug` 问「退货要几天」 | 并排返回重排前后的排序，可直观对比 |
| 7 | 在聊天框问「生鲜食品可以退货吗」 | 回答正确，并标注「依据：[n]」 |
| 8 | 问「客服电话是多少」 | **如实回答查不到**，而不是编一个号码 |
| 9 | 追问「那多久能到账」，观察是否记得上文 | 多轮记忆生效 |
| 10 | 切换 Prompt 版本 v1 → v3，问同一个问题 | 回答风格与引用行为明显不同 |

> 第 3 步的「14 个切片」是在 `CHUNK_SIZE=450` 下的结果。
> 第 8 步是**幻觉防线**的关键验收点：知识库没有的内容必须承认不知道。

---

## 六、开发中实际修掉的问题

这一节是这份交付里最"实"的部分——下面每一条都是跑起来才暴露的，
不是照着文档抄的注意事项。详细排查过程见
[优化文档 §6](docs/RAG优化文档.md)。

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | 上传报 500，但数据**全部入库成功** | `created_at` 由数据库生成，ORM 未刷新就读取，异步 Session 抛 `MissingGreenlet` | 提交后显式 `await db.refresh(doc)` |
| 2 | 多文档问答崩 500 | 压缩模块把**全局下标**当**文档内下标**用；文档 0 恰好对齐，单文档测试永远绿 | 存下标时就换算成局部下标 + 越界防御 |
| 3 | 没配 key 时提示"服务器开小差了" | `get_llm()` 写在 `try` **外面**，异常漏到兜底分支 | 移进 `try`；`main.py` 单独注册 `LLMError` 处理器，code 用 503 |
| 4 | 重排参数开着，却**一次都没执行过** | 阈值过滤被放在检索阶段，候选只剩 2–3 条，触发"候选不够就跳过重排" | 拆开两件事：阈值负责拒答判断，重排负责排序 |
| 5 | PDF 标题被粘进正文（`退换货政策生效日期：`） | 换行合并规则太粗暴，把结构换行也当成排版换行 | 用「行宽 75 分位 + 句末标点 + 新块开头标记」三重判断 |
| 6 | Word 里表格文字丢失 | `doc.paragraphs` **不包含表格内容**，而运费/时效恰好都在表格里 | 遍历 `doc.element.body` 保持顺序，表格转 Markdown |

第 4 条最有代表性：**参数全开、看起来正常、实际一次没跑**，而且不报任何错。
这类"悄悄变差"的问题，只能靠打印中间结果才能发现。

第 2 条则典型在**只在特定数据组合下才炸**——单文档测试永远是绿的。

---

## 七、目录结构

```
mini-mall/
├── main.py                     启动入口（含四项启动自检）
├── init_db.py                  建表脚本
│
├── core/                       基础设施（最底层，不依赖任何业务）
│   ├── config.py               全部配置项（含 RAG 调参项与调参依据）
│   ├── database.py             异步引擎与会话
│   ├── exception.py            统一异常 → Result 信封
│   ├── security.py             密码哈希 + JWT
│   └── deps.py                 可选当前用户依赖
│
├── models/                     数据库模型
│   ├── user / product / cart / order      商城业务
│   ├── knowledge.py            ★ 知识文档 + 切片
│   └── chat.py                 ★ 会话 + 消息（含调优埋点）
│
├── schemas/rag.py              ★ RAG 出入参契约
│
├── services/                   业务逻辑
│   ├── knowledge_service.py    ★ 文档入库、重建索引、数据对账
│   ├── chat_service.py         ★ 会话记忆、滚动摘要
│   └── rag_service.py          ★ 问答主流程（含流式 SSE）
│
├── rag/                        ★ RAG 能力层（不认识 Web 和数据库）
│   ├── loader.py               文档解析 PDF/Word/TXT/MD
│   ├── splitter.py             中文友好切分
│   ├── embedder.py             BGE 向量化（查询加指令前缀，文档不加）
│   ├── vectorstore.py          Chroma 封装（距离→相似度转换）
│   ├── reranker.py             重排（含优雅降级）
│   ├── compressor.py           上下文压缩（句子级 + 邻居保留）
│   ├── retriever.py            检索编排（召回→拒答→重排→压缩）
│   ├── prompts.py              Prompt 模板与版本管理
│   └── llm.py                  大模型接入与错误翻译
│
├── api/                        接口层（16 个接口）
│   ├── knowledge_api.py        ★ 知识库接口
│   └── chat_api.py             ★ 对话接口
│
├── eval/                       ★ 评估
│   ├── dataset.py              22 题评估集（可答 13 / 不可答 6 / 陷阱 3）
│   └── run_eval.py             评估执行 + 对比表 + 逐题明细落盘
│
├── scripts/                    辅助脚本
│   ├── check_env.py            环境自检
│   ├── probe_threshold.py      相似度阈值标定
│   ├── experiment_chunk.py     切分粒度实验
│   └── make_sample_docs.py     生成演示文档
│
├── static/chat.html            聊天页面（单文件，无需构建）
├── docs/RAG优化文档.md         ★ 优化过程与实验数据
└── data/                       运行时数据（uploads / chroma / sample）
```

---

## 八、辅助脚本

| 脚本 | 作用 | 什么时候用 |
|---|---|---|
| `check_env.py` | 依赖、数据库、模型、大模型连通性一次验完 | **启动不起来时第一个跑它** |
| `probe_threshold.py` | 标定相似度阈值，找"有答案/没答案"的可分区间隙 | 换了 embedding 模型后必须重跑 |
| `experiment_chunk.py` | 实验：切分粒度对检索/重排的影响 | 调 `CHUNK_SIZE` 前先跑 |
| `make_sample_docs.py` | 生成 4 份演示文档 | 第一次跑项目时 |

> `probe_threshold.py` 的价值不只是"算出该填多少"，它还会给出一个
> **否定结论**：如果两组分数重叠，说明单靠阈值救不了这个场景，
> 硬调只会顾此失彼——脚本会直接告诉你该去查切分还是该换模型，
> 而不是让你在阈值上反复试。**会说"这条路走不通"的工具，比只会算数的有用。**

---

## 九、常用接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/knowledge/upload` | 上传文档（同步解析入库） |
| GET | `/api/knowledge/docs` | 文档列表 |
| GET | `/api/knowledge/docs/{id}/chunks` | 查看切片（排查切分问题） |
| POST | `/api/knowledge/docs/{id}/rebuild` | 重建索引（换模型 / 改参数后必做） |
| GET | `/api/knowledge/stats` | 知识库概况（含 MySQL 与 Chroma 两侧对账） |
| GET | `/api/knowledge/prompts` | 查看全部 Prompt 版本 |
| POST | `/api/knowledge/retrieve-debug` | **检索调试**：对比重排前后的排序 |
| POST | `/api/chat/ask` | 提问（非流式，适合脚本/评估） |
| POST | `/api/chat/ask/stream` | 提问（流式 SSE） |
| POST | `/api/chat/sessions` | 新建会话 |
| GET | `/api/chat/sessions/{id}/messages` | 会话历史 |

`retrieve-debug` 值得单独试一次：输入「退货要几天」，
它会并排返回向量检索的原始排序和重排后的排序——
**一眼就能看出重排到底有没有起作用**，比看日志快得多。

---

## 十、已知限制与后续计划

如实记录，避免过度承诺。

### 当前限制

| 限制 | 影响 | 改进方向 | 优先级 |
|---|---|---|---|
| 文档同步处理 | 大文件请求等待久 | 后台队列 + 状态轮询（`status` 状态机已预留） | 高 |
| 不支持扫描版 PDF | 扫描件无法入库（会明确报错，不静默入库空内容） | 接入 PaddleOCR / RapidOCR | 中 |
| 评估集仅 22 题 | 统计置信区间宽 | 扩到 100+ 题并入 CI | 中 |
| 重排模型约 1GB | 首次使用需下载（失败自动降级为不重排） | 换更小的蒸馏重排模型 | 低 |
| 流式中断的重试 | 半截回答会存库并标注「[回答中断]」，但前端重试未做 | 前端加重试逻辑 | 低 |

### 有意不做的

- **多租户知识库隔离**：当前是全局知识库，所有用户共享。
- **上传接口鉴权**：未做管理员校验（演示便利优先）。
  **上线前必须补上**，否则任何人都能往知识库里塞内容。
- **未引入 RAGAS 等评估框架**：22 题、单轮问答的规模，
  自建脚本（约 300 行）够用，接入框架的收益不足以抵消学习成本。
- **未做查询改写（Query Rewriting）**：多轮中的指代（"那它多少钱"）
  目前靠对话历史兜住。查询改写能改善，但会多一次 LLM 调用，延迟翻倍。

---

## 十一、常见问题

**Q：启动时报「向量模型加载失败」**
A：检查 `.env` 的 `HF_ENDPOINT=https://hf-mirror.com`。
国内直连 huggingface.co 会 **DNS 解析失败**，必须走镜像。

**Q：提问报「还没有配置大模型 API Key」**
A：在 `.env` 填 `LLM_API_KEY`。注意复制时**不要带引号或空格**。
上传文档不需要它，只有提问需要。

**Q：提问报 401 / Invalid token**
A：密钥无效或过期。若用的是网关服务，确认该 key 有对应模型的权限。

**Q：检索结果差、答非所问**
A：按这个顺序排查，**不要一上来就改 Prompt**：
1. `GET /api/knowledge/docs/{id}/chunks` —— 看切分有没有切坏
2. `POST /api/knowledge/retrieve-debug` —— 看重排前后的排序差异
3. 跑 `scripts/probe_threshold.py` —— 看当前阈值是否还在"空隙"里
4. 最后才考虑调 `RETRIEVE_SCORE_THRESHOLD` 和 `CHUNK_SIZE`

**Q：日志刷得太快**
A：把 `.env` 的 `DB_LOG` 改成 `False`。它为 True 时会打印每一条 SQL，
调 RAG 时真正有用的日志会被淹没。

**Q：改了 `CHUNK_SIZE` 或换了 embedding 模型，旧数据还能用吗？**
A：**不能直接混用。** 旧向量是用旧规则算的，和新向量不同源，
放在同一空间里比较没有意义（而且不会报错，只会结果变怪）。
必须对每份文档调用 `POST /api/knowledge/docs/{id}/rebuild` 重建索引。

**Q：重排模型下载失败怎么办？**
A：不影响功能。重排会自动降级为按向量顺序返回，
降级情况会在日志和响应里如实体现（`rerank_enabled=false`），不会假装成功。
