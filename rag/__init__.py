"""rag 包：RAG（检索增强生成）的核心能力层。

为什么要单独抽一个包，而不是把逻辑写进 services？

因为这一层是可以脱离 Web 框架独立运行的纯能力：
    给它一个文件路径 → 吐出文本块
    给它一个问题     → 吐出相关片段
它不知道 FastAPI，不知道数据库，不知道 HTTP。
好处很直接：
1. 能单独写脚本跑评估（做调优对比时必须能脱离接口批量跑）
2. 换 Web 框架 / 换向量库时，改动被限制在这一层

分层依赖方向是单向的，不能反向：
    api → services → rag
反过来 rag 里 import services 就说明分层烂了。
"""

__all__ = [
    "loader",
    "splitter",
    "embedder",
    "vectorstore",
    "retriever",
    "reranker",
    "compressor",
    "prompts",
    "llm",
]
