"""配置读取：把 .env 文件里的内容，变成代码里能直接用的对象。

规范要点：
1. 一个类管所有配置，类型写在注解上（pydantic 会自动校验和转换）
2. 配置项全部大写，和 .env 里的名字一致
3. 敏感信息只放 .env，代码里只留一个"默认值"兜底
"""

from urllib.parse import quote_plus

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """项目全部配置项。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------- 应用 ----------------
    PROJECT_NAME: str = "Mini Mall 商城"
    APP_ENV: str = "dev"
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8600
    DEBUG: bool = True

    # ---------------- 数据库 ----------------
    MYSQL_HOST: str = "127.0.0.1"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DB: str = "mini_mall"
    DB_LOG: bool = False

    # ---------------- 安全 ----------------
    TOKEN_SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # ================ 智能客服 RAG：大模型 ================
    # 走 DeepSeek 的 OpenAI 兼容接口，所以 base_url 填他们的地址，
    # 代码里就能直接用 LangChain 的 ChatOpenAI——不用为每家厂商写一套适配。
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_MODEL: str = "deepseek-chat"
    # 温度：RAG 场景必须压低。温度越高模型越"爱发挥"，而我们要的是"照着原文说"。
    # 0.1 是实践出来的经验值，压到 0 会让回答变得生硬重复。
    LLM_TEMPERATURE: float = 0.1
    LLM_TIMEOUT: int = 60
    LLM_MAX_RETRIES: int = 2

    # ================ 智能客服 RAG：向量化 ================
    # 本地跑的 BGE 中文小模型，首次使用会自动从 HuggingFace 下载（约 100MB）
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_DEVICE: str = "cpu"
    # ★ 国内网络下 huggingface.co 通常直接不可达（DNS 都解析不了），
    #   必须走镜像站，否则模型永远下载不下来。留空则用官方源。
    #   注意：这个值要落成真正的进程环境变量才生效，见文件末尾。
    HF_ENDPOINT: str = "https://hf-mirror.com"
    # BGE 系列模型检索时必须给「查询」加这个前缀，文档侧不加。
    # 忘了加会让召回率明显下降，是个非常隐蔽的坑，详见 rag/embedder.py
    EMBEDDING_QUERY_INSTRUCTION: str = "为这个句子生成表示以用于检索相关文章："

    # ================ 智能客服 RAG：向量库 ================
    # Chroma 用本地文件持久化，不需要额外起服务——这是"可本地完整运行"的关键
    CHROMA_DIR: str = "./data/chroma"
    CHROMA_COLLECTION: str = "mall_knowledge"

    # ================ 智能客服 RAG：检索与重排 ================
    RETRIEVE_RECALL_K: int = 20      # 粗排召回多少条进候选池
    RETRIEVE_TOP_K: int = 5          # 精排后最终喂给模型几条
    # 相似度下限。低于它的片段直接扔掉——宁可说"我不知道"，也不要拿不相关的
    # 内容去凑数，那正是幻觉的主要来源之一。
    #
    # ★ 这个值是【实测调出来的】，不是拍脑袋定的。经验过程：
    #   最初按"看起来合理"设成 0.25，跑起来发现完全不起作用——
    #   拿一个知识库里根本没有答案的问题（比如"客服电话是多少"）去检索，
    #   最高分仍然有 0.445。也就是说 0.25 这个阈值等于没设，
    #   无关内容照样被喂给模型，模型就只能硬凑一个答案。
    #
    #   实测本模型（bge-small-zh-v1.5 + 余弦距离）的分数分布：
    #       真正相关的片段：0.51 ~ 0.65
    #       完全无关的片段：0.38 ~ 0.45
    #   分界线落在 0.5 附近，所以取 0.5。
    #
    #   注意：这个值【和模型强相关】。换成别的 embedding 模型，
    #   相似度的整体分布会变（BGE 中文系列的分数普遍偏高），
    #   阈值必须重新实测，不能照搬。
    RETRIEVE_SCORE_THRESHOLD: float = 0.5
    # 重排开关。做成可关是刻意的：reranker 模型有 1GB+，
    # 模型没下下来时能自动降级成"只用向量检索"，不至于整个服务不可用
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "BAAI/bge-reranker-base"

    # ================ 智能客服 RAG：上下文压缩 ================
    COMPRESS_ENABLED: bool = True
    # 压缩时保留的"高相关句子"数量
    COMPRESS_KEEP_SENTENCES: int = 6

    # ================ 智能客服 RAG：文档切分 ================
    # 切分粒度。★ 这个值是用 scripts/experiment_chunk.py 实测出来的，
    # 不是拍脑袋定的。实验固定其它变量、只改 chunk_size，
    # 用 14 个带标准答案的问题测「top1 片段是否命中答案」：
    #
    #     chunk_size   块数   向量top1   重排top1   重排增益
    #        200       38     12/14      11/14       -1     ← 切太碎，重排反而帮倒忙
    #        300       23      9/14      11/14       +2
    #        450       14     11/14      13/14       +2     ← 综合最好
    #        600       11      9/14      12/14       +3     ← 增益最大，但绝对值低
    #
    # 两个反直觉的结论：
    #   1. 切得越碎不等于越好。块太小（200）时向量检索本身已经很准（12/14），
    #      重排反而把它搞乱了（掉到 11/14）——小块丢掉了太多上下文，
    #      重排模型反而缺少判断依据。
    #   2. 块太大时重排「增益」最大（+3），但绝对命中率上不去。
    #      说明重排能纠正排序，却补不回「关键信息被一大段无关文字稀释」的损失。
    #
    # 注意：14 道题样本偏小，450 与 600 的差距（13 vs 12）统计上未必显著，
    # 取 450 是「综合最优 + 不过度优化」的务实选择，不是精确最优解。
    CHUNK_SIZE: int = 450
    # 相邻块重叠这么多字。为什么必须重叠？见 rag/splitter.py 的说明
    CHUNK_OVERLAP: int = 60

    # ================ 智能客服 RAG：会话记忆 ================
    # 最近 N 轮原话直接进 Prompt（1 轮 = 一问一答）
    MEMORY_RECENT_TURNS: int = 3
    # 超过这个轮数就触发摘要压缩，把更早的内容揉成一段话
    MEMORY_SUMMARY_TRIGGER_TURNS: int = 5

    # ================ 智能客服 RAG：Prompt 版本 ================
    # ★ 这个值会被写进每条回答的埋点里。
    #   做 A/B 对比时改配置文件或环境变量就能切换，不用动代码——
    #   没有它，"优化后幻觉变少了"就是一句没法验证的话。
    PROMPT_VERSION: str = "v2"

    # ================ 文件上传 ================
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_MB: int = 20

    @property
    def database_url(self) -> str:
        """拼出 SQLAlchemy 用的异步连接串。

        坑点：密码里如果有 @ : / # 这类字符，直接塞进 URL 会解析错乱
        （@ 会被当成"主机名分隔符"）。所以必须用 quote_plus 转义。
        """
        password = quote_plus(self.MYSQL_PASSWORD)
        return (
            f"mysql+aiomysql://{self.MYSQL_USER}:{password}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}"
            f"?charset=utf8mb4"
        )

    @property
    def llm_configured(self) -> bool:
        """大模型是否配置好了。

        为什么要单独写个判断？因为最常见的翻车方式是：
        装完依赖、启动服务、上传文档全都没问题，一到提问就抛一个
        "api_key client option must be set"——报错信息在几百行栈里，很难一眼看懂。
        有了这个属性，启动时就能提前打出一句人话提示。
        """
        return bool(self.LLM_API_KEY.strip())


# 全局唯一的配置对象，别处直接 from core.config import settings
settings = Settings()


# ★ 这里做了一件"看起来越界"的事：把配置写回进程环境变量。
#   为什么必须这么干？因为 huggingface_hub 只在【被 import 的那一刻】
#   读取 HF_ENDPOINT，之后再设就已经晚了。
#   core.config 是整个项目最早被导入的模块之一（基本所有模块都依赖它），
#   所以在这里落地是最稳的位置。setdefault 保证不会覆盖用户手动设的值。
import os as _os  # noqa: E402

if settings.HF_ENDPOINT:
    _os.environ.setdefault("HF_ENDPOINT", settings.HF_ENDPOINT)
