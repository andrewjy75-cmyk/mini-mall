"""知识库业务逻辑：文档上传、解析、入库、删除、重建索引。

这是把 rag 包的能力接到数据库上的那一层：rag 包只管"把文件变成块"，
这里负责"块该存哪、状态怎么流转、出错怎么办"。

★ 状态机（KnowledgeDoc.status）是本模块的核心，先把它画出来：

    上传 → pending ──开始处理──→ processing ──成功──→ ready
                                          └──失败──→ failed（error_msg 记录原因）

为什么要有状态，而不是"处理完再落库"？
因为解析+向量化是耗时操作（大文档数十秒）。如果接口同步等它做完，
用户会看着一个转圈的页面，而且中途失败就什么都不剩，连"是哪份文件失败了"都查不到。
先落一条 pending 记录，再更新状态，好处是：
1. 失败有痕迹，能重试、能追溯
2. 前端可以显示进度列表
3. 将来改成后台队列（Celery / ARQ）时，数据结构不用动

★ 关于"当前是同步处理"这个事实，我要如实说明：
  上传接口里是同步跑完解析+向量化的，这样演示时上传完立刻就能提问，体验最顺。
  代价是大文件（比如 200 页 PDF）会让这个请求等很久。
  已知限制，改进方向写在文档里（改成后台任务 + 前端轮询 status）。
  把限制写清楚，比假装它不存在要专业。
"""

import hashlib
import logging
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.exception import BizException
from models.knowledge import (
    DOC_STATUS_FAILED,
    DOC_STATUS_PROCESSING,
    DOC_STATUS_READY,
    KnowledgeChunk,
    KnowledgeDoc,
)
from rag import vectorstore
from rag.loader import DocumentParseError, is_supported, load_document, supported_extensions_text
from rag.splitter import split_documents
from util.id_util import new_id

logger = logging.getLogger(__name__)

# 分片写入的大小：1MB。
# 为什么要分片？上传一个 20MB 的 PDF，如果一次 read() 全读进内存，
# 并发几个用户就能把服务的内存打满。分片读写是处理上传的标配做法。
WRITE_CHUNK_SIZE = 1024 * 1024


def get_upload_dir() -> Path:
    """上传目录的绝对路径（理由同 vectorstore.get_chroma_dir）。"""
    path = Path(settings.UPLOAD_DIR)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    path.mkdir(parents=True, exist_ok=True)
    return path


class KnowledgeService:
    """知识库相关的业务逻辑。"""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ==================== 查询 ====================

    async def list_docs(
        self, page: int = 1, page_size: int = 10, status: str | None = None
    ) -> tuple[list[KnowledgeDoc], int]:
        """分页查文档列表。"""
        conditions = []
        if status:
            conditions.append(KnowledgeDoc.status == status)

        count_stmt = select(func.count()).select_from(KnowledgeDoc)
        list_stmt = select(KnowledgeDoc).order_by(KnowledgeDoc.created_at.desc())

        if conditions:
            count_stmt = count_stmt.where(*conditions)
            list_stmt = list_stmt.where(*conditions)

        total = await self.db.scalar(count_stmt) or 0
        list_stmt = list_stmt.offset((page - 1) * page_size).limit(page_size)
        docs = list((await self.db.scalars(list_stmt)).all())
        return docs, total

    async def get_doc(self, doc_id: str) -> KnowledgeDoc:
        """按 id 取文档，取不到就抛业务异常。"""
        doc = await self.db.get(KnowledgeDoc, doc_id)
        if doc is None:
            raise BizException("文档不存在或已被删除", code=2001)
        return doc

    async def list_chunks(self, doc_id: str, limit: int = 20) -> list[KnowledgeChunk]:
        """看某份文档被切成了什么样。

        这个接口是做调优时的排查利器：效果不好的时候，
        第一件事就是先看看"切分是不是切坏了"，而不是急着改 Prompt。
        """
        stmt = (
            select(KnowledgeChunk)
            .where(KnowledgeChunk.doc_id == doc_id)
            .order_by(KnowledgeChunk.chunk_index)
            .limit(limit)
        )
        return list((await self.db.scalars(stmt)).all())

    # ==================== 上传与处理 ====================

    async def upload(
        self, filename: str, content: bytes, title: str = ""
    ) -> tuple[KnowledgeDoc, bool]:
        """上传一个文件并完成入库。

        返回 (文档对象, 是否重复)。重复时不会重复入库，直接返回已有的那份。

        流程：校验 → 算指纹 → 判重 → 落盘 → 落库 → 解析切分 → 向量化 → 置就绪
        """
        # ---- 1. 校验扩展名 ----
        # 在写盘之前就拦掉，不要等解析失败了再报错
        if not is_supported(filename):
            raise BizException(
                f"不支持的文件格式。目前支持：{supported_extensions_text()}", code=2002
            )

        # ---- 2. 校验大小 ----
        # 注意：真实生产环境还要在网关/反代层也限制一次，
        # 否则恶意的大文件请求会先把内存吃掉。这里只是应用层的第一道闸
        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise BizException(
                f"文件超过 {settings.MAX_UPLOAD_MB}MB 限制，请压缩后再上传", code=2003
            )
        if not content:
            raise BizException("上传的文件是空的", code=2004)

        # ---- 3. 内容指纹 + 判重 ----
        # 用内容哈希而不是文件名判重：同一个文件改个名字再传，也应该被认出来
        file_hash = hashlib.sha256(content).hexdigest()
        existing = await self.db.scalar(
            select(KnowledgeDoc)
            .where(KnowledgeDoc.file_hash == file_hash)
            .where(KnowledgeDoc.status != DOC_STATUS_FAILED)
            .limit(1)
        )
        if existing is not None:
            logger.info("检测到重复文件 %s，复用已有文档 %s", filename, existing.id)
            return existing, True

        # ---- 4. 落盘 ----
        doc_id = new_id()
        suffix = Path(filename).suffix.lower()
        # ★★ 安全要点：文件名由我们自己生成，绝不用用户传上来的 filename 拼路径。
        #     用户可以在文件名里塞 "../../../windows/system32/xxx"，
        #     一旦拼进路径就直接写到项目外面去了——这叫路径穿越攻击。
        #     用 doc_id 当文件名，同时天然解决了同名文件互相覆盖的问题。
        #     原始文件名只存进数据库当"展示用"，绝不参与任何路径拼接。
        stored_path = get_upload_dir() / f"{doc_id}{suffix}"
        stored_path.write_bytes(content)

        # ---- 5. 落库（pending 状态）----
        doc = KnowledgeDoc(
            id=doc_id,
            title=(title or Path(filename).stem)[:200],
            filename=filename[:255],
            file_type=suffix.lstrip("."),
            file_size=len(content),
            file_hash=file_hash,
            stored_path=str(stored_path),
            status=DOC_STATUS_PROCESSING,
        )
        self.db.add(doc)
        await self.db.commit()

        # ---- 6. 解析 → 切分 → 向量化 ----
        try:
            await self._process(doc)
        except DocumentParseError as exc:
            await self._mark_failed(doc, str(exc))
            raise BizException(f"文档解析失败：{exc}", code=2005) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("处理文档 %s 时发生未预期错误", doc.id)
            await self._mark_failed(doc, f"处理失败：{exc}")
            raise BizException("文档处理失败，请查看服务端日志", code=2006) from exc

        return doc, False

    async def _process(self, doc: KnowledgeDoc) -> None:
        """解析、切分、向量化、写切片表。"""
        # 解析
        documents = load_document(
            doc.stored_path,
            # 把文档信息挂进 metadata，一路带到向量库，
            # 最终用于回答里的引用溯源（"这句话出自《退换货政策》第 2 块"）
            options=_build_parse_options(doc),
        )

        # 切分
        chunks = split_documents(
            documents,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            doc_id=doc.id,
            doc_title=doc.title,
        )
        if not chunks:
            raise DocumentParseError("文档切分后没有任何内容，可能是空文件")

        # 预生成 id：MySQL 和向量库要用同一个 id 才能对上
        chunk_ids = [new_id() for _ in chunks]
        for chunk, chunk_id in zip(chunks, chunk_ids):
            chunk.metadata["chunk_id"] = chunk_id

        # ---- 先写切片表（MySQL 是主数据）----
        self.db.add_all(
            [
                KnowledgeChunk(
                    id=chunk_id,
                    doc_id=doc.id,
                    chunk_index=chunk.metadata["chunk_index"],
                    content=chunk.page_content,
                    char_count=len(chunk.page_content),
                )
                for chunk, chunk_id in zip(chunks, chunk_ids)
            ]
        )
        await self.db.commit()

        # ---- 再写向量库 ----
        # 顺序为什么是"先 MySQL 后向量库"？
        # 因为 MySQL 是唯一事实来源。万一向量库写失败，
        # MySQL 里还有完整的切片记录，可以一键重建索引（rebuild 接口就是干这个的）。
        # 反过来先写向量库的话，一旦 MySQL 失败，就会留下一批
        # 谁也不知道属于谁的孤儿向量，只能整库清空重来。
        try:
            vectorstore.add_chunks(chunks, chunk_ids)
        except Exception as exc:  # noqa: BLE001
            # 补偿：把刚写进 MySQL 的切片删掉，保持两边一致，
            # 否则会出现"切片表里有记录、但检索不到"的幽灵数据
            await self.db.execute(
                delete(KnowledgeChunk).where(KnowledgeChunk.doc_id == doc.id)
            )
            await self.db.commit()
            raise RuntimeError(f"写入向量库失败：{exc}") from exc

        # ---- 置为就绪 ----
        doc.status = DOC_STATUS_READY
        doc.chunk_count = len(chunks)
        doc.char_count = sum(len(c.page_content) for c in chunks)
        doc.error_msg = ""
        await self.db.commit()

        # ★★ 必须显式 refresh，这是一个非常隐蔽的坑，务必保留这行：
        #
        #   created_at / updated_at 是【数据库端】生成的
        #   （模型里写的 server_default=func.now()），插入之后
        #   ORM 并不知道数据库给它们填了什么值。
        #
        #   如果这里不刷新，等到接口层执行 KnowledgeDocOut.model_validate(doc)
        #   去读 doc.created_at 时，SQLAlchemy 会临时发起一次查询来补这个字段 ——
        #   而这是"隐式惰性加载"。在异步 Session 里，这种同步 IO 会直接抛：
        #       MissingGreenlet: greenlet_spawn has not been called
        #
        #   最坑的地方在于它的表现：数据其实已经完整入库了（文档、切片、
        #   向量全都在），但接口返回"服务器开小差了"，从报错完全看不出
        #   是"少刷新了一个字段"导致的。所以在这里一次性刷掉。
        await self.db.refresh(doc)

        logger.info("文档入库完成：%s（%s 块）", doc.title, doc.chunk_count)

    async def _mark_failed(self, doc: KnowledgeDoc, reason: str) -> None:
        """把文档标记为失败，并把原因记下来。

        失败时【不删除】记录，这是刻意的：
        用户需要看到"哪份文件没成功、为什么"，而不是传完就消失。
        """
        doc.status = DOC_STATUS_FAILED
        doc.error_msg = reason[:1000]
        await self.db.commit()

    # ==================== 删除与重建 ====================

    async def delete_doc(self, doc_id: str) -> None:
        """删除文档：MySQL 记录、向量、原始文件三处都要清干净。"""
        doc = await self.get_doc(doc_id)

        # 1) 先查切片 id，用于按 id 删向量（兜底路径）
        chunk_ids = list(
            (
                await self.db.scalars(
                    select(KnowledgeChunk.id).where(KnowledgeChunk.doc_id == doc_id)
                )
            ).all()
        )

        # 2) 删向量：先试按 doc_id 批量删（快），失败再按 id 逐个删（慢但可靠）
        deleted = vectorstore.delete_by_doc_id(doc_id)
        if not deleted and chunk_ids:
            vectorstore.delete_by_ids(chunk_ids)

        # 3) 删切片和文档记录
        await self.db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.doc_id == doc_id))
        await self.db.delete(doc)
        await self.db.commit()

        # 4) 删原始文件。
        #    放在最后且容错——文件删不掉（比如被别的进程占用）不该让整个删除失败，
        #    数据库记录已经清了，用户看到的结果是一致的
        try:
            path = Path(doc.stored_path)
            if path.exists():
                path.unlink()
        except OSError as exc:
            logger.warning("删除原始文件失败（不影响业务）：%s", exc)

        logger.info("文档已删除：%s（%s 块）", doc.title, len(chunk_ids))

    async def rebuild_index(self, doc_id: str) -> KnowledgeDoc:
        """重建索引：用当前配置重新解析、切分、向量化这份文档。

        为什么需要这个功能？因为它解决的是真实发生过的问题：
        换 embedding 模型、调 chunk_size、改切分策略之后，
        旧数据的向量是用旧规则算的，和新数据「不同源」，
        放在同一个空间里比较是没有意义的——检索会变得莫名其妙。
        这时候就必须把旧数据全部重新处理一遍。

        ★ 这是"可维护性"的具体体现：一个 RAG 系统如果没法重建索引，
          那它的切分参数和向量模型就永远定死在第一版，再也改不动了。
        """
        doc = await self.get_doc(doc_id)

        if not Path(doc.stored_path).exists():
            raise BizException(
                "原始文件已丢失，无法重建索引。请删除该文档后重新上传。", code=2007
            )

        logger.info("开始重建索引：%s", doc.title)

        # 1) 清掉旧的向量和切片
        vectorstore.delete_by_doc_id(doc.id)
        await self.db.execute(
            delete(KnowledgeChunk).where(KnowledgeChunk.doc_id == doc.id)
        )
        await self.db.commit()

        # 2) 重新走一遍处理流程
        doc.status = DOC_STATUS_PROCESSING
        doc.chunk_count = 0
        await self.db.commit()
        try:
            await self._process(doc)
        except Exception as exc:  # noqa: BLE001
            await self._mark_failed(doc, f"重建失败：{exc}")
            raise BizException(f"重建索引失败：{exc}", code=2008) from exc

        return doc

    # ==================== 统计 ====================

    async def stats(self) -> dict:
        """知识库概况。健康检查接口会用到。

        这里顺便把 MySQL 和向量库的条数【对账】——
        两边数量对不上就说明有数据不一致，这是个很有价值的监控点。
        """
        total_docs = await self.db.scalar(
            select(func.count()).select_from(KnowledgeDoc)
        ) or 0
        ready_docs = await self.db.scalar(
            select(func.count())
            .select_from(KnowledgeDoc)
            .where(KnowledgeDoc.status == DOC_STATUS_READY)
        ) or 0
        failed_docs = await self.db.scalar(
            select(func.count())
            .select_from(KnowledgeDoc)
            .where(KnowledgeDoc.status == DOC_STATUS_FAILED)
        ) or 0
        total_chunks = await self.db.scalar(
            select(func.count()).select_from(KnowledgeChunk)
        ) or 0

        try:
            vector_count = vectorstore.count_documents()
        except Exception as exc:  # noqa: BLE001
            logger.warning("统计向量条数失败：%s", exc)
            vector_count = -1

        return {
            "total_docs": total_docs,
            "ready_docs": ready_docs,
            "failed_docs": failed_docs,
            "total_chunks": total_chunks,       # MySQL 里的切片数
            "vector_count": vector_count,       # 向量库里的条数
            # 两边应当相等。不等就说明有数据没同步上，需要重建索引
            "consistent": vector_count == total_chunks,
        }


def _build_parse_options(doc: KnowledgeDoc):
    """构造解析选项，把文档标识挂进 metadata。"""
    from rag.loader import ParseOptions

    return ParseOptions(
        extra_metadata={"doc_id": doc.id, "doc_title": doc.title}
    )
