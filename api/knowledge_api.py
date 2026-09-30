"""知识库接口：上传文档、管理文档、查看切片、检索调优。

分层约定（和 user_api.py 完全一致）：
    这一层只做三件事 —— 收参数、调 service、套 Result 信封
    业务规则一行都不写在这里

★ 关于上传接口用同步处理这件事，接口描述里如实写清楚了。
  面试或考核时被问到"大文件怎么办"，能直接答上来：
  "当前是同步的，已知限制；改进方案是丢到后台队列 + 前端轮询 status，
   数据结构和状态机已经为这个改造预留好了。"
  这比假装它已经很完美要可信得多。
"""

import logging

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from rag import prompts
from rag.retriever import aretrieve
from rag.vectorstore import count_documents
from schemas.general import Result
from schemas.rag import (
    ChunkOut,
    KnowledgeDocOut,
    KnowledgeDocPage,
    KnowledgeDocUpdateIn,
    RetrievedItem,
    RetrieveDebugIn,
    RetrieveDebugOut,
    UploadResultOut,
)
from services.knowledge_service import KnowledgeService

logger = logging.getLogger(__name__)

knowledge_router = APIRouter(prefix="/knowledge", tags=["智能客服·知识库"])


def get_knowledge_service(db: AsyncSession = Depends(get_db)) -> KnowledgeService:
    """把 service 也做成依赖（理由同 user_api.py：便于统一替换和扩展）。"""
    return KnowledgeService(db)


# ==================== 文档管理 ====================


@knowledge_router.post(
    "/upload",
    response_model=Result[UploadResultOut],
    summary="上传知识文档（PDF / Word / TXT / Markdown）",
)
async def upload_document(
    file: UploadFile = File(..., description="要上传的文件"),
    title: str = Form(default="", description="文档标题，留空则用文件名"),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    """上传并立即完成解析、切分、向量化。

    同一个文件（内容相同）重复上传不会重复入库，会返回已有记录并标记 duplicate=true。

    ⚠️ 已知限制：这里是同步处理，大文件（如 200 页 PDF）会让请求等待较久。
    生产环境应改为后台任务 + 状态轮询，详见服务端文档。
    """
    content = await file.read()
    doc, duplicate = await service.upload(
        filename=file.filename or "unknown",
        content=content,
        title=title,
    )
    return Result.success(
        data=UploadResultOut(
            doc=KnowledgeDocOut.model_validate(doc), duplicate=duplicate
        ),
        message="该文件已存在，未重复入库" if duplicate else "上传并解析成功",
    )


@knowledge_router.get(
    "/docs",
    response_model=Result[KnowledgeDocPage],
    summary="文档列表（分页）",
)
async def list_documents(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    status: str | None = Query(default=None, description="按状态过滤：ready/failed/..."),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    """分页查看知识库里的文档。"""
    docs, total = await service.list_docs(page=page, page_size=page_size, status=status)
    return Result.success(
        data=KnowledgeDocPage(
            items=[KnowledgeDocOut.model_validate(d) for d in docs],
            total=total,
            page=page,
            page_size=page_size,
        )
    )


@knowledge_router.get(
    "/docs/{doc_id}",
    response_model=Result[KnowledgeDocOut],
    summary="文档详情",
)
async def get_document(
    doc_id: str,
    service: KnowledgeService = Depends(get_knowledge_service),
):
    doc = await service.get_doc(doc_id)
    return Result.success(data=KnowledgeDocOut.model_validate(doc))


@knowledge_router.get(
    "/docs/{doc_id}/chunks",
    response_model=Result[list[ChunkOut]],
    summary="查看文档的切片（调优排查用）",
)
async def list_document_chunks(
    doc_id: str,
    limit: int = Query(default=20, ge=1, le=200),
    service: KnowledgeService = Depends(get_knowledge_service),
):
    """看看这份文档被切成了什么样。

    效果不好的时候，第一件事应该是看切分，而不是急着改 Prompt——
    切分切坏了，Prompt 再怎么调也救不回来。
    """
    await service.get_doc(doc_id)   # 先确认文档存在，不存在会抛业务异常
    chunks = await service.list_chunks(doc_id, limit=limit)
    return Result.success(data=[ChunkOut.model_validate(c) for c in chunks])


@knowledge_router.put(
    "/docs/{doc_id}",
    response_model=Result[KnowledgeDocOut],
    summary="修改文档标题",
)
async def update_document(
    doc_id: str,
    data: KnowledgeDocUpdateIn,
    service: KnowledgeService = Depends(get_knowledge_service),
):
    doc = await service.get_doc(doc_id)
    doc.title = data.title
    await service.db.commit()
    await service.db.refresh(doc)

    # ★ 标题改了，向量库里的 doc_title 也会跟着过期。
    #   这里明确提示用户需要重建索引，而不是假装没关系——
    #   因为检索时返回的引用标题仍然会是旧标题，那会让用户困惑。
    return Result.success(
        data=KnowledgeDocOut.model_validate(doc),
        message="标题已修改。注意：已入库的切片仍带着旧标题，如需完全同步请重建索引",
    )


@knowledge_router.delete(
    "/docs/{doc_id}",
    response_model=Result[None],
    summary="删除文档（同时清理切片与向量）",
)
async def delete_document(
    doc_id: str,
    service: KnowledgeService = Depends(get_knowledge_service),
):
    await service.delete_doc(doc_id)
    return Result.success(message="删除成功")


@knowledge_router.post(
    "/docs/{doc_id}/rebuild",
    response_model=Result[KnowledgeDocOut],
    summary="重建索引（换模型/改切分参数后必做）",
)
async def rebuild_document_index(
    doc_id: str,
    service: KnowledgeService = Depends(get_knowledge_service),
):
    """用当前配置重新解析、切分、向量化这份文档。

    什么时候必须用它？改了 embedding 模型或 chunk_size 之后。
    旧向量和新向量「不同源」，混在同一个空间里比较是没有意义的。
    """
    doc = await service.rebuild_index(doc_id)
    return Result.success(data=KnowledgeDocOut.model_validate(doc), message="索引重建完成")


# ==================== 统计与调试 ====================


@knowledge_router.get(
    "/stats",
    response_model=Result[dict],
    summary="知识库概况（含 MySQL 与向量库的对账）",
)
async def knowledge_stats(
    service: KnowledgeService = Depends(get_knowledge_service),
):
    """返回文档数、切片数、向量数，以及两边是否一致。

    consistent=false 说明有数据没同步上（比如上次向量写入失败），
    需要重建索引。这是个很实用的健康检查点。
    """
    return Result.success(data=await service.stats())


@knowledge_router.get(
    "/prompts",
    response_model=Result[list[dict]],
    summary="列出所有 Prompt 版本及其改动说明",
)
async def list_prompt_versions():
    """把 Prompt 的版本演进直接暴露成接口。

    调优文档里写的"v2 加了不知道出口、v3 强制标注依据"，
    在这里可以现场看到，不用翻代码。
    """
    return Result.success(data=prompts.list_prompt_versions())


@knowledge_router.post(
    "/retrieve-debug",
    response_model=Result[RetrieveDebugOut],
    summary="检索调试：对比「向量检索」与「重排后」的排序差异",
)
async def retrieve_debug(
    data: RetrieveDebugIn,
):
    """把粗排和精排的结果并排返回。

    ★ 这个接口是讲清楚"重排到底做了什么"的最直接证据。
      演示时输入"退货要几天"，能亲眼看到重排把原本排第 7、第 9 的
      正确条款提到了第 1、第 2 位——比任何描述都有说服力。
    """
    # 强制开启重排来做对比（同时拿到两种排序）
    result = await aretrieve(
        data.query,
        recall_k=data.recall_k,
        top_k=data.top_k,
        use_rerank=True,
        use_compression=False,   # 调试排序时不能压缩，否则内容被裁剪看不清楚
        score_threshold=0.0,     # 调试时不做阈值过滤，要看全貌
    )

    def to_items(pairs, with_vector_rank: bool = False):
        items = []
        # 建立 chunk_id → 向量阶段排名的映射，用于标注"它原本排第几"
        vector_rank_map = {
            doc.metadata.get("chunk_id"): idx + 1
            for idx, (doc, _) in enumerate(result.vector_hits)
        }
        for idx, (doc, score) in enumerate(pairs, start=1):
            meta = doc.metadata or {}
            chunk_id = str(meta.get("chunk_id", ""))
            items.append(
                RetrievedItem(
                    chunk_id=chunk_id,
                    doc_id=str(meta.get("doc_id", "")),
                    doc_title=str(meta.get("doc_title") or meta.get("source") or "未知文档"),
                    snippet=doc.page_content[:200],
                    score=round(float(score), 4),
                    rank=idx,
                    vector_rank=vector_rank_map.get(chunk_id) if with_vector_rank else None,
                )
            )
        return items

    return Result.success(
        data=RetrieveDebugOut(
            query=data.query,
            vector_hits=to_items(result.vector_hits[: data.top_k]),
            reranked=to_items(result.items, with_vector_rank=True),
            rerank_enabled=result.rerank_enabled,
            elapsed_ms=result.elapsed_ms,
        )
    )
