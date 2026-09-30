"""实验：切分粒度对检索与重排效果的影响。

★ 这个实验的由来（很重要，它说明了一个容易被忽略的事实）：

  上线重排之后发现一个反常现象——重排确实跑起来了，但排序几乎没变化。
  打印分数才看明白：

      问题「生鲜食品可以退货吗」的三个候选块，重排后得分是：
          0.7268  /  0.7257  /  0.6295
      前两条只差 0.001。也就是说模型对它们的判断几乎一样，
      这个"排序"跟随机差不多。

  为什么区分不开？问题出在块的大小上。
  当时 chunk_size=600，而示例文档总共才 1000 多字，
  一个块里塞了 500 多字，其中和问题真正相关的可能就一句话。
  把这样一大段文字丢给重排模型，它看到的是"95% 的内容都无关"，
  信号被稀释掉了，自然给不出区分度。

  ★ 结论：重排不是插上就能用的。它的效果上限，
    取决于切分粒度给的"信号浓度"。块太大 → 重排失效。

  这个实验就是来量化这一点的：用同样的文档、同样的问题，
  只改变 chunk_size，看 top1 命中率怎么变。

用法：
    .venv\\Scripts\\python.exe scripts\\experiment_chunk.py
    .venv\\Scripts\\python\\python.exe scripts\\experiment_chunk.py --sizes 200 300 450 600
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 每个问题配一个"答案必须出现的关键词"。
# 用关键词而不是人工判分，是为了让这个实验可自动跑、可复现
CASES: list[tuple[str, list[str]]] = [
    ("退货要几天到账", ["3 个工作日"]),
    ("生鲜食品可以退货吗", ["生鲜食品、冷藏冷冻食品"]),
    ("新疆的运费是多少", ["15 元"]),
    ("整机保修期多久", ["整机保修：1 年"]),
    ("主要部件保修几年", ["主要部件", "2 年"]),
    ("金卡会员退货几天", ["15 个自然日"]),
    ("定制商品能退吗", ["定制类商品"]),
    ("退款可以退到银行卡吗", ["银行卡"]),
    ("换货之后保修期怎么算", ["重新计算"]),
    ("大促期间配送会延迟吗", ["顺延"]),
    ("积分怎么抵扣", ["100 积分可抵扣 1 元"]),
    ("钻石会员有什么特权", ["上门取件"]),
    ("未满包邮门槛的运费是多少", ["8 元"]),
    ("保修期内质量问题运费谁承担", ["平台承担"]),
]

SAMPLE_DIR = ROOT / "data" / "sample"


def build_index(chunk_size: int, chunk_overlap: int, persist_dir: Path, collection: str):
    """把示例文档按指定 chunk_size 切分并入库到临时向量库。"""
    from langchain_chroma import Chroma

    from rag.embedder import get_embeddings
    from rag.loader import load_document
    from rag.splitter import split_documents

    all_chunks = []
    for path in sorted(SAMPLE_DIR.iterdir()):
        if path.suffix.lower() not in {".pdf", ".docx", ".txt", ".md"}:
            continue
        docs = load_document(path)
        chunks = split_documents(
            docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            doc_id=path.stem, doc_title=path.stem,
        )
        all_chunks.extend(chunks)

    store = Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    # chunk_id 平时是在 service 层生成的（要同时写进 MySQL 当主键），
    # 这个实验不落库，所以自己组合一个稳定唯一的 id 就够了：
    # 同一份文档内 chunk_index 必然唯一
    ids = [f"{c.metadata['doc_id']}#{c.metadata['chunk_index']}" for c in all_chunks]
    store.add_documents(all_chunks, ids=ids)
    return store, all_chunks


def evaluate(store, rerank: bool) -> tuple[int, int]:
    """跑一遍问题集，统计 top1 块里是否包含答案关键词。"""
    from rag.reranker import rerank_documents

    hit = 0
    for question, keys in CASES:
        results = store.similarity_search_with_score(
            question, k=20, filter=None
        )
        candidates = [(doc, 1.0 - score) for doc, score in results]

        if rerank:
            ranked, _ = rerank_documents(
                question, [d for d, _ in candidates], top_k=3, enabled=True
            )
        else:
            ranked = candidates[:3]

        if ranked and any(k in ranked[0][0].page_content for k in keys):
            hit += 1
    return hit, len(CASES)


def main() -> None:
    parser = argparse.ArgumentParser(description="切分粒度对检索/重排效果的影响")
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 300, 450, 600],
                        help="要对比的 chunk_size 列表")
    parser.add_argument("--overlap", type=int, default=50, help="chunk_overlap")
    args = parser.parse_args()

    print("=" * 84)
    print("  实验：切分粒度 → 检索与重排效果")
    print("=" * 84)
    print(f"  文档：{SAMPLE_DIR}")
    print(f"  问题：{len(CASES)} 条，判定标准 = top1 片段是否包含答案关键词")
    print(f"  对比的 chunk_size：{args.sizes}（overlap={args.overlap}）")
    print()

    rows = []
    for size in args.sizes:
        tmp = Path(tempfile.mkdtemp(prefix=f"chunkexp_{size}_"))
        try:
            store, chunks = build_index(size, args.overlap, tmp, f"exp{size}")
            raw_hit, total = evaluate(store, rerank=False)
            rr_hit, _ = evaluate(store, rerank=True)
            avg_len = sum(len(c.page_content) for c in chunks) / len(chunks)

            rows.append((size, len(chunks), avg_len, raw_hit, rr_hit, total))
            print(f"  chunk_size={size:<4} 块数={len(chunks):<3} 平均块长={avg_len:5.0f}字  "
                  f"向量检索top1命中={raw_hit}/{total}  重排后top1命中={rr_hit}/{total}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("=" * 84)
    print("  汇总")
    print("=" * 84)
    print(f"  {'chunk_size':>10} {'块数':>6} {'平均块长':>10} {'向量top1':>10} {'重排top1':>10} {'重排增益':>10}")
    for size, n, avg_len, raw_hit, rr_hit, total in rows:
        gain = rr_hit - raw_hit
        gain_str = f"{gain:+d}" if gain else "0"
        print(f"  {size:>10} {n:>6} {avg_len:>10.0f} "
              f"{raw_hit}/{total:>7} {rr_hit}/{total:>7} {gain_str:>10}")

    print()
    best = max(rows, key=lambda r: (r[4], -r[2]))
    print(f"  → 重排效果最好的是 chunk_size={best[0]}"
          f"（重排 top1 命中 {best[4]}/{best[5]}，平均块长 {best[2]:.0f} 字）")
    print()
    print("  注意：块太小会切断完整条款，块太大则稀释重排信号。")
    print("  这个实验的意义不是找到'唯一正确值'，而是说明：")
    print("  重排的效果上限由切分粒度决定——切分没做好，重排再强也救不回来。")
    print()


if __name__ == "__main__":
    main()
