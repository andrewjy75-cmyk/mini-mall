"""相似度阈值标定：用数据决定 RETRIEVE_SCORE_THRESHOLD 该设多少。

★ 为什么需要这个脚本？

  这个阈值是幻觉治理里性价比最高的一个参数：它决定了"什么东西有资格
  进入 Prompt"。设得太松，一堆不相关的内容被塞给模型，模型就会硬凑答案；
  设得太紧，本来能答的问题也会因为召回被砍而答不出来。

  它【只能靠实测决定】，原因有两个：
    1. 不同 embedding 模型的相似度分布完全不同，照搬别人的数值没有意义。
       BGE 中文系列的分数普遍偏高，同一个问题换模型可能差 0.2。
    2. 它没有"理论最优值"，只有在你的知识库+你的问题上测出来的经验值。

★ 方法：找"可分区间隙"

  准备两组问题：
      A 组：知识库里【确实有答案】的问题 → 希望它们被召回（分数应该高）
      B 组：知识库里【根本没有答案】的问题 → 希望它们被拦掉（分数应该低）

  分别测出两组的最高分。如果 A 组最低分明显高于 B 组最高分，
  说明两者之间存在一道"空隙"——阈值就取在空隙里，这样两边都不误伤。

  如果两组分数重叠（A 组有些比 B 组还低），那说明单靠阈值无法区分，
  需要先改进切分或换 embedding 模型，而不是硬调阈值。

用法：
    .venv\\Scripts\\python.exe scripts\\probe_threshold.py
    .venv\\Scripts\\python\\python.exe scripts\\probe_threshold.py --csv   # 导出 CSV
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# A 组：知识库里确有答案。这些问题的答案分散在 data/sample 的四份文档里
ANSWERABLE = [
    "退货要几天到账",
    "生鲜食品可以退货吗",
    "新疆的运费是多少",
    "整机保修期多久",
    "金卡会员退货几天",
    "运费多少包邮",
    "定制商品能退吗",
    "主要部件保修几年",
    "积分怎么抵扣",
    "大促期间配送会延迟吗",
    "退款可以退到银行卡吗",
    "换货之后保修期怎么算",
]

# B 组：知识库里【根本没有】这些信息。理想情况下检索应该什么都召回不到，
# 模型也就只能老实回答"没查到"，而不是编一个出来
UNANSWERABLE = [
    "客服电话是多少",
    "实体店地址在哪里",
    "老板是谁",
    "你们公司什么时候成立",
    "有没有优惠券可以领",
    "怎么联系人工客服",
    "支持货到付款吗",
    "发票怎么开",
    "有没有分期免息",
    "可以用花呗吗",
    "怎么注销账号",
    "可以开发票吗",
]


def probe(question: str, top_k: int = 5) -> list[float]:
    """用阈值 0 检索，拿到未经任何过滤的原始分数。"""
    from rag.retriever import retrieve

    result = retrieve(question, use_rerank=False, use_compression=False,
                      top_k=top_k, score_threshold=0.0)
    return [score for _, score in result.vector_hits]


def main() -> None:
    parser = argparse.ArgumentParser(description="标定相似度阈值")
    parser.add_argument("--csv", action="store_true", help="额外导出 CSV 到 data/")
    args = parser.parse_args()

    print("=" * 78)
    print("  相似度阈值标定")
    print("=" * 78)
    print()

    rows: list[tuple[str, str, float]] = []

    def run(label: str, questions: list[str]) -> list[float]:
        print(f"----- {label} -----")
        tops: list[float] = []
        for q in questions:
            scores = probe(q)
            top = scores[0] if scores else 0.0
            tops.append(top)
            top3 = ", ".join(f"{s:.3f}" for s in scores[:3])
            print(f"  {q:22s}  最高 {top:.4f}   前3: {top3}")
            rows.append((label, q, top))
        print()
        return tops

    good = run("A 组：知识库里确实有答案（希望被召回）", ANSWERABLE)
    bad = run("B 组：知识库里没有答案（希望被拦掉）", UNANSWERABLE)

    good_min, good_max = min(good), max(good)
    bad_min, bad_max = min(bad), max(bad)

    print("=" * 78)
    print("  结论")
    print("=" * 78)
    print(f"  A 组（有答案）分数区间：{good_min:.4f} ~ {good_max:.4f}")
    print(f"  B 组（没答案）分数区间：{bad_min:.4f} ~ {bad_max:.4f}")
    print()

    if good_min > bad_max:
        # 有间隙：取中点作为建议阈值
        gap_low, gap_high = bad_max, good_min
        suggest = (gap_low + gap_high) / 2
        print(f"  ✓ 两组之间存在可分区间隙：{gap_low:.4f} ~ {gap_high:.4f}")
        print(f"    没有重叠，说明单靠相似度就能把'有答案'和'没答案'分开。")
        print()
        print(f"    建议阈值：{suggest:.2f}（取间隙中点，两边都不误伤）")
        print()
        # 顺便看看当前配置处在什么位置
        from core.config import settings

        current = settings.RETRIEVE_SCORE_THRESHOLD
        print(f"    当前配置：{current:.2f}", end="")
        if gap_low < current < gap_high:
            print("  ✓ 落在间隙内，是合理的")
        elif current <= gap_low:
            print(f"  ✗ 偏低（≤ {gap_low:.2f}），会有 {sum(1 for s in bad if s > current)} 个"
                  f"没有答案的问题漏过去，导致模型硬凑回答")
        else:
            print(f"  ✗ 偏高（≥ {gap_high:.2f}），会误杀 "
                  f"{sum(1 for s in good if s < current)} 个本来能答的问题")
    else:
        # 重叠：阈值解决不了，别硬调
        overlap = sum(1 for s in bad if s >= good_min)
        print(f"  ✗ 两组分数【重叠】了：有 {overlap} 个没有答案的问题，"
              f"得分比'有答案'中最低的那个还高")
        print()
        print("    这说明单靠阈值无法区分这两类问题，硬调只会顾此失彼：")
        print("      调高 → 误杀能答的问题；调低 → 放无关内容进来引发幻觉")
        print()
        print("    应该先去解决根因，按优先级：")
        print("      1. 检查切分是否把答案切散了（看 /docs/{id}/chunks）")
        print("      2. 换更强的 embedding 模型（如 bge-large-zh-v1.5）")
        print("      3. 开启检索重排，让精排来提升区分度")

    if args.csv:
        out = ROOT / "data" / "threshold_probe.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8-sig") as f:
            f.write("分组,问题,最高相似度\n")
            for label, q, score in rows:
                f.write(f"{label},{q},{score:.4f}\n")
        print()
        print(f"  明细已导出：{out}")

    print()


if __name__ == "__main__":
    main()
