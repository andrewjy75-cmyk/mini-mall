"""RAG 效果评估脚本 —— 幻觉调优对比的执行引擎。

它做的事很简单：拿评估集里的每一道题去问系统，再用另一个大模型当裁判，
判断回答是对是错。然后把不同配置的结果横向摆在一起，形成对比表。

★ 为什么必须"用另一个大模型当裁判"（LLM-as-judge）？
  因为答案不是标准化的对错，而是自然语言。
  用关键词匹配去判分非常脆弱：「七天」和「7 天」要不要算一致？
  模型换个说法但意思正确，关键词就匹配不上，会误判成错。
  LLM 裁判能理解语义，判定要稳得多。
  代价是裁判本身也可能错，所以必须给它足够明确的标准（见 JUDGE_PROMPT），
  并且把裁判的理由一起存下来——理由可以人工抽查，判断裁判可不可信。

★ 关于「配置矩阵」的设计意图，这是本脚本最核心的价值：
  做对比实验最容易犯的错误是【一次改好几个变量】。
  那样即使最后效果好，也说不清是谁的功劳。
  所以这里的配置是逐层叠加的：

      v1-基线        ：关闭重排和压缩，只看 Prompt 的影响起点
      v2-防幻觉      ：只换 Prompt（相比 v1 的唯一变量）  ← 隔离 Prompt 的贡献
      v2+重排        ：只开重排（相比上一个的唯一变量）    ← 隔离重排的贡献
      v2+重排+压缩   ：只加压缩                            ← 隔离压缩的贡献
      v4-零温度      ：只把温度降到 0                      ← 隔离采样随机性的贡献

  每一步都只动一个变量，得到的结论才站得住。

用法：
    # 先启动服务（另开一个终端）：python main.py
    # 然后跑评估
    .venv\\Scripts\\python.exe eval\\run_eval.py
    .venv\\Scripts\\python.exe eval\\run_eval.py --only v1-基线 v2-防幻觉
    .venv\\Scripts\\python.exe eval\\run_eval.py --dry-run     # 只看评估集，不调接口
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from eval.dataset import EVAL_CASES, EvalCase, summary  # noqa: E402

DEFAULT_BASE_URL = "http://127.0.0.1:8600"
RESULT_DIR = ROOT / "eval" / "results"


# ===========================================================================
# 配置矩阵
# ===========================================================================


@dataclass
class EvalConfig:
    """一组待评估的服务端配置。

    name 会直接出现在最终的对比表里，所以要起得能看懂。
    """

    name: str
    prompt_version: str
    use_rerank: bool
    use_compression: bool
    note: str = ""


# 逐层叠加，每次只改一个变量（理由见模块 docstring）
CONFIGS: list[EvalConfig] = [
    EvalConfig("v1-基线", "v1", False, False,
               "对照组：朴素 Prompt，无重排无压缩"),
    EvalConfig("v2-防幻觉", "v2", False, False,
               "变量=仅换 Prompt（加我不知道的出口）"),
    EvalConfig("v3-强制引用", "v3", False, False,
               "变量=仅 Prompt 再加强制标注依据"),
    EvalConfig("v2+重排", "v2", True, False,
               "变量=仅开重排"),
    EvalConfig("v2+重排+压缩", "v2", True, True,
               "变量=仅开上下文压缩"),
    EvalConfig("v4-零温度", "v4", True, True,
               "变量=仅把 temperature 从 0.1 降到 0"),
]


# ===========================================================================
# 裁判
# ===========================================================================

JUDGE_PROMPT_ANSWERABLE = """你是一个严格的事实核查员。请判断客服的回答是否正确。

用户问题：{question}

回答必须覆盖的要点：
{key_points}

客服的实际回答：
{answer}

判定规则：
1. 逐条核对要点。要点必须【在回答中明确体现】，意思对就算，措辞不同没关系。
2. 如果回答里包含与要点【相矛盾】的信息，即使盖住了要点，也判为错误。
3. 如果回答只覆盖了部分要点，判为错误（不完整即错误）。
4. 如果回答说不知道、查不到，但要点确实在资料里，判为错误。

只输出一个 JSON 对象，不要有任何其他文字：
{{"correct": true 或 false, "covered": ["已覆盖的要点"], "missing": ["缺失的要点"], "reason": "一句话理由"}}"""

JUDGE_PROMPT_UNANSWERABLE = """你是一个严格的幻觉检测员。

背景：知识库中【没有】下面这个问题对应的任何信息。因此正确的客服回答应当是
明确表示无法提供该信息（例如说查不到、建议联系人工客服），
而【绝对不能】给出任何具体的、看起来像事实的信息。

用户问题：{question}

客服的实际回答：
{answer}

判定规则：
1. 如果回答给出了具体的电话号码、地址、金额、政策条款、活动内容等
   看起来像事实的具体信息 —— 这是【编造】，判 hallucinated=true。
2. 如果回答明确表示没有相关信息、查不到、建议联系人工核实 —— 判 hallucinated=false。
3. 如果回答只是泛泛而谈、复述用户问题、或者拒绝回答，但【没有】给出任何
   具体事实信息 —— 判 hallucinated=false（虽然不够好，但不算幻觉）。
4. 判断的关键是【有没有凭空给出具体信息】，而不是语气好不好。

只输出一个 JSON 对象，不要有任何其他文字：
{{"hallucinated": true 或 false, "fabricated": "编造的具体信息（没有则为空字符串）", "reason": "一句话理由"}}"""


@dataclass
class CaseResult:
    """单题的评估结果。"""

    case_id: str
    category: str
    question: str
    answer: str
    correct: bool
    reason: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    retrieved_count: int
    sources_count: int
    invalid_citations: list[int] = field(default_factory=list)
    error: str = ""


@dataclass
class ConfigResult:
    """一个配置跑完的汇总。"""

    config: EvalConfig
    cases: list[CaseResult] = field(default_factory=list)

    # ---- 指标计算 ----

    @property
    def total(self) -> int:
        return len(self.cases)

    def _by_category(self, category: str) -> list[CaseResult]:
        return [c for c in self.cases if c.category == category and not c.error]

    @property
    def answerable_accuracy(self) -> float:
        """可回答题的正确率。"""
        items = self._by_category("answerable")
        if not items:
            return 0.0
        return sum(1 for c in items if c.correct) / len(items)

    @property
    def trap_accuracy(self) -> float:
        """陷阱题正确率。"""
        items = self._by_category("trap")
        if not items:
            return 0.0
        return sum(1 for c in items if c.correct) / len(items)

    @property
    def refusal_rate(self) -> float:
        """不可回答题的【正确拒答率】。这个值越高越好。"""
        items = self._by_category("unanswerable")
        if not items:
            return 0.0
        return sum(1 for c in items if c.correct) / len(items)

    @property
    def hallucination_rate(self) -> float:
        """幻觉率 = 1 - 拒答率。

        ★ 之所以把这个指标单独拎出来作为标题指标，
          是因为它最贴近"模型有没有在编"这个本质问题，
          而且它的判定标准最客观：只有"有没有给出具体信息"这一个维度。
        """
        return 1.0 - self.refusal_rate

    @property
    def overall_accuracy(self) -> float:
        items = [c for c in self.cases if not c.error]
        if not items:
            return 0.0
        return sum(1 for c in items if c.correct) / len(items)

    @property
    def avg_latency_ms(self) -> int:
        items = [c for c in self.cases if not c.error]
        if not items:
            return 0
        return int(sum(c.latency_ms for c in items) / len(items))

    @property
    def avg_tokens(self) -> int:
        items = [c for c in self.cases if not c.error]
        if not items:
            return 0
        return int(sum(c.prompt_tokens + c.completion_tokens for c in items) / len(items))

    @property
    def invalid_citation_count(self) -> int:
        return sum(len(c.invalid_citations) for c in self.cases)

    @property
    def error_count(self) -> int:
        return sum(1 for c in self.cases if c.error)


# ===========================================================================
# 执行
# ===========================================================================


async def call_ask(
    client: httpx.AsyncClient,
    base_url: str,
    question: str,
    config: EvalConfig,
) -> dict:
    """调一次问答接口（非流式）。

    故意不传 session_id：每题一个独立会话。
    这样题目之间不会互相污染上下文——如果共用一个会话，
    第 5 题的答案可能受第 4 题的对话历史影响，评估结果就不可比了。
    """
    payload = {
        "question": question,
        "prompt_version": config.prompt_version,
        "use_rerank": config.use_rerank,
        "use_compression": config.use_compression,
    }
    resp = await client.post(f"{base_url}/api/chat/ask", json=payload, timeout=180)
    resp.raise_for_status()
    body = resp.json()
    if body.get("code") != 0:
        raise RuntimeError(body.get("message") or "接口返回失败")
    return body["data"]


async def judge(
    llm,
    case: EvalCase,
    answer: str,
) -> tuple[bool, str]:
    """用大模型判定这道题答得对不对。返回 (是否正确, 理由)。"""
    if case.answerable:
        prompt = JUDGE_PROMPT_ANSWERABLE.format(
            question=case.question,
            key_points="\n".join(f"- {p}" for p in case.key_points),
            answer=answer,
        )
    else:
        prompt = JUDGE_PROMPT_UNANSWERABLE.format(
            question=case.question, answer=answer
        )

    try:
        resp = await llm.ainvoke(prompt)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception as exc:  # noqa: BLE001
        return False, f"裁判调用失败：{exc}"

    # 裁判要求只输出 JSON，但现实里它常常会包一层 ```json 代码块，
    # 所以这里要容错地把 JSON 抠出来，而不是直接 json.loads 整个串
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                cleaned = part
                break
    elif not cleaned.startswith("{"):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return False, f"裁判输出无法解析：{text[:120]}"

    if case.answerable:
        return bool(data.get("correct")), str(data.get("reason", ""))
    # 不可回答的题：没有编造 = 正确
    return not bool(data.get("hallucinated")), str(data.get("reason", ""))


async def run_config(
    client: httpx.AsyncClient,
    judge_llm,
    base_url: str,
    config: EvalConfig,
    verbose: bool = True,
) -> ConfigResult:
    """跑完一个配置下的全部题目。"""
    result = ConfigResult(config=config)

    print(f"\n{'=' * 70}")
    print(f"  配置：{config.name}")
    print(f"  Prompt={config.prompt_version}  重排={config.use_rerank}  "
          f"压缩={config.use_compression}")
    print(f"  说明：{config.note}")
    print(f"{'=' * 70}")

    for index, case in enumerate(EVAL_CASES, start=1):
        prefix = f"  [{index:2d}/{len(EVAL_CASES)}] {case.id} {case.question[:26]}"
        try:
            data = await call_ask(client, base_url, case.question, config)
            answer = data.get("answer", "")
            correct, reason = await judge(judge_llm, case, answer)

            case_result = CaseResult(
                case_id=case.id,
                category=case.category,
                question=case.question,
                answer=answer,
                correct=correct,
                reason=reason,
                latency_ms=data.get("latency_ms", 0),
                prompt_tokens=data.get("prompt_tokens", 0),
                completion_tokens=data.get("completion_tokens", 0),
                retrieved_count=data.get("retrieved_count", 0),
                sources_count=len(data.get("sources") or []),
            )
            mark = "✓" if correct else "✗"
            if verbose:
                print(f"{prefix}  {mark}  {reason[:44]}")
        except Exception as exc:  # noqa: BLE001
            case_result = CaseResult(
                case_id=case.id, category=case.category, question=case.question,
                answer="", correct=False, reason="", latency_ms=0,
                prompt_tokens=0, completion_tokens=0,
                retrieved_count=0, sources_count=0, error=str(exc),
            )
            print(f"{prefix}  ! 调用失败：{exc}")

        result.cases.append(case_result)

    return result


def print_comparison(results: list[ConfigResult]) -> str:
    """打印对比表，并返回 Markdown 版本（直接可以贴进文档）。"""
    headers = ["配置", "可回答正确率", "陷阱题正确率", "幻觉率↓", "拒答率↑",
               "平均延迟", "平均tokens"]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for r in results:
        lines.append(
            "| " + " | ".join([
                r.config.name,
                f"{r.answerable_accuracy * 100:.1f}%",
                f"{r.trap_accuracy * 100:.1f}%",
                f"{r.hallucination_rate * 100:.1f}%",
                f"{r.refusal_rate * 100:.1f}%",
                f"{r.avg_latency_ms} ms",
                f"{r.avg_tokens}",
            ]) + " |"
        )

    table = "\n".join(lines)
    print(f"\n{'=' * 70}")
    print("  对比结果")
    print(f"{'=' * 70}")
    print(table)
    return table


async def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 效果评估 / 幻觉调优对比")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="服务地址")
    parser.add_argument("--only", nargs="*", default=None,
                        help="只跑指定的配置名（可多个），默认跑全部")
    parser.add_argument("--dry-run", action="store_true", help="只打印评估集，不调接口")
    parser.add_argument("--tag", default="", help="给这次结果加个标记，用于文件名")
    args = parser.parse_args()

    info = summary()
    print("评估集概况：")
    print(f"  总计 {info['total']} 题 —— 可回答 {info['answerable']} / "
          f"不可回答 {info['unanswerable']} / 陷阱 {info['trap']}")

    if args.dry_run:
        print("\n[dry-run] 只列题目，不调接口：")
        for c in EVAL_CASES:
            flag = "可答" if c.answerable else "不可答"
            print(f"  {c.id} [{flag}] {c.question}")
        return

    configs = CONFIGS
    if args.only:
        configs = [c for c in CONFIGS if c.name in args.only]
        if not configs:
            print(f"没有匹配的配置名。可用：{[c.name for c in CONFIGS]}")
            return

    # 裁判用零温度：判分要求稳定，同样的输入应该给同样的结论
    from rag.llm import get_llm, LLMNotConfiguredError

    try:
        judge_llm = get_llm(temperature=0.0)
    except LLMNotConfiguredError as exc:
        print(f"\n无法启动评估：{exc}")
        return

    # 先探一下服务是否活着。用一个明确的错误提示，好过让用户看 21 个连接错误
    # 注意这里用的是 /api/hello，它在 hello_api.py 里定义为 GET /hello
    async with httpx.AsyncClient() as probe:
        try:
            r = await probe.get(f"{args.base_url}/api/hello", timeout=5)
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"\n连不上服务 {args.base_url}：{exc}")
            print("请先在另一个终端启动服务：.venv\\Scripts\\python.exe main.py")
            return

    print(f"\n服务地址：{args.base_url}")
    print(f"待评估配置：{len(configs)} 组 × {len(EVAL_CASES)} 题 "
          f"= 约 {len(configs) * len(EVAL_CASES) * 2} 次大模型调用")
    print("（每次调用约 1~5 秒，整体可能需要几分钟，请耐心等待）")

    started = time.time()
    results: list[ConfigResult] = []

    async with httpx.AsyncClient() as client:
        for config in configs:
            results.append(
                await run_config(client, judge_llm, args.base_url, config)
            )

    table = print_comparison(results)

    # ---- 落盘 ----
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"-{args.tag}" if args.tag else ""
    out_path = RESULT_DIR / f"eval{tag}-{stamp}.json"

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": args.base_url,
        "elapsed_seconds": int(time.time() - started),
        "comparison_table_markdown": table,
        "configs": [
            {
                "config": asdict(r.config),
                "metrics": {
                    "answerable_accuracy": round(r.answerable_accuracy, 4),
                    "trap_accuracy": round(r.trap_accuracy, 4),
                    "refusal_rate": round(r.refusal_rate, 4),
                    "hallucination_rate": round(r.hallucination_rate, 4),
                    "overall_accuracy": round(r.overall_accuracy, 4),
                    "avg_latency_ms": r.avg_latency_ms,
                    "avg_tokens": r.avg_tokens,
                    "invalid_citations": r.invalid_citation_count,
                    "errors": r.error_count,
                },
                "cases": [asdict(c) for c in r.cases],
            }
            for r in results
        ],
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n结果已保存：{out_path}")
    print(f"总耗时：{int(time.time() - started)} 秒")
    print("\n提示：把上面那张 Markdown 表格直接贴进优化文档即可。"
          "逐题明细（含模型原文与裁判理由）在同名 JSON 里。")


if __name__ == "__main__":
    asyncio.run(main())
