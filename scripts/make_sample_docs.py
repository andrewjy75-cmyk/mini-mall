"""生成演示用的知识库文档（电商客服场景）。

为什么要有这个脚本？
1. 演示时手上得有份像样的资料。临场找 PDF 很容易翻车：
   找到的可能是扫描件（提不出文字），或者内容和问题对不上。
2. 考核要求明确提到"PDF/Word 文件解析"。要证明它真的能用，
   就得有真文件跑一遍，只写"支持 PDF"是没有说服力的。
3. 四种格式各生成一份，正好把每个解析分支都覆盖到：
       PDF  → 测 pypdf 提取和"视觉换行"拼接
       DOCX → 测段落顺序 + 表格转 Markdown
       TXT  → 测编码识别
       MD   → 测 Markdown 结构保留

用法：
    .venv\\Scripts\\python.exe scripts\\make_sample_docs.py

输出到 data/sample/ 目录。

★ 文档内容是刻意设计的，每份都写到 1500~2000 字，目的是：
  让 chunk_size=600 能切出 3~4 块，四份文档合计 14 块以上。
  如果每份文档只有几百字（切成 2 块），检索结果的区分度会非常低，
  重排也看不出效果——演示时会显得"这个东西没什么用"。

  内容里还埋了这些"坑"，用于演示和评估：

  1. 退换货政策说"7 天无理由"，会员权益说"金卡会员延长到 15 天" —— 跨文档交叉信息
  2. 生鲜/定制/拆封数码有除外条款 —— 容易被模型忽略的限定条件
  3. 运费文档里有表格 —— 测表格解析，也测模型能否读懂行列关系
  4. 多份文档都反复出现"退货""运费""保修"这些词 —— 制造检索干扰，
     真正的答案需要靠重排和上下文区分度才能被顶上来
  5. 有一批问题文档里【根本没有答案】（比如实体店地址、客服电话）——
     专门用来测模型会不会硬编答案。这是衡量幻觉最直接的场景。
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

# 让脚本能直接跑（把项目根目录加进 sys.path），否则 import 不到项目模块
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "sample"

# Windows 上可用的中文字体。reportlab 生成中文 PDF 必须嵌入字体，
# 否则显示出来全是黑方块
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\Deng.ttf"),
]


@dataclass
class Block:
    """文档里的一个块。用统一结构描述，再渲染成不同格式。"""

    kind: str                       # h1 / h2 / p / ul / table
    text: str = ""
    items: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)


# ===========================================================================
# 文档内容
# ===========================================================================

RETURN_POLICY: list[Block] = [
    Block("h1", "Mini Mall 退换货政策"),
    Block("p", "文档版本：V3.2　　生效日期：2026 年 1 月 1 日"),
    Block("p", "本政策适用于 Mini Mall 商城自营商品。第三方商家商品的退换货规则以其店铺公示为准。"),

    Block("h2", "一、七天无理由退货"),
    Block("p", "自商品签收之日起 7 个自然日内，消费者在保证商品完好、不影响二次销售的前提下，可申请无理由退货。"),
    Block("p", "「商品完好」的具体标准包括：商品本身、配件、吊牌、标识、包装完整，无使用痕迹。各类商品的细化标准如下："),
    Block("ul", items=[
        "服装鞋帽类：吊牌未剪、未洗涤、无穿着痕迹，鞋底无磨损",
        "数码产品类：包装未拆封，或拆封后未激活、未使用、未注册保修",
        "家居用品类：未安装、未使用，原包装完整，配件齐全",
        "美妆个护类：外包装完整，未拆封、未试用",
        "图书音像类：无笔记、无划线、无破损，塑封未拆",
    ]),
    Block("p", "「7 个自然日」的起算时间是快递物流显示签收的当日，而非下单日或发货日。签收日期以物流轨迹记录为准。"),
    Block("p", "若因个人原因导致商品不完好（如已剪吊牌、已使用、包装损坏），商家有权拒绝退货申请。"),

    Block("h2", "二、不适用无理由退货的商品"),
    Block("p", "根据《消费者权益保护法》第二十五条规定，以下商品不适用七天无理由退货："),
    Block("ul", items=[
        "生鲜食品、冷藏冷冻食品、短保质期食品",
        "定制类商品，包括刻字、改尺寸、专属配色的商品",
        "贴身衣物、内衣、袜子等个人卫生用品（已拆封）",
        "已拆封的数码产品、软件、音像制品",
        "一经激活或试用后价值贬损较大的商品",
        "在线下载或消费者拆封的数字化商品",
    ]),
    Block("p", "特别提示：定制类商品一经下单即进入生产流程，不支持任何形式的退货，请下单前仔细确认尺寸与款式。"),
    Block("p", "上述除外条款适用于全部会员等级。会员权益可在其他方面提供便利，但不改变法定不适用无理由退货的商品范围。"),

    Block("h2", "三、质量问题退换货"),
    Block("p", "商品存在质量问题的，不受 7 天限制。具体规则如下："),
    Block("ul", items=[
        "自签收之日起 15 个自然日内，可申请换货",
        "自签收之日起 30 个自然日内，可申请退货",
        "超过 30 个自然日的，按商品保修政策处理，安排维修",
    ]),
    Block("p", "商品是否属于质量问题，以平台或第三方检测机构的鉴定结论为准。"),
    Block("p", "因质量问题产生的往返运费由 Mini Mall 承担，用户无需支付任何物流费用。"),
    Block("p", "注意区分两类规则：「质量问题退换货」与「七天无理由退货」是相互独立的条款。"
              "定制类商品虽然不支持无理由退货，但若存在质量问题，仍然适用本条的 15 天换货、30 天退货规定。"),

    Block("h2", "四、退款时效"),
    Block("p", "退货商品经商家签收并验收合格后，退款将在 3 个工作日内按原支付路径退回。"),
    Block("p", "若原支付渠道已失效（如银行卡注销、账户关闭），退款将转为平台账户余额，可用于平台内消费，也可申请提现。"),
    Block("p", "实际到账时间受支付渠道处理速度影响，参考时效如下："),
    Block("ul", items=[
        "微信支付：1 至 2 个工作日",
        "支付宝：1 至 2 个工作日",
        "银行卡：3 至 7 个工作日",
        "平台余额：即时到账",
    ]),
    Block("p", "退款金额按实付金额计算。已使用的优惠券不予退还，但若该优惠券仍在有效期内，将原路返还至用户账户。"),

    Block("h2", "五、申请流程"),
    Block("p", "1. 进入「我的订单」页面，找到对应订单，点击「申请售后」。"),
    Block("p", "2. 选择服务类型（退货 / 换货 / 维修），填写原因并上传商品照片或视频。"),
    Block("p", "3. 客服在 1 个工作日内完成审核。审核通过后，系统生成退货地址与物流单号。"),
    Block("p", "4. 用户按提示寄回商品，务必填写正确的物流单号，否则无法跟踪。"),
    Block("p", "5. 商家签收并验收后，系统自动发起退款或换货流程。"),

    Block("h2", "六、常见问题"),
    Block("p", "问：退货后运费会退吗？答：无理由退货的寄回运费由用户承担；质量问题退货的往返运费由平台承担。"),
    Block("p", "问：可以只退订单中的一件商品吗？答：可以。订单支持部分退货，退款金额按该商品实付金额计算。"),
    Block("p", "问：赠品需要一并退回吗？答：需要。主商品退货时赠品须一并寄回，未寄回将从退款中扣除赠品价值。"),
]

SHIPPING: list[Block] = [
    Block("h1", "Mini Mall 运费与配送说明"),
    Block("p", "本文档说明订单的运费计算方式、配送范围与时效。适用对象为平台自营商品。"),

    Block("h2", "一、包邮门槛"),
    Block("p", "单笔订单实付金额满 99 元即可享受全国包邮（偏远地区除外）。"),
    Block("p", "未满 99 元的订单，统一收取基础运费 8 元。"),
    Block("p", "实付金额按扣除优惠券、积分抵扣后的金额计算，不含运费本身。"),
    Block("p", "多笔订单合并支付时，若合并后金额达到包邮门槛，同样享受包邮。"),

    Block("h2", "二、偏远地区运费"),
    Block("p", "以下地区因物流成本较高，需在基础运费基础上加收偏远地区附加费："),
    Block("table", rows=[
        ["地区", "附加费", "包邮门槛", "预计时效"],
        ["新疆", "15 元", "满 199 元", "5-8 个工作日"],
        ["西藏", "15 元", "满 199 元", "6-10 个工作日"],
        ["内蒙古", "10 元", "满 149 元", "4-6 个工作日"],
        ["青海、宁夏", "10 元", "满 149 元", "4-6 个工作日"],
        ["甘肃、云南、贵州", "8 元", "满 129 元", "4-6 个工作日"],
        ["港澳台及海外", "暂不支持配送", "—", "—"],
    ]),
    Block("p", "偏远地区的包邮门槛高于全国标准。例如寄往新疆的订单，需实付满 199 元才免运费，"
              "未满 199 元的订单需支付基础运费 8 元加附加费 15 元，合计 23 元。"),

    Block("h2", "三、配送方式与时效"),
    Block("p", "订单在付款后 24 小时内出库（预售商品除外）。出库后按下表时效送达："),
    Block("table", rows=[
        ["配送方式", "覆盖范围", "时效", "费用"],
        ["标准快递", "全国大部分地区", "2-4 个工作日", "免费或 8 元"],
        ["顺丰特快", "省会及地级市", "1-2 个工作日", "加收 12 元"],
        ["次日达", "部分一线城市", "次日 18 点前", "加收 20 元"],
        ["自提点自提", "已开通城市", "出库后即可自提", "免费"],
    ]),
    Block("p", "时效自快递公司揽收次日起算，不含下单当日与节假日。"),
    Block("p", "大件商品（如家具、家电）由专线物流配送，时效与普通快递不同，下单时会单独提示。"),

    Block("h2", "四、特殊时段说明"),
    Block("p", "大促期间（如双十一、618、年货节）物流压力较大，配送时效可能顺延 2 至 3 个工作日。"),
    Block("p", "法定节假日期间快递公司会调整揽收时间，具体以物流页面显示为准。"),
    Block("p", "如遇不可抗力（极端天气、自然灾害、疫情管控等）导致配送延迟，我们将通过短信通知，"
              "并可在订单页面申请延迟赔付。"),

    Block("h2", "五、签收与异常处理"),
    Block("p", "建议当面验货后再签收。若发现外包装破损、商品缺失或损坏，可拒收并联系客服。"),
    Block("p", "包裹显示已签收但实际未收到的，请在 48 小时内联系客服，我们将向快递公司发起查询。"),
    Block("p", "包裹在运输途中丢失或损毁的，由平台先行赔付，用户无需承担损失。"),
    Block("p", "因用户提供地址错误、电话无法接通导致退回的，重新发货需重新计算运费。"),

    Block("h2", "六、运费与退货的关系"),
    Block("p", "无理由退货的寄回运费由用户自行承担。若平台提供运费险且该订单已投保，可申请理赔。"),
    Block("p", "因质量问题、商品损坏、发错货导致的退换货，往返运费由平台全额承担。"),
    Block("p", "退货运费的计算与配送费用无关，按实际寄回的快递费用结算。"),
]

WARRANTY: list[Block] = [
    Block("h1", "Mini Mall 商品保修与售后服务"),
    Block("p", ""),
    Block("p", "本文档适用于平台自营商品。第三方商家商品的保修政策以其店铺公示为准。"),

    Block("h2", "第一条 保修期限"),
    Block("p", "自商品签收之日起计算，各类商品的保修期限如下："),
    Block("ul", items=[
        "整机保修：1 年（自签收之日起）",
        "主要部件（电机、压缩机、主板、显示屏等）：2 年",
        "耗材类（滤芯、刷头、电池、刀头等）：6 个月",
        "随机附件（数据线、充电器、遥控器等）：6 个月",
        "赠品：不单独提供保修服务，随主商品享受同等保修期",
    ]),
    Block("p", "「整机保修」与「主要部件保修」是两个独立的期限。例如一台洗衣机，"
              "整机保修 1 年，但其中的电机享受 2 年保修。第 2 年内若电机故障，仍可免费维修；"
              "但若是外壳、按键等非主要部件故障，超出 1 年后需自费。"),

    Block("h2", "第二条 保修范围"),
    Block("p", "在保修期内，因产品本身质量问题导致的故障，提供免费维修服务，包括工时费与配件费。"),
    Block("p", "以下情况不属于免费保修范围，需收取材料费与人工费："),
    Block("ul", items=[
        "人为损坏，包括摔落、进水、挤压、私自拆机",
        "未按说明书要求使用或保养造成的损坏",
        "超出保修期限的商品",
        "因不可抗力（火灾、水灾、雷击等）造成的损坏",
        "无法提供有效购买凭证的商品",
        "非平台或非授权服务商拆修过的商品",
    ]),

    Block("h2", "第三条 维修流程"),
    Block("p", "1. 在「我的订单」中提交维修申请，并描述故障现象，建议附上照片或视频。"),
    Block("p", "2. 客服在 1 个工作日内审核，审核通过后生成寄修地址或安排上门服务。"),
    Block("p", "3. 用户将商品寄回（保修期内质量问题，往返运费由平台承担）。"),
    Block("p", "4. 维修中心检测并维修，一般 5 至 7 个工作日完成，复杂故障不超过 15 个工作日。"),
    Block("p", "5. 维修完成后寄回，用户签收后该维修部件的保修期顺延 90 天。"),
    Block("p", "大件家电（冰箱、洗衣机、空调等）支持上门维修，无需寄回。"),

    Block("h2", "第四条 退换与保修的关系"),
    Block("p", "退货后，该订单的保修服务自动终止。"),
    Block("p", "换货后，新商品的保修期自新商品签收之日起重新计算，不沿用原商品的剩余期限。"),
    Block("p", "这意味着换货不会让用户损失保修时间，反而重新获得完整的新品保修期。"),
    Block("p", "部分换货（仅更换故障部件）的，仅该部件保修期重新计算，整机保修期不变。"),

    Block("h2", "第五条 延长保修服务"),
    Block("p", "用户在购买商品时可选购延保服务，将整机保修期延长 1 至 3 年。"),
    Block("p", "延保服务需在商品签收后 30 天内购买，超期不可补购。"),
    Block("p", "延保服务的保修范围与原厂保修一致，同样不包含人为损坏。"),

    Block("h2", "第六条 争议处理"),
    Block("p", "如对维修结果有异议，可在收到商品后 7 个工作日内申请复检。"),
    Block("p", "复检由双方认可的第三方检测机构执行，检测费用由责任方承担。"),
    Block("p", "复检结论为商品确存在质量问题的，平台承担全部检测费用并重新维修。"),
]

MEMBERSHIP: list[Block] = [
    Block("h1", "Mini Mall 会员等级与权益说明"),
    Block("p", "本说明介绍会员等级的评定方式、各级权益，以及积分规则。"),

    Block("h2", "一、会员等级"),
    Block("p", "会员等级根据近 12 个月内的累计消费金额自动评定，每月 1 日更新。"),
    Block("table", rows=[
        ["等级", "累计消费", "商品折扣", "积分倍率", "专属客服"],
        ["普通会员", "0 元", "无", "1 倍", "否"],
        ["银卡会员", "满 1000 元", "98 折", "1.5 倍", "否"],
        ["金卡会员", "满 5000 元", "95 折", "2 倍", "是"],
        ["钻石会员", "满 20000 元", "92 折", "3 倍", "是"],
    ]),
    Block("p", "累计消费金额按实付金额计算，不含运费与使用积分抵扣的部分。"),
    Block("p", "退款订单会从累计消费中扣除。例如购买 6000 元商品升为金卡后发生全额退款，"
              "累计消费回落至门槛以下，将在次月 1 日降级。"),

    Block("h2", "二、退货权益"),
    Block("p", "金卡及以上等级会员，七天无理由退货期限延长至 15 个自然日。"),
    Block("p", "该延长权益仅适用于【原本支持无理由退货】的普通商品。"),
    Block("p", "生鲜食品、定制商品、已拆封数码产品等法定不适用无理由退货的商品，"
              "会员权益亦不适用，一律不支持退货。"),
    Block("p", "钻石会员额外享受每年 3 次免费上门取件退货服务，由平台安排快递上门，"
              "无需用户自行寄回，也无需垫付运费。"),
    Block("p", "银卡及普通会员的退货期限与平台标准一致，均为 7 个自然日。"),

    Block("h2", "三、积分规则"),
    Block("p", "积分获取：每消费 1 元累计 1 积分，按会员等级乘以对应倍率。"),
    Block("p", "例如金卡会员（2 倍）消费 100 元，可获得 200 积分。"),
    Block("p", "积分有效期：自获得之日起 24 个月，过期自动清零，不可恢复、不可延期。"),
    Block("p", "积分使用：100 积分可抵扣 1 元，单笔订单最多抵扣订单金额的 30%。"),
    Block("p", "以下情形不累计积分：使用积分抵扣的部分、运费、已退款订单、礼品卡支付部分。"),

    Block("h2", "四、会员专享活动"),
    Block("p", "每月 8 日为会员日，金卡及以上会员可领取专属优惠券。"),
    Block("p", "钻石会员可提前 24 小时参与新品首发与限时抢购。"),
    Block("p", "会员积分可在积分商城兑换商品、优惠券或运费券，兑换后积分即时扣除。"),

    Block("h2", "五、会员降级与查询"),
    Block("p", "若近 12 个月累计消费未达到当前等级门槛，将在次月 1 日降级至对应等级。"),
    Block("p", "降级后，已获得的积分不受影响，但折扣与退货权益将同步调整。"),
    Block("p", "用户可在「我的-会员中心」查看当前等级、累计消费金额、升级所需金额与积分余额。"),
    Block("p", "如对等级评定有异议，可提交申诉，客服将在 3 个工作日内核实答复。"),
]


# ===========================================================================
# 渲染器
# ===========================================================================


def _get_font() -> Path:
    for path in FONT_CANDIDATES:
        if path.exists():
            return path
    raise SystemExit(
        "找不到可用的中文字体（尝试过 simhei.ttf / Deng.ttf）。\n"
        "生成 PDF 必须嵌入中文字体。你可以跳过 PDF，"
        "或者手动修改 FONT_CANDIDATES 指向一个 .ttf 字体文件。"
    )


def _table_to_markdown(rows: list[list[str]]) -> str:
    """把二维数据转成 Markdown 表格（txt/md 格式里用它保持结构）。"""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    lines.extend("| " + " | ".join(r) + " |" for r in rows[1:])
    return "\n".join(lines)


def to_markdown(blocks: list[Block]) -> str:
    out: list[str] = []
    for b in blocks:
        if b.kind == "h1":
            out.append(f"# {b.text}\n")
        elif b.kind == "h2":
            out.append(f"## {b.text}\n")
        elif b.kind == "p":
            if b.text:
                out.append(b.text + "\n")
        elif b.kind == "ul":
            out.extend(f"- {item}" for item in b.items)
            out.append("")
        elif b.kind == "table":
            out.append(_table_to_markdown(b.rows) + "\n")
    return "\n".join(out)


def to_text(blocks: list[Block]) -> str:
    """纯文本版：不带 Markdown 记号，模拟"从别处导出的 txt"。"""
    out: list[str] = []
    for b in blocks:
        if b.kind == "h1":
            out.append(b.text + "\n" + "=" * 40 + "\n")
        elif b.kind == "h2":
            out.append("\n" + b.text + "\n")
        elif b.kind == "p":
            if b.text:
                out.append(b.text + "\n")
        elif b.kind == "ul":
            out.extend(f"  · {item}" for item in b.items)
            out.append("")
        elif b.kind == "table":
            # txt 里没有表格概念，用制表符分隔——这也顺便测试解析器
            # 面对"非 Markdown 表格"时的表现
            for row in b.rows:
                out.append("\t".join(row))
            out.append("")
    return "\n".join(out)


def write_txt(blocks: list[Block], path: Path) -> None:
    path.write_text(to_text(blocks), encoding="utf-8")
    print(f"  [txt ] {path.name}")


def write_markdown(blocks: list[Block], path: Path) -> None:
    path.write_text(to_markdown(blocks), encoding="utf-8")
    print(f"  [md  ] {path.name}")


def write_docx(blocks: list[Block], path: Path) -> None:
    """生成 Word 文档。

    ★ 这里刻意把表格生成成【真正的 Word 表格】（而不是用文本模拟），
      因为这样才能验证 python-docx 那一段表格解析代码是否真的работает。
      用文本假装表格的话，测不出"doc.paragraphs 里没有表格文字"这个坑。
    """
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    document = docx.Document()

    # 设置中文字体。python-docx 默认字体不含中文，
    # 必须同时设置 ascii/hAnsi 和 eastAsia 三处，否则中文会显示成方框
    style = document.styles["Normal"]
    style.font.name = "微软雅黑"
    style.font.size = Pt(10.5)
    from docx.oxml.ns import qn

    style.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")

    for b in blocks:
        if b.kind == "h1":
            document.add_heading(b.text, level=1)
        elif b.kind == "h2":
            document.add_heading(b.text, level=2)
        elif b.kind == "p":
            if b.text:
                document.add_paragraph(b.text)
        elif b.kind == "ul":
            for item in b.items:
                p = document.add_paragraph(item, style="List Bullet")
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        elif b.kind == "table":
            if not b.rows:
                continue
            table = document.add_table(rows=len(b.rows), cols=len(b.rows[0]))
            table.style = "Table Grid"
            for r, row in enumerate(b.rows):
                for c, cell in enumerate(row):
                    table.cell(r, c).text = str(cell)

    document.save(str(path))
    print(f"  [docx] {path.name}")


def write_pdf(blocks: list[Block], path: Path) -> None:
    """生成 PDF。中文必须嵌入 TTF 字体，否则全是黑方块。"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font_path = _get_font()
    pdfmetrics.registerFont(TTFont("CN", str(font_path)))

    styles = {
        "h1": ParagraphStyle("h1", fontName="CN", fontSize=17, leading=24, spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName="CN", fontSize=13, leading=19,
                             spaceBefore=12, spaceAfter=6),
        "p": ParagraphStyle("p", fontName="CN", fontSize=10.5, leading=17, spaceAfter=5),
        "li": ParagraphStyle("li", fontName="CN", fontSize=10.5, leading=17,
                             leftIndent=14, spaceAfter=3),
    }

    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm, topMargin=20 * mm, bottomMargin=20 * mm,
        title=blocks[0].text if blocks else "document",
    )

    flow = []
    for b in blocks:
        if b.kind == "h1":
            flow.append(Paragraph(b.text, styles["h1"]))
        elif b.kind == "h2":
            flow.append(Paragraph(b.text, styles["h2"]))
        elif b.kind == "p":
            if b.text:
                flow.append(Paragraph(b.text, styles["p"]))
        elif b.kind == "ul":
            for item in b.items:
                flow.append(Paragraph(f"· {item}", styles["li"]))
        elif b.kind == "table":
            if not b.rows:
                continue
            data = [[Paragraph(str(c), styles["p"]) for c in row] for row in b.rows]
            table = Table(data, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, (0.6, 0.6, 0.6)),
                ("BACKGROUND", (0, 0), (-1, 0), (0.93, 0.95, 0.98)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            flow.append(table)
            flow.append(Spacer(1, 6))

    doc.build(flow)
    print(f"  [pdf ] {path.name}")


# ===========================================================================
# 主流程
# ===========================================================================


def main() -> None:
    try:
        import reportlab  # noqa: F401
    except ImportError:
        print("缺少 reportlab，无法生成 PDF。请先执行：")
        print("    .venv\\Scripts\\python.exe -m pip install reportlab")
        raise SystemExit(1)

    try:
        import docx  # noqa: F401
    except ImportError:
        print("缺少 python-docx，无法生成 Word。请先执行：")
        print("    .venv\\Scripts\\python.exe -m pip install python-docx")
        raise SystemExit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"正在生成演示文档到：{OUT_DIR}\n")

    docs = [
        ("退换货政策.pdf", RETURN_POLICY, write_pdf),
        ("运费与配送说明.docx", SHIPPING, write_docx),
        ("商品保修与售后服务.txt", WARRANTY, write_txt),
        ("会员等级与权益.md", MEMBERSHIP, write_markdown),
    ]

    for name, blocks, writer in docs:
        writer(blocks, OUT_DIR / name)
        chars = sum(len(b.text) + sum(len(i) for i in b.items)
                    + sum(len(c) for r in b.rows for c in r) for b in blocks)
        print(f"         约 {chars} 字")

    print("\n完成。可以在聊天页面依次上传这 4 个文件"
          "（覆盖 PDF / Word / TXT / MD 四种格式）。")


if __name__ == "__main__":
    main()
