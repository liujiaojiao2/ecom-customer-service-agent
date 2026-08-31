"""从阶段0 测试集分层抽 20 条，产出配图清单：
- 每场景 5 条：calm 2 / impatient 2 / angry 1（其中含 1 条过度索赔）
- 为每条生成"你需要一张什么图"的自然语言提示，方便用户手工搜集
- 产出：data/mm_samples/samples.json（含金标签 + 期望图片描述 + 行为流）
- 图片文件由用户放入 data/mm_samples/images/{case_id}.jpg
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = json.loads((ROOT / "data" / "test_cases.json").read_text(encoding="utf-8"))
OUT_DIR = ROOT / "data" / "mm_samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "images").mkdir(exist_ok=True)

IMAGE_HINT = {
    "refund": {
        "shipped_false": "商品订单详情页截图（显示'未发货'状态）",
    },
    "return": {
        "default": "商品有明显质量问题/破损的实物照片（如衣物掉色、破损包装、瑕疵商品）",
    },
    "exchange": {
        "default": "错发或与描述不符的商品实物照片（如型号错误、颜色不符、尺码问题）",
    },
    "logistics": {
        "default": "物流面单或快递轨迹截图（显示长期未更新/异常状态）",
    },
}

BEHAVIOR_TMPL = {
    "refund": ["查看订单详情", "长按订单-点击'申请退款'", "咨询客服"],
    "return": ["查看订单详情", "点击'申请售后'", "选择'退货退款'", "上传照片"],
    "exchange": ["查看订单详情", "点击'申请售后'", "选择'换货'", "上传照片"],
    "logistics": ["查看物流轨迹", "点击'催发货'/'联系快递'", "咨询客服"],
}


def pick():
    rng = random.Random(42)
    picks = []
    for scenario in ["refund", "return", "exchange", "logistics"]:
        subset = [c for c in CASES if c["issue_type"] == scenario]
        calms = [c for c in subset if c["emotion"] == "calm"]
        impatients = [c for c in subset if c["emotion"] == "impatient"]
        angrys = [c for c in subset if c["emotion"] == "angry"]
        excessive = [c for c in angrys if c["demand_excessive"]]

        chosen = (rng.sample(calms, 2) + rng.sample(impatients, 2) +
                  rng.sample(excessive, 1))
        picks.extend(chosen)
    return picks


def build_sample(c: dict) -> dict:
    scenario = c["issue_type"]
    if scenario == "refund" and not c["order"]["shipped"]:
        image_hint = IMAGE_HINT["refund"]["shipped_false"]
    else:
        image_hint = IMAGE_HINT[scenario]["default"]
    return {
        "case_id": c["case_id"],
        "image_path": f"data/mm_samples/images/{c['case_id']}.jpg",
        "image_hint": image_hint,
        "product": c["order"]["product"],
        "first_utterance": c["first_utterance"],
        "behaviors": BEHAVIOR_TMPL[scenario],
        "gold": {
            "issue_type": scenario,
            "has_evidence": scenario in {"return", "exchange"},
        },
    }


def main():
    picks = pick()
    samples = [build_sample(c) for c in picks]
    out = OUT_DIR / "samples.json"
    out.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"20 条样本 → {out}\n")

    print("===== 配图清单（请为每条 case 放一张图到 data/mm_samples/images/{case_id}.jpg）=====\n")
    for s in samples:
        print(f"[{s['case_id']}]  商品: {s['product']}")
        print(f"  需要一张: {s['image_hint']}")
        print(f"  首句: {s['first_utterance']}\n")


if __name__ == "__main__":
    main()
