"""合成测试集生成：骨架确定性构造保证分布均衡，LLM 填充自然语言内容。"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import llm

OUT_PATH = ROOT / "data" / "test_cases.json"

SCENARIOS = ["refund", "return", "exchange", "logistics"]
SCENARIO_ZH = {"refund": "退款（未发货）", "return": "退货（质量/破损问题）",
               "exchange": "换货（尺寸/型号不符）", "logistics": "物流异常（长期未送达/丢件）"}
# 每场景 15 条：calm 5 / impatient 5 / angry 5；angry 中 2 条为过度索赔（期望升级）
EMOTIONS = ["calm"] * 5 + ["impatient"] * 5 + ["angry"] * 5
PATIENCE = {"calm": 4, "impatient": 3, "angry": 2}

REQUIRED_FIELDS = ["case_id", "issue_type", "emotion", "patience", "order",
                   "persona_desc", "first_utterance", "expected_end",
                   "demand_excessive"]
ORDER_FIELDS = ["product", "amount", "shipped"]


def build_skeletons() -> list[dict]:
    rng = random.Random(42)
    skeletons = []
    for scenario in SCENARIOS:
        for i, emotion in enumerate(EMOTIONS):
            demand_excessive = emotion == "angry" and i >= 13  # 每场景最后2条
            skeletons.append({
                "case_id": f"{scenario}_{i+1:02d}",
                "issue_type": scenario,
                "emotion": emotion,
                "patience": PATIENCE[emotion],
                "order": {"amount": rng.choice([39, 89, 159, 299, 599, 1299, 2999]),
                          "shipped": scenario != "refund"},
                "expected_end": "escalation" if demand_excessive else "resolution",
                "demand_excessive": demand_excessive,
            })
    return skeletons


PROMPT_TMPL = """你是电商售后测试数据构造器。给定 {n} 条测试用例骨架，为每条补全三个字段：
1. product: 具体商品名（品类要多样：服装/数码/家居/食品/美妆/母婴等，与金额匹配）
2. persona_desc: 用户人设一句话（性别年龄职业+说话风格，与情绪档位匹配）
3. first_utterance: 用户找客服说的第一句话（口语化、自然，像真实电商用户，可带错别字或省略，不要报订单号）

情绪档位说明：calm=平和讲理；impatient=着急催促；angry=愤怒指责。
demand_excessive=true 的用户：首句或人设中体现"要求高额赔偿（超过订单金额20%）、态度强硬不接受拒绝"。

骨架：
{skeletons}

输出 JSON：{{"cases": [{{"case_id": "...", "product": "...", "persona_desc": "...", "first_utterance": "..."}}]}}，顺序与输入一致。"""


def fill_batch(batch: list[dict]) -> None:
    slim = [{"case_id": s["case_id"], "issue_type": SCENARIO_ZH[s["issue_type"]],
             "amount": s["order"]["amount"], "emotion": s["emotion"],
             "demand_excessive": s["demand_excessive"]} for s in batch]
    prompt = PROMPT_TMPL.format(n=len(batch), skeletons=json.dumps(slim, ensure_ascii=False, indent=1))
    out = llm.chat_json([{"role": "user", "content": prompt}], temperature=1.0, max_tokens=2048)
    filled = {c["case_id"]: c for c in out["cases"]}
    for s in batch:
        c = filled[s["case_id"]]
        s["order"]["product"] = c["product"]
        s["persona_desc"] = c["persona_desc"]
        s["first_utterance"] = c["first_utterance"]


def validate(cases: list[dict]) -> list[str]:
    errors = []
    for c in cases:
        for f in REQUIRED_FIELDS:
            if f not in c:
                errors.append(f"{c.get('case_id', '?')}: 缺字段 {f}")
        for f in ORDER_FIELDS:
            if f not in c.get("order", {}):
                errors.append(f"{c.get('case_id', '?')}: order 缺 {f}")
        if not c.get("first_utterance", "").strip():
            errors.append(f"{c.get('case_id', '?')}: 首句为空")
    return errors


def main():
    skeletons = build_skeletons()
    assert len(skeletons) == 60
    for i in range(0, 60, 5):
        batch = skeletons[i:i + 5]
        fill_batch(batch)
        print(f"批次 {i // 5 + 1}/12 完成: {[s['case_id'] for s in batch]}")

    errors = validate(skeletons)
    if errors:
        print("校验失败:", *errors, sep="\n  ")
        sys.exit(1)

    OUT_PATH.write_text(json.dumps(skeletons, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n60 条全部通过 schema 校验 → {OUT_PATH}")

    print("\n===== 人工抽查样本 =====")
    rng = random.Random(7)
    for c in rng.sample(skeletons, 5):
        print(f"\n[{c['case_id']}] {c['emotion']} 期望={c['expected_end']}")
        print(f"  商品: {c['order']['product']} ¥{c['order']['amount']} 已发货={c['order']['shipped']}")
        print(f"  人设: {c['persona_desc']}")
        print(f"  首句: {c['first_utterance']}")


if __name__ == "__main__":
    main()
