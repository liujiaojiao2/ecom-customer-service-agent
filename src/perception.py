"""多模态感知层：图片 + 文本 + 行为流 → SOP 分流字段（issue_type, has_evidence）。

Zero-shot Qwen-VL；调用注入可 mock，方便单测。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm_vl

ISSUE_TYPES = ("refund", "return", "exchange", "logistics")

SYSTEM_PROMPT = """你是电商售后场景的多模态分类器。给定：
- 用户首句文本
- 一张相关图片（商品实物/订单截图/物流面单等）
- 用户近期行为流（按时间顺序的操作序列）

请判断两件事：
1. issue_type：售后诉求类型，四选一
   - "refund"：仅退款（商品未发货，用户要求撤单退款）
   - "return"：退货退款（商品有质量/破损问题，需寄回）
   - "exchange"：换货（尺寸/型号/颜色不符，需要换发）
   - "logistics"：物流异常（长期未送达/丢件）
2. has_evidence：图片是否为**支撑退货/换货诉求的有效实物证据**
   - true：图片是商品破损、瑕疵、错发的实物照片
   - false：图片是订单截图、物流面单、页面截图等**非实物证据**
     （refund/logistics 类无需实物证据，has_evidence 通常为 false）

输出严格 JSON：
{"issue_type": "refund|return|exchange|logistics",
 "has_evidence": true|false,
 "confidence": 0.0~1.0,
 "reason": "一句话说明"}"""

USER_TMPL = """用户首句：{utterance}

用户近期行为流：
{behaviors}

请综合图片、文本、行为流做出分类。"""


class Perception:
    def __init__(self, chat_fn=None):
        self._chat = chat_fn or llm_vl.chat_vl_json

    def perceive(self, mm_input: dict) -> dict:
        """mm_input: {text, image, behaviors}。"""
        for k in ("text", "image", "behaviors"):
            if k not in mm_input:
                raise ValueError(f"mm_input 缺字段 {k}")
        behaviors_txt = " → ".join(mm_input["behaviors"]) if mm_input["behaviors"] else "（无）"
        user_prompt = SYSTEM_PROMPT + "\n\n" + USER_TMPL.format(
            utterance=mm_input["text"], behaviors=behaviors_txt)
        out = self._chat(user_prompt, images=[mm_input["image"]])
        return self._normalize(out)

    @staticmethod
    def _normalize(raw: dict) -> dict:
        issue = raw.get("issue_type")
        if issue not in ISSUE_TYPES:
            raise ValueError(f"感知输出 issue_type 非法: {issue}")
        return {
            "issue_type": issue,
            "has_evidence": bool(raw.get("has_evidence", False)),
            "confidence": float(raw.get("confidence", 0.0)),
            "reason": str(raw.get("reason", "")),
        }
