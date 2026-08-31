"""M15 感知层：注入 fake chat_fn，覆盖正常/边界/错误三档。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from perception import Perception


def make_input(**overrides):
    base = {"text": "衣服掉色了", "image": "/tmp/x.jpg",
            "behaviors": ["查看订单", "点击售后"]}
    base.update(overrides)
    return base


def fake_chat(returns):
    calls = []

    def _chat(prompt, images=None):
        calls.append({"prompt": prompt, "images": images})
        return returns.pop(0) if isinstance(returns, list) else returns
    _chat.calls = calls
    return _chat


def test_normal_return_with_evidence():
    p = Perception(chat_fn=fake_chat(
        {"issue_type": "return", "has_evidence": True,
         "confidence": 0.92, "reason": "掉色图 + 售后行为"}))
    out = p.perceive(make_input())
    assert out == {"issue_type": "return", "has_evidence": True,
                   "confidence": 0.92, "reason": "掉色图 + 售后行为"}


def test_refund_flags_no_evidence():
    p = Perception(chat_fn=fake_chat(
        {"issue_type": "refund", "has_evidence": False, "confidence": 0.88, "reason": "未发货"}))
    out = p.perceive(make_input(text="还没发货，退款", behaviors=["申请退款"]))
    assert out["issue_type"] == "refund" and out["has_evidence"] is False


def test_prompt_includes_utterance_and_behaviors():
    """边界：prompt 装配包含文本与行为流。"""
    chat = fake_chat({"issue_type": "logistics", "has_evidence": False, "confidence": 0.7})
    Perception(chat_fn=chat).perceive(make_input(
        text="快递不动了", behaviors=["查看物流", "催发货"]))
    call = chat.calls[0]
    assert "快递不动了" in call["prompt"]
    assert "查看物流 → 催发货" in call["prompt"]
    assert call["images"] == ["/tmp/x.jpg"]


def test_empty_behaviors_ok():
    p = Perception(chat_fn=fake_chat(
        {"issue_type": "logistics", "has_evidence": False, "confidence": 0.5}))
    out = p.perceive(make_input(behaviors=[]))
    assert out["issue_type"] == "logistics"


def test_missing_field_raises():
    p = Perception(chat_fn=fake_chat({}))
    with pytest.raises(ValueError, match="缺字段"):
        p.perceive({"text": "x", "image": "/tmp/x.jpg"})


def test_illegal_issue_type_raises():
    p = Perception(chat_fn=fake_chat(
        {"issue_type": "gift_card", "has_evidence": False}))
    with pytest.raises(ValueError, match="issue_type 非法"):
        p.perceive(make_input())


def test_normalize_defaults():
    """confidence/has_evidence 缺失时不崩，取默认值。"""
    p = Perception(chat_fn=fake_chat({"issue_type": "exchange"}))
    out = p.perceive(make_input())
    assert out["has_evidence"] is False and out["confidence"] == 0.0
