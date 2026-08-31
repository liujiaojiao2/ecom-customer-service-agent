"""M16 多模态评估：手工构造记录验证指标。"""
import sys
from pathlib import Path

from pytest import approx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mm_eval import compute_metrics, format_report


def rec(gold_it, gold_ev, pred_it, pred_ev):
    return {"gold": {"issue_type": gold_it, "has_evidence": gold_ev},
            "pred": {"issue_type": pred_it, "has_evidence": pred_ev}}


def test_all_correct():
    records = [rec("refund", False, "refund", False),
               rec("return", True, "return", True),
               rec("exchange", True, "exchange", True),
               rec("logistics", False, "logistics", False)]
    m = compute_metrics(records)
    assert m["issue_type_accuracy"] == 1.0
    assert m["has_evidence_accuracy"] == 1.0
    assert m["joint_accuracy"] == 1.0
    for c in ("refund", "return", "exchange", "logistics"):
        assert m["per_class"][c]["f1"] == 1.0


def test_partial_confusion():
    """return 类 2 条：1 对 1 错（预测成 exchange）→ recall 50%, precision 100%（return 无假阳）。
    exchange 类 1 条正确 + 1 条假阳来自 return → precision 50%。"""
    records = [
        rec("return", True, "return", True),
        rec("return", True, "exchange", True),
        rec("exchange", True, "exchange", True),
    ]
    m = compute_metrics(records)
    assert m["issue_type_accuracy"] == approx(2 / 3)
    assert m["per_class"]["return"] == approx(
        {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "support": 2}, rel=1e-6)
    assert m["per_class"]["exchange"] == approx(
        {"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "support": 1}, rel=1e-6)
    assert m["confusion"]["return"]["exchange"] == 1
    assert m["confusion"]["return"]["return"] == 1


def test_has_evidence_independent():
    """issue_type 全对但 has_evidence 全错。"""
    records = [rec("return", True, "return", False),
               rec("refund", False, "refund", True)]
    m = compute_metrics(records)
    assert m["issue_type_accuracy"] == 1.0
    assert m["has_evidence_accuracy"] == 0.0
    assert m["joint_accuracy"] == 0.0


def test_empty():
    assert compute_metrics([]) == {"n": 0}
    assert format_report({"n": 0}).startswith("(空")


def test_format_report_contains_key_lines():
    records = [rec("refund", False, "refund", False)]
    text = format_report(compute_metrics(records))
    assert "issue_type 准确率" in text and "混淆矩阵" in text
