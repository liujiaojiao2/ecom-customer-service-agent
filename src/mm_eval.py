"""多模态感知评估：准确率 / per-class 精确率召回率 / 混淆矩阵。"""
from collections import defaultdict

from mm_dataset import ISSUE_TYPES

CLASSES = tuple(sorted(ISSUE_TYPES))


def compute_metrics(records: list[dict]) -> dict:
    """records: [{gold: {issue_type, has_evidence}, pred: {issue_type, has_evidence}}]。"""
    if not records:
        return {"n": 0}

    n = len(records)
    it_correct = sum(1 for r in records
                     if r["pred"]["issue_type"] == r["gold"]["issue_type"])
    ev_correct = sum(1 for r in records
                     if r["pred"]["has_evidence"] == r["gold"]["has_evidence"])
    joint_correct = sum(1 for r in records
                        if r["pred"]["issue_type"] == r["gold"]["issue_type"]
                        and r["pred"]["has_evidence"] == r["gold"]["has_evidence"])

    # per-class 精确率/召回率
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for r in records:
        g, p = r["gold"]["issue_type"], r["pred"]["issue_type"]
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    per_class = {}
    for c in CLASSES:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "support": tp[c] + fn[c]}

    # 混淆矩阵 confusion[gold][pred]=count
    confusion = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    for r in records:
        confusion[r["gold"]["issue_type"]][r["pred"]["issue_type"]] += 1

    return {
        "n": n,
        "issue_type_accuracy": it_correct / n,
        "has_evidence_accuracy": ev_correct / n,
        "joint_accuracy": joint_correct / n,
        "per_class": per_class,
        "confusion": confusion,
    }


def format_report(m: dict) -> str:
    if m.get("n", 0) == 0:
        return "(空评估集)"
    lines = [
        f"总数: {m['n']}",
        f"issue_type 准确率:   {m['issue_type_accuracy']:.1%}",
        f"has_evidence 准确率: {m['has_evidence_accuracy']:.1%}",
        f"联合准确率:          {m['joint_accuracy']:.1%}",
        "",
        f"{'类别':10s} {'精确率':>8s} {'召回率':>8s} {'F1':>8s} {'支持':>6s}",
    ]
    for c, s in m["per_class"].items():
        lines.append(f"{c:10s} {s['precision']:8.1%} {s['recall']:8.1%} "
                     f"{s['f1']:8.1%} {s['support']:6d}")
    lines.append("")
    lines.append("混淆矩阵（行=真实，列=预测）:")
    header = " " * 10 + "".join(f"{c:>10s}" for c in CLASSES)
    lines.append(header)
    for g in CLASSES:
        row = f"{g:10s}" + "".join(f"{m['confusion'][g][p]:>10d}" for p in CLASSES)
        lines.append(row)
    return "\n".join(lines)
