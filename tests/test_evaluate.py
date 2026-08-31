"""M7 评估模块：用手工构造的轨迹验证指标计算。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate import compute_metrics


def make_turn(strict_v=False, resync_v=False, n=1):
    return {"turn": n, "strict_violation": strict_v, "resync_violation": resync_v}


def test_all_compliant_completed():
    """全合规轨迹：合规率100%，完成率100%。"""
    trajs = [{
        "status": "ok", "completed": True, "final_node_resync": "S10",
        "expected_end": "resolution", "num_turns": 4, "issue_type": "refund",
        "turns": [make_turn(n=i) for i in range(1, 5)],
    }]
    m = compute_metrics(trajs)
    assert m["strict_compliance_rate"] == 1.0
    assert m["completion_rate"] == 1.0
    assert m["avg_turns"] == 4
    assert m["expected_end_match_rate"] == 1.0


def test_with_violations_and_mismatch():
    """含违规轨迹：4 动作中严格违规 2、重同步违规 1；期望终点不符。"""
    trajs = [{
        "status": "ok", "completed": True, "final_node_resync": "S11",
        "expected_end": "resolution", "num_turns": 4, "issue_type": "return",
        "turns": [make_turn(), make_turn(True, True), make_turn(True), make_turn()],
    }]
    m = compute_metrics(trajs)
    assert m["strict_compliance_rate"] == 0.5
    assert m["resync_compliance_rate"] == 0.75
    assert m["expected_end_match_rate"] == 0.0


def test_error_and_incomplete():
    """错误轨迹计入 n_errors 且不参与指标；未完成轨迹拉低完成率。"""
    trajs = [
        {"status": "error", "completed": False, "final_node_resync": "S1",
         "expected_end": "resolution", "num_turns": 1, "issue_type": "refund",
         "turns": [make_turn(), {"error": "boom"}]},
        {"status": "ok", "completed": False, "final_node_resync": "S7",
         "expected_end": "resolution", "num_turns": 12, "issue_type": "refund",
         "turns": [make_turn(n=i) for i in range(1, 13)]},
    ]
    m = compute_metrics(trajs)
    assert m["n_errors"] == 1
    assert m["total_actions"] == 12  # error 轨迹的动作不计
    assert m["completion_rate"] == 0.0
    assert m["avg_turns"] == 12
