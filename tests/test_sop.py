import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from sop import SOPEngine, SOPState


@pytest.fixture
def eng():
    return SOPEngine()


def walk(eng, actions_ctx):
    """依次执行 (action, context) 序列，全程断言无违规，返回最终状态。"""
    state = SOPState()
    for action, ctx in actions_ctx:
        r = eng.step(state, action, ctx)
        assert not r.violation, f"{action}@{state.node}: {r.violation_reason}"
        state = r.state
    return state


# ---------- 正常路径：4 类场景 ----------

def test_refund_path(eng):
    state = walk(eng, [
        ("greet_and_ask", None),
        ("ask_order_info", None),
        ("query_order", {"issue_type": "refund"}),
        ("check_refund_policy", None),
        ("propose_solution", None),
        ("execute_action", None),
        ("confirm_resolution", None),
    ])
    assert state.node == "S10" and eng.is_terminal(state.node)


def test_return_path_with_evidence(eng):
    state = walk(eng, [
        ("greet_and_ask", None),
        ("ask_order_info", None),
        ("query_order", {"issue_type": "return"}),
        ("request_evidence", None),
        ("review_evidence", {"evidence_valid": True}),
        ("propose_solution", None),
        ("execute_action", None),
        ("confirm_resolution", None),
    ])
    assert state.node == "S10"


def test_exchange_path(eng):
    state = walk(eng, [
        ("greet_and_ask", None),
        ("ask_order_info", None),
        ("query_order", {"issue_type": "exchange"}),
    ])
    assert state.node == "S4"


def test_logistics_path(eng):
    state = walk(eng, [
        ("greet_and_ask", None),
        ("ask_order_info", None),
        ("query_order", {"issue_type": "logistics"}),
        ("query_logistics", None),
        ("propose_solution", None),
        ("execute_action", None),
        ("confirm_resolution", None),
    ])
    assert state.node == "S10"


# ---------- 扰动：非法动作 ----------

def test_illegal_action_records_violation_and_keeps_state(eng):
    state = SOPState()  # S0
    r = eng.step(state, "execute_action", None)  # 未核实订单直接执行
    assert r.violation and r.state.node == "S0"


def test_unknown_action(eng):
    r = eng.step(SOPState(), "make_up_action", None)
    assert r.violation and "未知动作" in r.violation_reason


def test_evidence_review_before_collection_is_illegal(eng):
    state = walk(eng, [
        ("greet_and_ask", None),
        ("ask_order_info", None),
        ("query_order", {"issue_type": "return"}),
    ])  # S4
    r = eng.step(state, "review_evidence", {"evidence_valid": True})
    assert r.violation  # 必须先 request_evidence 到 S5


def test_branch_missing_context(eng):
    state = walk(eng, [("greet_and_ask", None), ("ask_order_info", None)])
    r = eng.step(state, "query_order", {})  # 缺 issue_type
    assert r.violation


# ---------- 扰动：自环上限 ----------

def test_clarify_issue_limit(eng):
    state = walk(eng, [("greet_and_ask", None)])
    for _ in range(2):
        r = eng.step(state, "clarify_issue", None)
        assert not r.violation
        state = r.state
    r = eng.step(state, "clarify_issue", None)
    assert r.violation  # 第 3 次超限


def test_comfort_user_selfloop_keeps_node(eng):
    state = walk(eng, [("greet_and_ask", None)])
    r = eng.step(state, "comfort_user", None)
    assert not r.violation and r.state.node == "S1"


def test_revise_solution_limit(eng):
    state = walk(eng, [
        ("greet_and_ask", None),
        ("ask_order_info", None),
        ("query_order", {"issue_type": "refund"}),
        ("check_refund_policy", None),
        ("propose_solution", None),
    ])  # S8
    for _ in range(2):
        state = walk_one(eng, state, "revise_solution")
        state = walk_one(eng, state, "propose_solution")
    r = eng.step(state, "revise_solution", None)
    assert r.violation


def walk_one(eng, state, action, ctx=None):
    r = eng.step(state, action, ctx)
    assert not r.violation, r.violation_reason
    return r.state


# ---------- 终止节点行为 ----------

def test_escalate_from_anywhere(eng):
    state = walk(eng, [("greet_and_ask", None)])
    r = eng.step(state, "escalate", None)
    assert not r.violation and r.state.node == "S11" and r.terminal


def test_terminal_only_allows_end_session(eng):
    state = SOPState(node="S10")
    assert eng.legal_actions(state) == ["end_session"]
    r = eng.step(state, "propose_solution", None)
    assert r.violation


# ---------- 重同步口径 ----------

def test_resync_illegal_action_advances_state(eng):
    """重同步：跳步动作记违规但状态跳到目标节点。"""
    state = SOPState()  # S0，用户已报订单号，Agent 直接查订单
    r = eng.step(state, "query_order", {"issue_type": "refund"}, resync=True)
    assert r.violation and r.state.node == "S3"


def test_resync_legal_action_no_violation(eng):
    state = SOPState()
    r = eng.step(state, "greet_and_ask", None, resync=True)
    assert not r.violation and r.state.node == "S1"


def test_resync_unknown_action_stays(eng):
    r = eng.step(SOPState(), "made_up", None, resync=True)
    assert r.violation and r.state.node == "S0"


def test_resync_reaches_terminal(eng):
    state = SOPState(node="S8")
    r = eng.step(state, "confirm_resolution", None, resync=True)  # 跳过 execute
    assert r.violation and r.state.node == "S10" and r.terminal


def test_coupon_rule(eng):
    assert eng.check_coupon_amount(100, 20)
    assert not eng.check_coupon_amount(100, 20.01)
