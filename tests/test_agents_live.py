"""M4 模拟器 / M5 基线 Agent 的活体测试（调真实 DeepSeek API）。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest
from baseline_agent import BaselineAgent
from sop import SOPEngine
from user_simulator import UserSimulator

CASES = json.loads((ROOT / "data" / "test_cases.json").read_text(encoding="utf-8"))
CASE_BY_ID = {c["case_id"]: c for c in CASES}


# ---------- M4 用户模拟器 ----------

def test_sim_accepts_reasonable_solution():
    """正常：平和用户对满足诉求的方案表示接受。"""
    case = next(c for c in CASES if c["emotion"] == "calm" and c["issue_type"] == "refund")
    sim = UserSimulator(case)
    history = [
        {"role": "user", "content": case["first_utterance"]},
        {"role": "agent", "content": "您好，已核实您的订单，商品尚未发货，现在就为您办理全额退款，1-3个工作日原路退回，您看可以吗？"},
    ]
    out = sim.respond(history)
    print("模拟器-接受方案:", out)
    assert out["satisfaction"] == "satisfied"


def test_sim_low_patience_gets_angry_on_repetition():
    """边界：低耐心用户被反复索要同一信息后情绪恶化。"""
    case = next(c for c in CASES if c["emotion"] == "angry")
    sim = UserSimulator(case)
    ask = "麻烦您提供一下订单号，我帮您查询。"
    history = [{"role": "user", "content": case["first_utterance"]}]
    for _ in range(3):
        history.append({"role": "agent", "content": ask})
        history.append({"role": "user", "content": "订单号不是发过了吗？！"})
    history.append({"role": "agent", "content": ask})
    out = sim.respond(history)
    print("模拟器-重复追问:", out)
    assert out["satisfaction"] != "satisfied"


def test_sim_rejects_offtopic_agent():
    """错误扰动：Agent 答非所问时模拟器不配合、不满意。"""
    case = next(c for c in CASES if c["issue_type"] == "return")
    sim = UserSimulator(case)
    history = [
        {"role": "user", "content": case["first_utterance"]},
        {"role": "agent", "content": "今天天气不错，最近我们店铺有新品上架，您要看看吗？"},
    ]
    out = sim.respond(history)
    print("模拟器-答非所问:", out)
    assert out["satisfaction"] != "satisfied"


# ---------- M5 基线 Agent ----------

def test_agent_first_turn_greets():
    """正常：空历史首轮应输出 greet_and_ask。"""
    agent = BaselineAgent()
    out = agent.act([])
    print("基线-首轮:", out)
    assert out["action"] == "greet_and_ask"


def test_agent_action_in_action_set():
    """格式：动作名必须在 SOP 动作全集内。"""
    eng = SOPEngine()
    agent = BaselineAgent()
    case = CASES[0]
    history = [
        {"role": "agent", "content": "您好，请问有什么可以帮您？"},
        {"role": "user", "content": case["first_utterance"]},
    ]
    out = agent.act(history)
    print("基线-动作:", out)
    assert out["action"] in eng.all_actions


@pytest.mark.parse_rate
def test_agent_json_parse_rate():
    """稳定性：20 次采样 JSON 可解析且含必需字段的比例 ≥ 95%。"""
    agent = BaselineAgent()
    case = CASES[1]
    history = [
        {"role": "agent", "content": "您好，请问有什么可以帮您？"},
        {"role": "user", "content": case["first_utterance"]},
    ]
    ok = 0
    for i in range(20):
        try:
            agent.act(history)
            ok += 1
        except Exception as e:
            print(f"第{i}次失败: {e}")
    print(f"解析成功率: {ok}/20")
    assert ok >= 19
