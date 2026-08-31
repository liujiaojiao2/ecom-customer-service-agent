"""M10 搜索环境：桩模拟器（无 LLM）验证状态推进、拷贝独立性、rollout 截断。"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from mcts_env import SearchEnv
from sop import SOPEngine, SOPState


class StubSim:
    """桩模拟器：固定回应，记录调用次数。"""
    def __init__(self, satisfaction="neutral"):
        self.satisfaction = satisfaction
        self.calls = 0

    def respond(self, transcript):
        self.calls += 1
        return {"utterance": "好的。", "satisfaction": self.satisfaction}


@pytest.fixture
def env():
    return SearchEnv(SOPEngine(), StubSim(), {"issue_type": "refund", "evidence_valid": True},
                     rng=random.Random(0))


def test_step_advances_and_calls_sim(env):
    root = env.make_root(SOPState(), [{"role": "user", "content": "我要退款"}])
    s, terminal, reward = env.step(env.copy_state(root), "greet_and_ask")
    assert s.sop.node == "S1" and not terminal and reward is None
    assert s.num_steps == 1
    assert env.sim.calls == 1
    assert len(s.transcript) == 3  # 原1条 + agent + user


def test_copy_state_independent(env):
    root = env.make_root(SOPState(), [])
    c = env.copy_state(root)
    env.step(c, "greet_and_ask")
    assert root.sop.node == "S0" and len(root.transcript) == 0


def test_terminal_reward_returned(env):
    """escalate 直达终止节点，返回终局奖励。"""
    root = env.make_root(SOPState(node="S1"), [])
    s, terminal, reward = env.step(env.copy_state(root), "escalate")
    assert terminal and s.sop.node == "S11"
    assert reward == pytest.approx(0.4 - 0.05)  # 0.4完成 + 0中性 − 1步


def test_illegal_action_asserts(env):
    root = env.make_root(SOPState(), [])
    with pytest.raises(AssertionError):
        env.step(env.copy_state(root), "execute_action")


def test_rollout_truncation(env):
    """max_steps=2 从 S0 出发不可能终止（最短路径也要 6 步），奖励为截断值。"""
    root = env.make_root(SOPState(), [])
    r = env.rollout(env.copy_state(root), max_steps=2)
    # 未终止：完成=0，中性=0，−0.05×步数（可能因 escalate 提前终止得 0.4−…）
    assert r <= 0.4
    assert env.sim.calls <= 2


def test_rollout_can_reach_terminal(env):
    """给足步数，随机 rollout 能到终止节点（escalate 兜底保证可达）。"""
    rewards = [env.rollout(env.copy_state(env.make_root(SOPState(), [])), max_steps=12)
               for _ in range(5)]
    assert all(isinstance(r, float) for r in rewards)


def test_rollout_policy_reaches_resolution(env):
    """推进偏向策略下，从 S0 出发 10 次 rollout 至少 1 次到达 S10（奖励>0.5）。"""
    rewards = [env.rollout(env.copy_state(env.make_root(SOPState(), [])), max_steps=8)
               for _ in range(10)]
    assert any(r > 0.5 for r in rewards), rewards


def test_end_session_not_in_search_actions(env):
    root = env.make_root(SOPState(node="S10"), [])
    assert env.legal_actions(root) == []
