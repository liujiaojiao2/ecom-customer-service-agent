"""MCTS 核心：确定性玩具 MDP 验证（无 LLM）。"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from mcts import search


class ToyEnv:
    """确定性玩具 MDP：
    root --good--> g --finish--> 终局奖励 1.0
    root --bad---> b --finish--> 终局奖励 0.1
    root --dead--> d（无后续动作，rollout 返回 0）
    'illegal' 动作永远不在合法集合里。
    """
    TRANSITIONS = {
        ("root", "good"): "g", ("root", "bad"): "b", ("root", "dead"): "d",
        ("g", "finish"): "WIN", ("b", "finish"): "LOSE",
    }
    LEGAL = {"root": ["good", "bad", "dead"], "g": ["finish"], "b": ["finish"], "d": []}
    REWARD = {"WIN": 1.0, "LOSE": 0.1}

    def copy_state(self, s):
        return s

    def legal_actions(self, s):
        return list(self.LEGAL[s])

    def step(self, s, a):
        assert a in self.LEGAL[s], f"MCTS 使用了非法动作 {a}@{s}"
        nxt = self.TRANSITIONS[(s, a)]
        if nxt in self.REWARD:
            return nxt, True, self.REWARD[nxt]
        return nxt, False, None

    def rollout(self, s, max_steps):
        rng = random.Random(0)
        for _ in range(max_steps):
            acts = self.LEGAL.get(s, [])
            if not acts:
                return 0.0
            s2, terminal, r = self.step(s, rng.choice(acts))
            if terminal:
                return r
            s = s2
        return 0.0


def test_finds_optimal_action():
    best, stats = search(ToyEnv(), "root", budget=50, rng=random.Random(42))
    assert best == "good"
    assert stats["good"]["visits"] > stats["bad"]["visits"] > 0


def test_masking_only_legal_in_tree():
    _, stats = search(ToyEnv(), "root", budget=50, rng=random.Random(1))
    assert set(stats) <= {"good", "bad", "dead"}  # 'illegal' 从未进树


def test_backprop_conservation_and_q():
    """访问数守恒：根子节点访问数之和 == 预算；确定性奖励下 Q 值精确。"""
    best, stats = search(ToyEnv(), "root", budget=60, rng=random.Random(7))
    assert sum(s["visits"] for s in stats.values()) == 60
    assert stats["good"]["q"] == 1.0
    assert stats["bad"]["q"] == 0.1
    assert stats["dead"]["q"] == 0.0


def test_budget_one_no_crash():
    best, stats = search(ToyEnv(), "root", budget=1, rng=random.Random(3))
    assert best in {"good", "bad", "dead"}


def test_no_legal_actions_raises():
    with pytest.raises(ValueError):
        search(ToyEnv(), "d", budget=4)


def test_depth_cap_truncates():
    """深度上限 1：扩展后不再深入，good/bad 的 Q 都来自截断 rollout（0 步→0）。"""
    _, stats = search(ToyEnv(), "root", budget=30, max_depth=1, rng=random.Random(5))
    assert stats["good"]["q"] == 0.0 and stats["bad"]["q"] == 0.0
