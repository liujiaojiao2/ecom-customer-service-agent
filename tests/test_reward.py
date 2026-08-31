import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pytest import approx

from reward import RewardConfig, terminal_reward


def test_ordering():
    """核心序关系：满意短对话解决 > 中性解决 > 升级 > 未终止。"""
    r_best = terminal_reward("S10", "satisfied", 4)
    r_neutral = terminal_reward("S10", "neutral", 6)
    r_escalate = terminal_reward("S11", "neutral", 6)
    r_unfinished = terminal_reward("S5", "neutral", 12)
    assert r_best > r_neutral > r_escalate > r_unfinished


def test_turn_penalty_moderate():
    """轮数惩罚不得压倒完成项：12 轮满解决仍优于 4 轮升级。"""
    assert terminal_reward("S10", "neutral", 12) > terminal_reward("S11", "neutral", 4)


def test_angry_resolution_vs_calm_escalation():
    """用户愤怒的解决 vs 平和的升级：完成项主导。"""
    assert terminal_reward("S10", "angry", 6) > terminal_reward("S11", "neutral", 6)


def test_exact_values():
    assert terminal_reward("S10", "satisfied", 4) == approx(1.1)
    assert terminal_reward("S11", "angry", 6) == approx(-0.2)
    assert terminal_reward("S3", None, 12) == approx(-0.6)


def test_config_override():
    cfg = RewardConfig(w_turn=0.0)
    assert terminal_reward("S10", "neutral", 4, cfg) == terminal_reward("S10", "neutral", 12, cfg)
