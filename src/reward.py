"""多因素终局奖励：R = w1·完成 + w2·满意度 − w3·轮数。"""
from dataclasses import dataclass

SATISFACTION_SCORE = {"satisfied": 1.0, "neutral": 0.0, "angry": -1.0}

# PRM Step 1：手写稠密步奖励表。量级远小于终局奖励（S10 完成项 = 1.0），
# 避免稠密项淹没终局信号（典型成功路径 ~6 步推进 ≈ +0.35 << 1.0）。
DENSE_STEP_REWARD = {
    # 推进动作小正奖励
    "greet_and_ask": 0.05, "ask_order_info": 0.05, "query_order": 0.05,
    "check_refund_policy": 0.05, "request_evidence": 0.05,
    "review_evidence": 0.05, "query_logistics": 0.05,
    "propose_solution": 0.08, "execute_action": 0.08,
    "confirm_resolution": 0.05,
    # 非推进动作小负奖励
    "comfort_user": -0.02, "clarify_issue": -0.02,
    "revise_solution": -0.02, "escalate": -0.05,
    "end_session": 0.0,
}


# PRM Step 2：手写状态势能表 Φ（PBRS，Ng et al. 1999）。
# 三个硬约束：① Φ 只依赖状态不依赖动作；② 终止状态 Φ ≡ 0；
# ③ γ 与 MCTS 回传折扣一致（本实现无折扣，γ=1）。
# 赋值依据：距 S10 的最坏剩余步数 d(s)，Φ(s) = 0.05 × (8 − d(s))。
# γ=1 + 终止 Φ=0 下，完整轨迹的塑形总和恒为 −Φ(根)，不改变轨迹排序；
# 截断 rollout 得 Φ(截断态) − Φ(根)，推进越深奖励越高——收益正来自于此。
PHI_TABLE = {
    "S0": 0.00,   # d=8（return/exchange 最长路径）
    "S1": 0.05,   # d=7
    "S2": 0.10,   # d=6
    "S3": 0.20,   # d=4
    "S4": 0.15,   # d=5
    "S5": 0.20,   # d=4
    "S6": 0.20,   # d=4
    "S7": 0.25,   # d=3
    "S8": 0.30,   # d=2
    "S9": 0.35,   # d=1
    "S10": 0.0,   # 终止，硬约束
    "S11": 0.0,   # 终止，硬约束
}


def pbrs_bonus(prev_node: str, next_node: str) -> float:
    """势能塑形项 γ·Φ(s′) − Φ(s)，γ=1。"""
    return PHI_TABLE[next_node] - PHI_TABLE[prev_node]


@dataclass
class RewardConfig:
    w_completion: float = 1.0
    w_satisfaction: float = 0.3
    w_turn: float = 0.05
    resolution_value: float = 1.0   # S10
    escalation_value: float = 0.4   # S11：合法出口但非最优


def terminal_reward(final_node: str, satisfaction: str | None, num_turns: int,
                    cfg: RewardConfig = RewardConfig()) -> float:
    if final_node == "S10":
        completion = cfg.resolution_value
    elif final_node == "S11":
        completion = cfg.escalation_value
    else:
        completion = 0.0
    sat = SATISFACTION_SCORE.get(satisfaction or "neutral", 0.0)
    return (cfg.w_completion * completion
            + cfg.w_satisfaction * sat
            - cfg.w_turn * num_turns)
