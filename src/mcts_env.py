"""MCTS 搜索环境：SOP 引擎 + 用户模拟器 + 奖励函数。

动作级 rollout（裁决 D1）：搜索内客服话语用固定模板，不调 LLM 渲染；
模拟器对模板回应（每次 env.step 一次 LLM 调用）。真实对话轮不经过本模块。
"""
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reward import DENSE_STEP_REWARD, RewardConfig, pbrs_bonus, terminal_reward
from sop import SOPEngine, SOPState

ACTION_TEMPLATES = {
    "greet_and_ask": "您好，请问有什么可以帮您？麻烦您描述一下遇到的问题。",
    "clarify_issue": "能再详细说明一下您遇到的问题吗？",
    "ask_order_info": "麻烦您提供一下订单号，我帮您核实。",
    "query_order": "我已为您查询并核实了订单信息。",
    "check_refund_policy": "已核实退款政策：您的订单符合退款条件。",
    "request_evidence": "麻烦您提供问题商品的照片或相关凭证。",
    "review_evidence": "我已查看您提供的凭证，确认情况属实。",
    "query_logistics": "我已为您查询物流，确认包裹当前的异常状态。",
    "propose_solution": "根据您的情况，我按流程为您提出处理方案（退款/换货/补发等），您看是否可以？",
    "revise_solution": "理解您的顾虑，我为您调整一个新的处理方案，您看这样是否可以？",
    "execute_action": "好的，已为您提交处理（退款/退货单/补发已执行）。",
    "confirm_resolution": "您的问题已处理完成，感谢您的耐心等待，请问还有其他需要帮助的吗？",
    "comfort_user": "非常理解您的心情，给您带来不便我们深表歉意，一定尽快为您解决。",
    "escalate": "这边为您转接人工专员进一步处理，请稍候。",
    "end_session": "感谢您的咨询，祝您生活愉快。",
}

# PUCT 规则先验（归一化后）
PUCT_PRIOR = {
    "escalate": 0.02,
    "comfort_user": 0.03,
    "clarify_issue": 0.03,
    "revise_solution": 0.03,
    "greet_and_ask": 0.10,
    "ask_order_info": 0.10,
    "query_order": 0.10,
    "check_refund_policy": 0.10,
    "request_evidence": 0.10,
    "review_evidence": 0.10,
    "query_logistics": 0.10,
    "propose_solution": 0.14,
    "execute_action": 0.14,
    "confirm_resolution": 0.01,
    "end_session": 0.00,
}


@dataclass
class SimState:
    sop: SOPState
    transcript: list = field(default_factory=list)  # [{role, content}]
    num_steps: int = 0
    last_satisfaction: str = "neutral"


class SearchEnv:
    def __init__(self, engine: SOPEngine, simulator, branch_ctx: dict,
                 reward_cfg: RewardConfig = RewardConfig(),
                 rng: random.Random | None = None, rollout_sim=None,
                 reward_mode: str = "terminal"):
        """reward_mode: "terminal"（仅终局）| "dense"（手写步奖励）| "pbrs"（势能塑形）。"""
        if reward_mode not in ("terminal", "dense", "pbrs"):
            raise ValueError(f"未知 reward_mode: {reward_mode!r}（应为 terminal | dense | pbrs）")
        self.engine = engine
        self.sim = simulator
        self.rollout_sim = rollout_sim
        self.branch_ctx = branch_ctx
        self.reward_cfg = reward_cfg
        self.reward_mode = reward_mode
        self.rng = rng or random.Random()

    def make_root(self, sop_state: SOPState, history: list) -> SimState:
        return SimState(sop=sop_state.copy(), transcript=list(history),
                        num_steps=0, last_satisfaction="neutral")

    def copy_state(self, s: SimState) -> SimState:
        return SimState(sop=s.sop.copy(), transcript=list(s.transcript),
                        num_steps=s.num_steps, last_satisfaction=s.last_satisfaction)

    def legal_actions(self, s: SimState) -> list[str]:
        acts = self.engine.legal_actions(s.sop)
        return [a for a in acts if a != "end_session"]

    def compute_prior(self, actions: list[str]) -> dict[str, float]:
        """计算规则先验并按 legal_actions 归一化。

        返回 {action: normalized_prior} 对应 actions 列表中的动作。
        """
        raw_priors = {a: PUCT_PRIOR.get(a, 0.1) for a in actions}
        total = sum(raw_priors.values())
        if total == 0:
            # 如果所有先验都是 0，均匀分配
            return {a: 1.0 / len(actions) for a in actions}
        return {a: p / total for a, p in raw_priors.items()}

    def _reward(self, s: SimState) -> float:
        return terminal_reward(s.sop.node, s.last_satisfaction, s.num_steps,
                               self.reward_cfg)

    def step(self, s: SimState, action: str):
        """执行动作（模板话语）→ 模拟器回应。返回 (state, terminal, reward)。

        terminal 模式：非终止步 reward=None，终止步 reward=终局奖励。
        dense 模式：非终止步 reward=步奖励，终止步 reward=步奖励+终局奖励。
        两种模式下，一次 simulation 的总回报 = 沿途所有非 None reward 之和
        （+ 叶子处 rollout/截断奖励），由调用方累加、只回传一次。
        """
        prev_node = s.sop.node
        result = self.engine.step(s.sop, action, self.branch_ctx)
        assert not result.violation, f"搜索环境收到违规动作: {result.violation_reason}"
        s.sop = result.state
        s.num_steps += 1
        s.transcript.append({"role": "agent", "content": ACTION_TEMPLATES[action]})

        user_out = self.sim.respond(s.transcript)
        s.transcript.append({"role": "user", "content": user_out["utterance"]})
        s.last_satisfaction = user_out["satisfaction"]

        if self.reward_mode == "dense":
            step_r = DENSE_STEP_REWARD[action]
        elif self.reward_mode == "pbrs":
            step_r = pbrs_bonus(prev_node, s.sop.node)
        else:
            step_r = None
        if result.terminal:
            term_r = self._reward(s)
            return s, True, term_r + step_r if step_r is not None else term_r
        return s, False, step_r

    # 非推进动作：自环（comfort/clarify）、回退（revise）、放弃（escalate）。
    # 均匀随机 rollout 连续命中 6+ 步正确动作的概率 ≈ 1/1800，终局奖励不可见，
    # 故 rollout 默认策略偏向推进动作（领域知情 default policy）；树内 UCT 不受影响。
    NON_PROGRESS = {"comfort_user", "clarify_issue", "revise_solution", "escalate"}

    def _rollout_action(self, acts: list[str]) -> str:
        """有推进动作时必选推进；escalate/comfort 的价值由树内 UCT 探索。"""
        progress = [a for a in acts if a not in self.NON_PROGRESS]
        return self.rng.choice(progress or acts)

    def rollout(self, s: SimState, max_steps: int) -> float:
        """按默认策略推演至终止或步数上限；未终止按截断状态计奖励。

        dense 模式下返回值 = 沿途步奖励之和 + 终局/截断奖励
        （终止步的 step() 返回已含"步奖励+终局奖励"，直接累加即可）。
        terminal 模式下非终止步 reward=None 不累加，行为与原实现一致。
        若 rollout_sim 不为 None，使用它代替 self.sim 进行 rollout 的模拟。
        """
        # 选择要使用的模拟器
        sim_to_use = self.rollout_sim if self.rollout_sim is not None else self.sim

        # 保存原来的 self.sim，临时替换为 sim_to_use
        original_sim = self.sim
        self.sim = sim_to_use
        total = 0.0
        try:
            for _ in range(max_steps):
                acts = self.legal_actions(s)
                if not acts:
                    break
                s, terminal, reward = self.step(s, self._rollout_action(acts))
                if reward is not None:
                    total += reward
                if terminal:
                    return total
            return total + self._reward(s)
        finally:
            # 恢复原来的 self.sim
            self.sim = original_sim
