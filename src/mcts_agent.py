"""MCTS Agent：每个真实对话轮跑一次 UCT 搜索选动作，LLM 只做回复渲染。

与 BaselineAgent 同接口 act(history)，但有状态（跟踪自身 SOP 节点），一个实例只服务一条对话。
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm
from mcts import search
from mcts_env import ACTION_TEMPLATES, SearchEnv
from sop import SOPEngine, SOPState
from user_simulator import UserSimulator

# 可选导入本地 LLM
try:
    import local_llm
except ImportError:
    local_llm = None

RENDER_SYSTEM = """你是电商售后客服。上级规划系统已经决定了你这一轮必须执行的动作，你只负责把它表达成一句得体的话。

## 本轮必须执行的动作
{action}：{action_desc}

## 要求
- 回复必须忠实执行该动作，不得承诺动作之外的任何事情（不许擅自承诺退款/赔偿/时限）
- 口语化、专业、有温度，1~3 句话
- 直接输出回复文本，不要 JSON、不要解释"""


class MCTSAgent:
    def __init__(self, case: dict, engine: SOPEngine | None = None,
                 budget: int = 16, max_depth: int = 8, uct_c: float = 1.4,
                 branch_ctx: dict | None = None, use_local_rollout: bool = False,
                 reward_mode: str = "terminal"):
        """branch_ctx 未提供时用测试用例元数据（阶段0/1 口径）；
        阶段2 从感知层结果传入 {"issue_type", "evidence_valid"}。

        Args:
            use_local_rollout: 若为 True，rollout 使用本地小模型；
                              若为 False，rollout 使用云模型（默认行为）。
            reward_mode: "terminal"（仅终局奖励）| "dense"（附加手写步奖励）。
        """
        self.engine = engine or SOPEngine()
        self.budget = budget
        self.max_depth = max_depth
        self.uct_c = uct_c
        self.sop_state = SOPState()
        seed = sum(ord(ch) for ch in case["case_id"])
        self.rng = random.Random(seed)
        if branch_ctx is None:
            branch_ctx = {"issue_type": case["issue_type"], "evidence_valid": True}

        # 构造 rollout_sim（若启用本地 rollout）
        rollout_sim = None
        if use_local_rollout:
            if local_llm is None:
                raise RuntimeError(
                    "使用本地 rollout 需要 local_llm 模块可用"
                )
            rollout_sim = UserSimulator(case, chat_fn=local_llm.chat_json)

        self.env = SearchEnv(self.engine, UserSimulator(case), branch_ctx,
                             rng=random.Random(seed + 1), rollout_sim=rollout_sim,
                             reward_mode=reward_mode)

    def act(self, history: list[dict]) -> dict:
        root = self.env.make_root(self.sop_state, history)
        action, stats = search(self.env, root, budget=self.budget,
                               c=self.uct_c, max_depth=self.max_depth, rng=self.rng,
                               prior_fn=self.env.compute_prior)
        result = self.engine.step(self.sop_state, action,
                                  self.env.branch_ctx)
        assert not result.violation, f"MCTS 选出违规动作: {result.violation_reason}"
        self.sop_state = result.state

        system = RENDER_SYSTEM.format(action=action, action_desc=ACTION_TEMPLATES[action])
        transcript = "\n".join(
            f"{'用户' if t['role'] == 'user' else '客服'}: {t['content']}" for t in history)
        reply = llm.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": f"对话记录：\n{transcript}\n\n请输出你这一轮的回复。"}],
            temperature=0.5)
        return {"action": action, "reply": reply.strip(), "search_stats": stats}
