"""SOP 引擎：状态图加载、动作掩码、转移与合规检查。"""
import json
from dataclasses import dataclass, field
from pathlib import Path

SOP_PATH = Path(__file__).resolve().parent.parent / "sop" / "sop_graph.json"

INITIAL_NODE = "S0"


@dataclass
class SOPState:
    node: str = INITIAL_NODE
    counters: dict = field(default_factory=dict)

    def copy(self) -> "SOPState":
        return SOPState(node=self.node, counters=dict(self.counters))


@dataclass
class StepResult:
    state: SOPState
    violation: bool
    violation_reason: str = ""
    terminal: bool = False


class SOPEngine:
    def __init__(self, path: Path = SOP_PATH):
        with open(path, encoding="utf-8") as f:
            self.graph = json.load(f)
        self.nodes = self.graph["nodes"]
        self.actions = self.graph["actions"]
        self.rules = self.graph.get("rules", {})

    @property
    def all_actions(self) -> list[str]:
        return list(self.actions)

    def is_terminal(self, node: str) -> bool:
        return self.nodes[node]["terminal"]

    def legal_actions(self, state: SOPState) -> list[str]:
        """当前状态下的合法动作集合（action mask），含自环次数上限过滤。"""
        if self.is_terminal(state.node):
            return ["end_session"] if state.node in self.actions["end_session"]["from"] else []
        legal = []
        for name, spec in self.actions.items():
            frm = spec["from"]
            if frm == "any_nonterminal":
                allowed = True
            else:
                allowed = state.node in frm
            if not allowed:
                continue
            max_uses = spec.get("max_uses")
            if max_uses is not None and state.counters.get(name, 0) >= max_uses:
                continue
            legal.append(name)
        # end_session 只在终止节点合法，前面 terminal 分支已处理
        return [a for a in legal if a != "end_session"]

    def _resolve_target(self, action: str, context: dict) -> str | None:
        """解析动作目标节点；分支变量缺失时返回 None。'self' 表示自环。"""
        spec = self.actions[action]
        if "branch_on" in spec:
            val = str(context.get(spec["branch_on"])).lower()
            return spec["to_map"].get(val)
        return spec["to"]

    def step(self, state: SOPState, action: str, context: dict | None = None,
             resync: bool = False) -> StepResult:
        """执行动作。context 提供分支变量（issue_type / evidence_valid）。

        非法动作一律记违规；状态处理分两种口径：
        - 严格（resync=False）：状态不变（一次跳步后续级联违规）
        - 重同步（resync=True）：状态仍跳到该动作的目标节点（每步独立计量）
        """
        context = context or {}
        illegal_reason = ""
        if action not in self.actions:
            return StepResult(state.copy(), True, f"未知动作: {action}",
                              self.is_terminal(state.node))
        if action not in self.legal_actions(state):
            illegal_reason = f"动作 {action} 在节点 {state.node} 不合法"
            if not resync:
                return StepResult(state.copy(), True, illegal_reason,
                                  self.is_terminal(state.node))

        target = self._resolve_target(action, context)
        if target is None:
            return StepResult(state.copy(), True,
                              f"动作 {action} 缺少分支变量 {self.actions[action]['branch_on']}",
                              self.is_terminal(state.node))

        new_state = state.copy()
        if self.actions[action].get("max_uses") is not None:
            new_state.counters[action] = new_state.counters.get(action, 0) + 1
        if target != "self":
            new_state.node = target
        return StepResult(new_state, bool(illegal_reason), illegal_reason,
                          self.is_terminal(new_state.node))

    def check_coupon_amount(self, order_amount: float, coupon_amount: float) -> bool:
        """合规规则4：赔付券金额不得超过订单金额的固定比例。"""
        return coupon_amount <= order_amount * self.rules["coupon_max_ratio"]
