"""标准 UCT 的开环 MCTS。环境接口（鸭子类型）：

- copy_state(state) -> state          模拟前深拷贝
- legal_actions(state) -> list[str]   SOP 硬掩码
- step(state, action) -> (state, terminal: bool, reward: float | None)
                                      terminal 时 reward 为终局奖励
- rollout(state, max_steps) -> float  随机推演至终止/步数上限，返回奖励
"""
import math
import random
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Node:
    action: str | None = None
    parent: "Node | None" = None
    children: dict = field(default_factory=dict)
    untried: list = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    prior: float = 1.0  # PUCT 先验（默认1.0归一化效果）

    @property
    def q(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0


def _uct_select(node: Node, c: float) -> Node:
    log_n = math.log(node.visits)
    return max(node.children.values(),
               key=lambda ch: ch.q + c * math.sqrt(log_n / ch.visits))


def _puct_select(node: Node, c: float) -> Node:
    """PUCT 公式：Q(s,a) + c·P(s,a)·√N(s)/(1+N(s,a))"""
    sqrt_n = math.sqrt(node.visits)
    return max(node.children.values(),
               key=lambda ch: ch.q + c * ch.prior * sqrt_n / (1 + ch.visits))


def search(env, root_state, budget: int = 16, c: float = 1.4,
           max_depth: int = 4, rng: random.Random | None = None,
           prior_fn: Callable[[list[str]], dict[str, float]] | None = None):
    """返回 (最优动作, 统计)。最优动作 = 访问数最多的根子节点。

    prior_fn: 若提供，应返回 {action: prior_prob}；用于 PUCT 搜索。
              若为 None，退化为标准 UCT。
    """
    rng = rng or random.Random()
    root = Node(untried=list(env.legal_actions(root_state)))
    if not root.untried:
        raise ValueError("根状态无合法动作")

    # 提前计算根节点所有动作的先验
    root_prior_dict = None
    if prior_fn is not None:
        root_prior_dict = prior_fn(root.untried)

    # 选择对应的选择函数
    select_fn = _puct_select if prior_fn is not None else _uct_select

    for _ in range(budget):
        state = env.copy_state(root_state)
        node, depth = root, 0
        terminal = False
        # 本次 simulation 的总回报 = 选择/扩展沿途步奖励之和 + 叶子处
        # rollout/终局奖励。terminal 奖励模式下非终止步 reward=None（不累加），
        # dense 模式下每步累加步奖励；无论哪种模式，总回报只在最后回传一次，
        # 不存在重复计算。
        sim_return = 0.0

        # 选择
        while not node.untried and node.children and depth < max_depth:
            node = select_fn(node, c)
            state, terminal, reward = env.step(state, node.action)
            if reward is not None:
                sim_return += reward
            depth += 1
            if terminal:
                break

        # 扩展
        if not terminal and node.untried and depth < max_depth:
            action = node.untried.pop(rng.randrange(len(node.untried)))
            state, terminal, reward = env.step(state, action)
            if reward is not None:
                sim_return += reward
            depth += 1
            child = Node(action=action, parent=node)

            # 设置先验：仅根节点的直接子节点使用 PUCT 先验
            if node == root and root_prior_dict is not None:
                child.prior = root_prior_dict.get(action, 1.0)

            if not terminal:
                child.untried = list(env.legal_actions(state))
            node.children[action] = child
            node = child

        # 模拟（rollout 返回值在 dense 模式下已含沿途步奖励 + 终局/截断奖励）
        if not terminal:
            sim_return += env.rollout(state, max_steps=max_depth - depth)

        # 反向传播（每次 simulation 只回传一次总回报）
        while node is not None:
            node.visits += 1
            node.total_value += sim_return
            node = node.parent

    best_action = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
    stats = {a: {"visits": ch.visits, "q": round(ch.q, 4)}
             for a, ch in root.children.items()}
    return best_action, stats
