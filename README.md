# 电商售后智能客服 Agent：SOP 约束下的 MCTS 决策框架

面向电商**售后场景**（退款 / 退货 / 换货 / 物流异常）的智能客服 Agent 研究框架。核心思想是把业务 SOP 建模为**状态图 + 动作掩码**，用 **MCTS（蒙特卡洛树搜索）** 在硬合规约束下做多轮对话决策，并用**多模态感知层**从「文本 + 图片 + 行为流」推断问题类型，驱动搜索分支。

> 定位：LLM Agent 方向的研究/实验项目。已在合成测试集上完成阶段 0–2 的完整闭环，并扩展了 PUCT 先验、本地小模型 rollout、过程奖励模型（PRM）等探索。

---

## 核心结果

| 指标 | 基线（单次 LLM） | SOP-MCTS | 变化 |
|------|-----------------|----------|------|
| **严格 SOP 合规率** | 25.1% | **100.0%** | **+74.9pp** |
| 任务完成率 | 100% | 100% | 持平 |
| 平均对话轮数 | 6.45 | ~6.25 | 基本持平 |
| **多模态感知准确率**（issue_type） | — | **95.0%** | 超 85% 目标 |
| 证据判定准确率（has_evidence） | — | **100.0%** | — |

- **核心 claim**：在合规率从 25% 提升到 100% 的同时，任务完成率与对话轮数几乎不受影响——合规提升不以体验为代价。
- 评估采用**双口径**：严格口径（违规不推进状态）与重同步口径（违规也跳转、逐步独立计量）。
- 消融①（去掉 SOP mask）验证了掩码约束的必要性；多模态端到端样例（潮湿坚果实物照 → 感知 → MCTS 9 轮到达终止节点、零违规）验证了感知层正确驱动搜索分支。

---

## 方法概览

```
用户输入（文本 + 图片 + 行为流）
        │
        ▼
  多模态感知层 (Qwen-VL, zero-shot)  ──►  issue_type / has_evidence
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │  每个真实对话轮：                              │
  │    MCTS 搜索（SOP 硬掩码下的合法动作空间）      │
  │      → 最优动作 a*                            │
  │      → LLM 渲染成自然语言回复                  │
  │      → 用户模拟器回应 → 下一轮                 │
  └─────────────────────────────────────────────┘
        │
        ▼
  终局奖励 R = w1·任务完成 + w2·用户满意 − w3·轮数成本
```

- **SOP 引擎**：12 个状态节点、15 个动作、硬性合规规则（未核实订单不得给方案、退货换货必须证据核验、赔付金额上限等）。非法动作通过 action mask 根本不进搜索树。
- **MCTS**：开环 UCT（`Q + c·√(ln N / n)`），领域知情的 default policy（rollout 偏向推进动作以缓解稀疏奖励）。已扩展 PUCT 先验版本。
- **奖励**：以终局奖励为主（S10 解决=1.0，S11 升级=0.4，未终止=0），叠加满意度与轮数成本。

---

## 目录结构

```
.
├── src/                      # 核心代码
│   ├── sop.py                # SOP 状态图引擎：动作掩码 / 转移 / 合规检查
│   ├── mcts.py               # 标准 UCT 开环 MCTS
│   ├── mcts_env.py           # 搜索环境：SOP 引擎 + 用户模拟器 + 奖励
│   ├── mcts_agent.py         # MCTS Agent（每轮搜索选动作，LLM 只渲染回复）
│   ├── baseline_agent.py     # 基线 Agent（单次 LLM，SOP 全文入 prompt）
│   ├── reward.py             # 多因素终局奖励函数
│   ├── perception.py         # 多模态感知层（图+文+行为 → SOP 分流字段）
│   ├── user_simulator.py     # LLM 用户模拟器（人设 + 满意度信号）
│   ├── runner.py             # 对话运行器（Agent vs 模拟器，落盘轨迹）
│   ├── evaluate.py           # 评估：合规率 / 完成率 / 轮数 / 分场景
│   ├── mm_dataset.py         # 多模态样本 loader
│   ├── mm_eval.py            # 感知评估（准确率 / P-R / 混淆矩阵）
│   ├── llm.py                # DeepSeek 客户端（OpenAI 兼容）
│   ├── llm_vl.py             # DashScope Qwen-VL 客户端
│   └── local_llm.py          # 本地小模型接口（Ollama HTTP，用于零成本 rollout）
├── scripts/                  # 数据生成与评估脚本
├── sop/sop_graph.json        # SOP 状态图定义
├── tests/                    # pytest 单元测试
├── data generation/          # SOP / MCTS 轨迹数据生成
├── 阶段0~34方案_*.md          # 各阶段技术方案与结果记录
├── PRM方案_过程奖励模型.md     # 过程奖励模型方案
├── requirements.txt
└── pytest.ini
```

> 说明：数据集 `data/`（多模态样本图片、测试用例）与实验输出 `runs/` 未纳入版本库，按需在本地生成。

---

## 快速开始

### 1. 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置密钥

在项目根目录创建 `.env`（**不入库**）：

```
DEEPSEEK_API_KEY=your_deepseek_key      # 基线 Agent / 用户模拟器 / MCTS 决策
DASHSCOPE_API_KEY=your_dashscope_key    # 多模态感知（Qwen-VL）
```

### 3. 运行

```bash
# 单元测试
pytest

# 基线：全量测试集
python src/runner.py --out runs/baseline --workers 6
python src/evaluate.py runs/baseline

# SOP-MCTS：全量测试集
python src/runner.py --agent mcts --workers 6 --out runs/mcts_60
python src/evaluate.py runs/mcts_60

# 单条冒烟
python src/runner.py --agent mcts --cases exchange_01 --out runs/smoke
```

### 4.（可选）本地小模型零成本 rollout

将 MCTS rollout 从云模型切到本地 `qwen2.5:1.5b`，rollout 阶段成本降为 0：

```bash
brew install ollama && ollama serve &
ollama pull qwen2.5:1.5b
python verify_local_rollout.py                       # 环境自检
python src/runner.py --agent mcts --use-local-rollout --workers 6 --out runs/mcts_local_60
```

---

## 阶段进展

| 阶段 | 内容 | 状态 |
|------|------|------|
| 0 | 环境 + SOP 引擎 + 基线 + 双口径评估体系（60 条合成测试集） | ✅ 完成 |
| 1 | SOP 约束下的 MCTS 核心（UCT + 领域知情 rollout） | ✅ 完成 |
| 2 | 多模态状态识别（Qwen-VL zero-shot）+ 端到端接入 | ✅ 完成 |
| 3/4 | 全量评估 + 消融（mask / 权重 / 单模态 vs 多模态） | ✅ 主体完成 |
| 扩展 | PUCT 先验、本地 rollout、过程奖励模型（PRM）可行性验证 | 🔬 探索中 |

各阶段的技术方案、裁决记录与结果数字详见根目录对应的 `阶段*方案_*.md` 与 `PRM方案_过程奖励模型.md`。

---

## 技术栈

Python · OpenAI 兼容 SDK（DeepSeek / DashScope Qwen-VL）· Ollama（本地小模型）· pytest。核心 MCTS / SOP 引擎为纯 Python 实现，无重型框架依赖。
