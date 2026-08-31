# 电商售后智能客服 Agent：SOP 约束下的 MCTS 决策框架

面向电商**售后场景**（退款 / 退货 / 换货 / 物流异常）的图文混合多轮决策 Agent 研究框架。

## 背景与痛点

电商售后是**图文混合的多轮决策场景**。大模型在平台规则（仅退款条件、举证要求、赔付上限）约束下频繁产生**规则幻觉**——实测基线（单次 LLM + SOP 规则全文入 prompt）严格合规率仅 **25.1%**。本项目将售后规则从"Prompt 里的软约束"升级为"架构层的硬约束"，在合规保证下做多轮策略搜索。

> 定位：实习期间基于业务需求延伸的个人研究方向 / LLM Agent 方向研究项目。已在 60 条合成测试集上完成端到端闭环与多组消融。

---

## 技术方案与创新

**1. 架构级合规约束**
将售后规则建模为 **12 状态、15 动作**的 SOP 状态图，运行时编译为**动态动作掩码**——非法动作根本不进入决策空间。合规性由此从 *Prompt 约束* 升级为 *架构硬约束*。

**2. MCTS 策略搜索与奖励设计**
以 **LLM 用户模拟器**作为环境模型，在掩码空间内做多轮「状态-动作」推演；**PUCT 引入规则先验**引导搜索。针对稀疏的终点奖励，设计 **PBRS 势能塑形**与**过程奖励模型（PRM）**，并完成 `dense / PBRS / terminal` **三方案消融**。

**3. 多模态状态定位**
融合**用户文本 + 商品/物流图片 + 近期行为流**三路输入，基于 **Qwen-VL** 定位当前 SOP 节点并判定图片**证据有效性**，结果直接作为搜索的**分支条件**（`branch_ctx`）。

**4. 评测闭环与成本优化**
自建**双口径**（合规 / 完成）自动化评测流水线，支持批量 rollout、难度分层与逐例错因归因；推演（rollout）环节**下沉本地小模型**（Ollama + Qwen2.5-1.5B）。

---

## 阶段成果

| 指标 | 结果 |
|------|------|
| **严格 SOP 合规率** | 25.1% → **100%**（零越权动作） |
| **终点符合率** | 71.2% → **85.7%**（经 PUCT 规则先验调优） |
| **感知层四分类准确率** | **95%**（19/20，Qwen-VL zero-shot） |
| **图片证据判定准确率** | **100%**（has_evidence） |
| **任务完成率** | 100%（保持不劣化） |
| **本地小模型 rollout** | 效果持平，**API 成本 0** |

- 评测双口径：**严格口径**（违规不推进状态）与**重同步口径**（违规也跳转、逐步独立计量）。
- 合规率消融验证了 SOP 掩码的必要性：去掉掩码后合规率回落至近似基线水平。
- 多模态端到端样例：潮湿坚果实物照 → 感知判定 `return / 证据有效` → MCTS 9 轮到达终止节点、**零违规**，证明感知输出正确驱动搜索分支。

---

## 方法概览

```
用户输入（文本 + 图片 + 行为流）
        │
        ▼
  多模态感知层 (Qwen-VL, zero-shot)  ──►  issue_type / has_evidence  ──► 搜索分支条件
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │  每个真实对话轮：                              │
  │    MCTS 搜索（SOP 动态掩码下的合法动作空间）    │
  │      PUCT = Q + c·P·√N / (1+n)   ← 规则先验 P  │
  │      → 最优动作 a* → LLM 渲染回复              │
  │      → LLM 用户模拟器回应 → 下一轮             │
  └─────────────────────────────────────────────┘
        │
        ▼
  奖励塑形： terminal ｜ PBRS 势能塑形 ｜ dense(PRM)  ← 三方案消融
```

- **SOP 引擎**：12 状态 / 15 动作，硬性合规规则（未核实订单不得给方案、退货换货必须证据核验、赔付金额上限等），非法动作经 action mask 直接屏蔽。
- **MCTS**：开环搜索，PUCT 以规则先验作为 P 项；领域知情的 rollout 默认策略缓解稀疏奖励。
- **奖励**：`R = w1·任务完成 + w2·用户满意 − w3·轮数成本`，支持 terminal / PBRS / dense 三种模式切换用于消融。

---

## 目录结构

```
.
├── src/                      # 核心代码
│   ├── sop.py                # SOP 状态图引擎：动态动作掩码 / 转移 / 合规检查
│   ├── mcts.py               # MCTS 核心（UCT / PUCT 规则先验）
│   ├── mcts_env.py           # 搜索环境：SOP 引擎 + 用户模拟器 + 奖励
│   ├── mcts_agent.py         # MCTS Agent（每轮搜索选动作，LLM 只渲染回复）
│   ├── baseline_agent.py     # 基线 Agent（单次 LLM，SOP 全文入 prompt）
│   ├── reward.py             # 奖励函数：terminal / PBRS 势能塑形 / dense(PRM)
│   ├── perception.py         # 多模态感知层（图+文+行为 → SOP 分流字段）
│   ├── user_simulator.py     # LLM 用户模拟器（人设 + 满意度信号，作环境模型）
│   ├── runner.py             # 对话运行器（批量 rollout，落盘轨迹）
│   ├── evaluate.py           # 双口径评测：合规率 / 完成率 / 轮数 / 分场景
│   ├── mm_dataset.py         # 多模态样本 loader
│   ├── mm_eval.py            # 感知评估（准确率 / P-R / 混淆矩阵 / 错因）
│   ├── llm.py                # DeepSeek 客户端（OpenAI 兼容）
│   ├── llm_vl.py             # DashScope Qwen-VL 客户端
│   └── local_llm.py          # 本地小模型接口（Ollama，rollout 成本下沉）
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

### 3. 运行与评测

```bash
# 单元测试
pytest

# 基线 vs SOP-MCTS（全量 60 条）
python src/runner.py               --workers 6 --out runs/baseline
python src/runner.py --agent mcts  --workers 6 --out runs/mcts_60
python src/evaluate.py runs/mcts_60          # 双口径指标 + 分场景

# 奖励塑形消融（terminal / PBRS / dense）
python src/runner.py --agent mcts --reward-mode pbrs  --out runs/mcts_pbrs
python src/runner.py --agent mcts --reward-mode dense --out runs/mcts_dense

# 单条冒烟
python src/runner.py --agent mcts --cases exchange_01 --out runs/smoke
```

### 4.（可选）本地小模型零成本 rollout

将 MCTS rollout 从云模型下沉到本地 `qwen2.5:1.5b`，推演环节 API 成本降为 0：

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
| 0 | 环境 + SOP 引擎 + 基线 + 双口径评测体系（60 条合成测试集） | ✅ |
| 1 | SOP 约束下的 MCTS 核心（UCT + 领域知情 rollout） | ✅ |
| 2 | 多模态状态定位（Qwen-VL zero-shot）+ 端到端接入 | ✅ |
| 3/4 | 全量评测 + 消融（mask / 奖励塑形 / 单模态 vs 多模态） | ✅ |
| 扩展 | PUCT 规则先验、本地 rollout 下沉、过程奖励模型（PRM）可行性验证 | 🔬 |

各阶段技术方案、裁决记录与结果数字详见根目录 `阶段*方案_*.md` 与 `PRM方案_过程奖励模型.md`。

---

## 技术栈

Python · OpenAI 兼容 SDK（DeepSeek / DashScope Qwen-VL）· Ollama + Qwen2.5-1.5B（本地 rollout）· pytest。核心 MCTS / SOP 引擎为纯 Python 实现，无重型框架依赖。
