# 本地小模型 Rollout 实现总结

## 实现完成状态

### 1. 新建 src/local_llm.py ✓
**文件路径：** `/Users/liujiaojiao/Downloads/电商智能体框架/src/local_llm.py`

**实现内容：**
- 封装 ollama HTTP API（http://localhost:11434/api/chat）
- `chat(messages, temperature, timeout=30, retries=3)` 主要接口
- 支持 qwen2.5:1.5b 模型
- 错误处理：
  - 连接失败检测（LocalLLMError 异常）
  - 超时保护（30秒默认超时）
  - 自动重试（最多3次）
  - JSON 解析失败处理
- 与 llm.py 保持一致的错误接口（LocalLLMError）

**关键特性：**
- 非阻塞设计：若 requests 库未安装，会给出清晰的安装提示
- Ollama 不可用时给出具体指引："请确保已运行 'ollama serve'"
- 请求超时 30 秒，适合 1.5B 模型的推理耗时

---

### 2. 修改 src/user_simulator.py ✓
**修改行数：** 40, 83

**修改内容：**
```python
# __init__ 方法增加参数
def __init__(self, case: dict, chat_fn=None):
    self.case = case
    self.chat_fn = chat_fn or llm.chat_json  # 新增
    ...

# respond 方法使用 self.chat_fn
out = self.chat_fn(messages, temperature=0.8)  # 之前是 llm.chat_json
```

**功能说明：**
- 默认行为保持不变（使用 llm.chat_json 即云模型）
- 支持注入自定义 chat_fn（如本地 local_llm.chat）
- 完全向后兼容：不传 chat_fn 参数时与原代码等价

---

### 3. 修改 src/mcts_env.py ✓
**修改行数：** 65, 127-149

**修改内容：**
```python
# __init__ 增加 rollout_sim 参数
def __init__(self, engine: SOPEngine, simulator, branch_ctx: dict,
             reward_cfg: RewardConfig = RewardConfig(),
             rng: random.Random | None = None, 
             rollout_sim=None):  # 新增参数
    self.engine = engine
    self.sim = simulator
    self.rollout_sim = rollout_sim  # 新增
    ...

# rollout 方法改进
def rollout(self, s: SimState, max_steps: int) -> float:
    # 选择要使用的模拟器
    sim_to_use = self.rollout_sim if self.rollout_sim is not None else self.sim
    
    # 临时替换 self.sim
    original_sim = self.sim
    self.sim = sim_to_use
    try:
        # 原有逻辑（使用 sim_to_use）
        ...
    finally:
        # 恢复原来的 self.sim
        self.sim = original_sim
```

**关键设计：**
- rollout_sim 为 None 时自动回退到 self.sim（向后兼容）
- 使用 try-finally 保证模拟器的正确恢复
- step() 方法在 rollout() 中会使用临时替换的模拟器
- PUCT 先验逻辑保持不变（只影响 rollout）

---

### 4. 修改 src/mcts_agent.py ✓
**修改行数：** 11-21, 35-65

**修改内容：**
```python
# 可选导入本地 LLM
try:
    import local_llm
except ImportError:
    local_llm = None

class MCTSAgent:
    def __init__(self, case: dict, engine: SOPEngine | None = None,
                 budget: int = 16, max_depth: int = 8, uct_c: float = 1.4,
                 branch_ctx: dict | None = None, 
                 use_local_rollout: bool = False):  # 新增参数
        """
        Args:
            use_local_rollout: 若为 True，rollout 使用本地小模型；
                              若为 False，rollout 使用云模型（默认行为）。
        """
        ...
        # 构造 rollout_sim（若启用本地 rollout）
        rollout_sim = None
        if use_local_rollout:
            if local_llm is None:
                raise RuntimeError("使用本地 rollout 需要 local_llm 模块可用")
            rollout_sim = UserSimulator(case, chat_fn=local_llm.chat)
        
        self.env = SearchEnv(self.engine, UserSimulator(case), branch_ctx,
                             rng=random.Random(seed + 1), rollout_sim=rollout_sim)
```

**功能说明：**
- `use_local_rollout=False`（默认）：保持原有行为，rollout_sim=None
- `use_local_rollout=True`：创建额外的 rollout 模拟器，使用本地 LLM
- 清晰的错误提示：若本地 LLM 导入失败，会告知具体原因
- 树内搜索和动作渲染仍使用云模型（DeepSeek），只有 rollout 用本地模型

---

### 5. 修改 src/runner.py ✓
**修改行数：** 94-95, 111

**修改内容：**
```python
# argparse 增加标志
parser.add_argument("--use-local-rollout", action="store_true", default=False,
                    help="MCTS agent 使用本地小模型进行 rollout")

# make_agent 函数传入参数
def make_agent(case):
    if args.agent == "mcts":
        return MCTSAgent(case, engine, use_local_rollout=args.use_local_rollout)
    return BaselineAgent()
```

**命令行使用：**
```bash
# 使用本地 rollout
python src/runner.py --agent mcts --use-local-rollout --out runs/mcts_local_60

# 不使用本地 rollout（默认）
python src/runner.py --agent mcts --out runs/mcts_puct_60
```

---

## 验证清单

### 代码验证 ✓
- [x] local_llm.py 新建完成
- [x] user_simulator.py 的 __init__ 和 respond 已修改
- [x] mcts_env.py 的 __init__ 和 rollout 已修改
- [x] mcts_agent.py 的导入和 __init__ 已修改
- [x] runner.py 的 argparse 和 make_agent 已修改

### 代码质量检查 ✓
- [x] 向后兼容性：所有修改都支持默认行为回退
- [x] 错误处理：关键路径都有异常捕获和清晰的错误提示
- [x] 资源管理：rollout() 中使用 try-finally 保证模拟器恢复
- [x] 接口一致性：local_llm.chat 与 llm.chat_json 都接受 temperature 参数

### 集成点检查 ✓
- [x] local_llm.chat 在 UserSimulator 中的使用
- [x] UserSimulator 在 MCTSAgent 中的构造
- [x] rollout_sim 在 SearchEnv 中的使用和回退
- [x] use_local_rollout 在 runner.py 中的透传

---

## 实施后的执行流程

### 不使用本地 Rollout 的流程（原始行为）
```
MCTSAgent(case, use_local_rollout=False)
  └─> SearchEnv(sim=UserSimulator(case), rollout_sim=None)
      └─> rollout() 中，sim_to_use = self.sim（云模型）
          └─> 所有回应都调用 llm.chat_json
```

### 使用本地 Rollout 的流程（新增功能）
```
MCTSAgent(case, use_local_rollout=True)
  └─> SearchEnv(sim=UserSimulator(case), rollout_sim=UserSimulator(case, chat_fn=local_llm.chat))
      └─> rollout() 中，sim_to_use = self.rollout_sim（本地模型）
          └─> rollout 中的回应调用 local_llm.chat
  └─> act() 中，回复渲染仍然调用 llm.chat（云模型）
```

---

## 使用指南

### 前置条件
1. 安装 requests 库：
   ```bash
   pip install requests
   ```

2. 启动 Ollama 服务：
   ```bash
   ollama serve
   ```

3. 拉取模型：
   ```bash
   ollama pull qwen2.5:1.5b
   ```

### 运行命令

**单条冒烟测试（exchange_01）：**
```bash
cd /Users/liujiaojiao/Downloads/电商智能体框架
python src/runner.py --agent mcts --use-local-rollout --cases exchange_01 --out runs/test_local_1
```

**60 条全量评估（本地 Rollout）：**
```bash
python src/runner.py --agent mcts --use-local-rollout --workers 6 --out runs/mcts_local_60
```

**60 条全量评估（云 Rollout，对比基线）：**
```bash
python src/runner.py --agent mcts --workers 6 --out runs/mcts_puct_60
```

**生成评估报告：**
```bash
python src/evaluate.py runs/mcts_local_60
python src/evaluate.py runs/mcts_puct_60
```

---

## 预期效果与成本

### 成本对比
| 版本 | 树内调用 | Rollout 调用 | 云模型成本 | 本地成本 |
|------|--------|-----------|---------|--------|
| PUCT 云版本 | ~100 | ~100 | ¥0.8 | ¥0 |
| 本地 Rollout | ~100 | ~100 | ¥0.8 | ¥0 |
| **成本差异** | - | - | 无 | **零成本** |

### 预期指标变化
- **成本：** 从 ¥0.8/条 降至 ¥0.8/条（云模型成本不变，但 Rollout 不再产生成本）
- **整体成本：** 依然取决于树内搜索（Agent 决策层）的模型调用
- **完成率 / 终点符合率：** 预期略有下降（因为 1.5B 模型弱于 DeepSeek），但仍在可接受范围

### 输出指标
验证完成后，应对比以下指标：
- 终点符合率（平均每条对话的符合率）
- 完成率（成功终止的对话比例）
- 平均轮数（对话长度）
- 总成本（¥0 for rollout）

---

## 后续扩展点

1. **模型选择：** 可在 local_llm.py 中增加 model_name 参数，支持不同的 Ollama 模型
2. **缓存优化：** rollout() 可复用相同的模拟器实例，减少内存开销
3. **参数调优：** 可增加 rollout_depth 等参数来平衡效率和效果
4. **多模型混用：** 树内用 DeepSeek，Rollout 用本地模型，进一步降低成本

---

## 文件修改摘要

| 文件 | 操作 | 关键改动 |
|------|------|--------|
| src/local_llm.py | 新建 | Ollama HTTP API 封装 |
| src/user_simulator.py | 修改 | +chat_fn 参数，默认 llm.chat_json |
| src/mcts_env.py | 修改 | +rollout_sim 参数，rollout() 改进 |
| src/mcts_agent.py | 修改 | +use_local_rollout 参数，rollout_sim 构造 |
| src/runner.py | 修改 | +--use-local-rollout 标志 |

**总行数变化：** ~200 行新增 + 修改

---

## 验证步骤（待执行）

### 步骤1：黑箱连通性测试
```bash
python3 -c "from src.local_llm import chat; result = chat([{'role':'user','content':'你好'}]); print('OK' if result else 'FAIL')"
```

### 步骤2：单条冒烟测试
```bash
python src/runner.py --agent mcts --use-local-rollout --cases exchange_01 --out runs/mcts_local_1
```

### 步骤3：全量评估对比
```bash
python src/runner.py --agent mcts --use-local-rollout --workers 6 --out runs/mcts_local_60
python src/runner.py --agent mcts --workers 6 --out runs/mcts_puct_60
python src/evaluate.py runs/mcts_local_60
python src/evaluate.py runs/mcts_puct_60
```

---

## 注意事项

1. **本地模型速度：** qwen2.5:1.5b 在 CPU 上推理较慢，建议使用 GPU 加速
2. **内存占用：** rollout_sim 会额外占用内存，建议在 workers <= 4 时使用
3. **超时配置：** 若本地模型推理超过 30 秒，需要调整 local_llm.py 中的 timeout 参数
4. **模型预热：** 第一次调用本地模型会加载模型，耗时较长，属于正常现象

---

**实现完成时间：** 2026-07-13
**实施状态：** ✓ 已完成所有代码修改
**可用性：** 等待验证步骤执行
