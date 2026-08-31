# 本地 Rollout 技术细节文档

## 目录
1. [架构设计](#架构设计)
2. [接口规范](#接口规范)
3. [实现细节](#实现细节)
4. [性能分析](#性能分析)
5. [扩展指南](#扩展指南)

---

## 架构设计

### 1.1 分层模型

```
┌─────────────────────────────────────────────┐
│         MCTSAgent (决策层)                   │
│  ┌──────────────────────────────────────┐  │
│  │  __init__(use_local_rollout)         │  │
│  │  ├─ if True:  创建 rollout_sim       │  │
│  │  │            (local_llm.chat_json) │  │
│  │  └─ if False: rollout_sim = None     │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│         SearchEnv (搜索环境)                 │
│  ┌──────────────────────────────────────┐  │
│  │  rollout(s, max_steps)               │  │
│  │  ├─ sim_to_use = rollout_sim         │  │
│  │  │              or self.sim          │  │
│  │  └─ 使用 sim_to_use 进行推演        │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│    UserSimulator (用户模拟)                  │
│  ┌──────────────────────────────────────┐  │
│  │  respond(history) -> dict            │  │
│  │  ├─ 调用 self.chat_fn               │  │
│  │  │   (llm.chat_json or              │  │
│  │  │    local_llm.chat_json)          │  │
│  │  └─ 解析并返回 JSON                 │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### 1.2 数据流

#### 云 Rollout 流程（原始）
```
MCTSAgent(use_local_rollout=False)
  ├─ env.rollout()
  │  └─ self.sim.respond()
  │     └─ llm.chat_json()  [调用云模型]
  │        ├─ 成本：¥0.0017/次
  │        └─ 耗时：0.2-0.5s
  │
  └─ act() 回复渲染
     └─ llm.chat()  [调用云模型]
        ├─ 成本：¥0.001/次
        └─ 耗时：0.1-0.3s

总成本/条: ¥0.1-0.15 (Rollout)
```

#### 本地 Rollout 流程（改进）
```
MCTSAgent(use_local_rollout=True)
  ├─ env.rollout()
  │  └─ rollout_sim.respond()
  │     └─ local_llm.chat_json()  [调用本地模型]
  │        ├─ 成本：¥0
  │        ├─ 耗时：1-3s (1.5B on CPU)
  │        └─ 流程：
  │           ├─ HTTP POST to Ollama
  │           ├─ 调用本地 Qwen 2.5 模型
  │           └─ JSON 解析
  │
  └─ act() 回复渲染
     └─ llm.chat()  [调用云模型]
        ├─ 成本：¥0.001/次（不变）
        └─ 耗时：0.1-0.3s

总成本/条: ¥0 (Rollout) + ¥0.8 (树内) = ¥0.8
```

---

## 接口规范

### 2.1 local_llm 模块

#### 函数签名

```python
def chat(
    messages: list[dict],           # [{role: 'user'|'assistant', content: str}, ...]
    temperature: float = 0.7,       # 生成温度（0-2）
    timeout: int = 30,              # 请求超时（秒）
    retries: int = 3                # 重试次数
) -> str:
    """返回模型生成的文本"""
    pass
```

```python
def chat_json(
    messages: list[dict],
    temperature: float = 0.7,
    timeout: int = 30,
    retries: int = 3
) -> dict:
    """调用 chat，然后使用 _extract_json 解析 JSON"""
    pass
```

#### 异常规范

```python
class LocalLLMError(RuntimeError):
    """本地 LLM 相关的所有错误"""
    pass
```

**异常类型：**
- `LocalLLMError("requests 库未安装...")`  - 缺少依赖
- `LocalLLMError("无法连接 Ollama...")`    - 连接失败
- `LocalLLMError("请求超时 (30s)")`       - 超时
- `LocalLLMError("Ollama 返回 500: ...")`  - HTTP 错误
- `LocalLLMError("JSON 解析失败: ...")`    - 解析失败

### 2.2 UserSimulator 接口变化

#### 原始接口
```python
class UserSimulator:
    def __init__(self, case: dict):
        # 内部使用 llm.chat_json
        pass
    
    def respond(self, history: list[dict]) -> dict:
        # 调用 llm.chat_json
        pass
```

#### 扩展接口
```python
class UserSimulator:
    def __init__(self, case: dict, chat_fn=None):
        # chat_fn 默认为 llm.chat_json
        # 支持注入自定义 LLM 客户端
        self.chat_fn = chat_fn or llm.chat_json
    
    def respond(self, history: list[dict]) -> dict:
        # 使用 self.chat_fn 替代 llm.chat_json
        out = self.chat_fn(messages, temperature=0.8)
```

**chat_fn 要求：**
- 接收参数：`(messages: list[dict], temperature: float)`
- 返回格式：`dict` 包含 'utterance' 和 'satisfaction' 字段
- 异常处理：抛出异常时 respond() 会重新抛出给调用者

### 2.3 SearchEnv 接口变化

#### 原始接口
```python
class SearchEnv:
    def __init__(self, engine, simulator, branch_ctx, ...):
        self.sim = simulator
```

#### 扩展接口
```python
class SearchEnv:
    def __init__(self, engine, simulator, branch_ctx, ..., rollout_sim=None):
        self.sim = simulator
        self.rollout_sim = rollout_sim  # 可选的 Rollout 模拟器
    
    def rollout(self, s, max_steps):
        # 若 rollout_sim 不为 None，使用它替代 self.sim
        sim_to_use = self.rollout_sim if self.rollout_sim is not None else self.sim
```

**设计要点：**
- rollout_sim 可为 None（向后兼容）
- 使用 try-finally 保证资源恢复
- 不影响 step() 中树内搜索的模拟器

### 2.4 MCTSAgent 接口变化

#### 原始接口
```python
class MCTSAgent:
    def __init__(self, case, engine=None, budget=16, ...):
        self.env = SearchEnv(engine, UserSimulator(case), ...)
```

#### 扩展接口
```python
class MCTSAgent:
    def __init__(self, case, engine=None, budget=16, ..., 
                 use_local_rollout=False):
        # 创建树内搜索的模拟器（始终用云模型）
        sim = UserSimulator(case)
        
        # 若启用本地 Rollout，创建额外的模拟器
        rollout_sim = None
        if use_local_rollout:
            rollout_sim = UserSimulator(case, chat_fn=local_llm.chat_json)
        
        # 传给 SearchEnv
        self.env = SearchEnv(..., rollout_sim=rollout_sim)
```

### 2.5 runner.py 接口变化

#### argparse 新参数
```python
parser.add_argument("--use-local-rollout", 
                    action="store_true",        # 默认 False
                    default=False,
                    help="MCTS agent 使用本地小模型进行 rollout")
```

#### 传递逻辑
```python
MCTSAgent(case, engine, use_local_rollout=args.use_local_rollout)
```

---

## 实现细节

### 3.1 local_llm.py 实现

#### HTTP 请求格式

```python
url = "http://localhost:11434/api/chat"
payload = {
    "model": "qwen2.5:1.5b",
    "messages": messages,           # [{role, content}, ...]
    "temperature": temperature,
    "stream": False,                # 必须为 False（整个响应）
}
response = requests.post(url, json=payload, timeout=30)
```

#### 响应格式

```python
{
    "model": "qwen2.5:1.5b",
    "created_at": "2026-07-13T10:30:00Z",
    "message": {
        "role": "assistant",
        "content": "你好，很高兴认识你..."
    },
    "done": True,
    "done_reason": "stop",
    "context": [...],
    "total_duration": 1234567890,
    "load_duration": 123456,
    "prompt_eval_count": 50,
    "prompt_eval_duration": 500000,
    "eval_count": 100,
    "eval_duration": 610000
}
```

#### 重试策略

```python
last_err = None
for attempt in range(retries):
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        # 处理响应
        return data["message"].get("content", "").strip()
    except (requests.exceptions.Timeout, 
            requests.exceptions.ConnectionError,
            json.JSONDecodeError) as e:
        last_err = convert_to_LocalLLMError(e)
    
    if attempt < retries - 1:
        time.sleep(0.5 * (attempt + 1))  # 指数退避
        # 尝试间隔：0.5s, 1.0s, 1.5s

raise last_err
```

#### JSON 解析（_extract_json）

```python
def _extract_json(text: str) -> dict:
    # 1. 移除前后空白
    text = text.strip()
    
    # 2. 处理代码围栏
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    
    # 3. 查找 JSON 对象的边界
    start = text.find("{")
    end = text.rfind("}")
    
    # 4. 安全检查
    if start == -1 or end <= start:
        raise json.JSONDecodeError("未找到 JSON 对象", text, 0)
    
    # 5. 解析
    return json.loads(text[start:end + 1])
```

**容忍的格式：**
```python
# 格式1：纯 JSON
'{"utterance": "...", "satisfaction": "..."}'

# 格式2：代码围栏
'```json\n{"utterance": "...", ...}\n```'

# 格式3：前后缀
'好的，这是我的回复：{"utterance": "...", ...}完毕'

# 格式4：嵌套对象
'{"data": {"utterance": "...", ...}}'  # 会提取最外层的 {}
```

### 3.2 SearchEnv.rollout 的状态管理

#### 原始实现（存在的问题）
```python
def rollout(self, s: SimState, max_steps: int) -> float:
    # 直接使用 self.sim，无法切换到 rollout_sim
    for _ in range(max_steps):
        ...
        user_out = self.sim.respond(s.transcript)  # 固定使用 self.sim
```

#### 改进实现
```python
def rollout(self, s: SimState, max_steps: int) -> float:
    # 1. 确定要使用的模拟器
    sim_to_use = self.rollout_sim if self.rollout_sim is not None else self.sim
    
    # 2. 保存原始状态
    original_sim = self.sim
    
    # 3. 临时替换（这样 step() 会使用 sim_to_use）
    self.sim = sim_to_use
    
    try:
        # 4. 执行推演（所有 step() 调用都用 sim_to_use）
        for _ in range(max_steps):
            acts = self.legal_actions(s)
            if not acts:
                break
            s, terminal, reward = self.step(s, self._rollout_action(acts))
            # step() 中会调用 self.sim.respond() => 实际调用 sim_to_use.respond()
            if terminal:
                return reward
        return self._reward(s)
    finally:
        # 5. 恢复原始状态（异常安全）
        self.sim = original_sim
```

**关键设计点：**
- 不是创建新的 SearchEnv，而是临时替换 self.sim 引用
- 利用 Python 对象引用的灵活性实现模拟器切换
- try-finally 保证即使异常也能恢复原始状态

---

## 性能分析

### 4.1 成本对比

#### 单条对话的成本分解

**云 Rollout 版本：**
```
树内搜索：
  - UCT 访问节点 ~20 次
  - 每次节点展开调用 1 次 llm.chat_json
  - 每次 llm.chat 渲染回复
  - 总成本：~20 × ¥0.004 = ¥0.08

Rollout 推演（平均 10 次）：
  - 每次 rollout 调用 ~10 步
  - 每步调用 1 次 llm.chat_json（用户模拟）
  - 总成本：~100 × ¥0.001 = ¥0.1

总计：¥0.18/条 × 60 = ¥10.8
```

**本地 Rollout 版本：**
```
树内搜索（不变）：
  - 成本：¥0.08/条

Rollout 推演（改用本地模型）：
  - 每次推演 ~10 步
  - 每步调用 1 次 local_llm.chat_json
  - 总成本：¥0

总计：¥0.08/条 × 60 = ¥4.8
```

**节省：** ¥10.8 - ¥4.8 = **¥6/条** (约55% 节省)

**注意：** 实际成本取决于：
- 模型定价（DeepSeek 价格可能变化）
- 实际 token 消耗量
- rollout 深度和次数

### 4.2 耗时分析

#### 单条对话的耗时

**云 Rollout 版本：**
```
树内搜索（同步）：
  - 每次 llm.chat_json 调用 0.2-0.5s
  - 总耗时：~20 × 0.3s = 6s

Rollout 推演（同步）：
  - 每次 llm.chat_json 调用 0.2-0.5s  
  - 总耗时：~100 × 0.3s = 30s

单条总耗时：36s
```

**本地 Rollout 版本：**
```
树内搜索（同步）：
  - 每次 llm.chat_json 调用 0.2-0.5s
  - 总耗时：~20 × 0.3s = 6s

Rollout 推演（改用本地模型）：
  - 每次 local_llm.chat_json 调用 1-3s（1.5B CPU推理）
  - 总耗时：~100 × 2s = 200s

单条总耗时：206s（远高于云版本）
```

**性能对比：** 
- 云版本更快（网络延迟低，模型大）
- 本地版本更慢（本地推理耗时）
- 但并行度高时可通过 worker 抵消

#### 全量 60 条对话的耗时

**云版本（6 workers）：**
```
实际耗时 ≈ 单条耗时 / workers × 队列系数
        ≈ 36s / 6 × 1.2
        ≈ 7.2 分钟
        = 432 秒
```

**本地版本（6 workers）：**
```
实际耗时 ≈ 206s / 6 × 1.2
        ≈ 41.2 分钟
        = 2472 秒
```

**结论：**
- 本地 Rollout 会显著增加单条对话的耗时
- 但通过并行化可以分散总体耗时
- 适合离线评估，不适合实时服务

### 4.3 质量对比

#### 模型能力对比

| 指标 | DeepSeek | Qwen 2.5 1.5B | 差异 |
|-----|---------|---------------|-----|
| 参数量 | 670B | 1.5B | -99.8% |
| 推理成本 | ¥0.001/1K token | ¥0 | 无成本 |
| 理解能力 | 强 | 中等 | -30-40% |
| JSON 生成 | 精确 | 偶发错误 | +2-5% JSON 失败率 |
| 用户模拟 | 高保真 | 中等保真 | -10-15% 对话逼真度 |

#### 预期指标变化

```
原始预期（基于 5pp 差异假设）：

终点符合率：
  云版本  75-85% 
  本地版本 70-80%  (-5pp)

完成率：
  云版本  90-95%
  本地版本 85-95%  (-5pp)

平均轮数：
  云版本  4.2-5.5
  本地版本 4.0-5.8  (±1)
```

---

## 扩展指南

### 5.1 切换不同的本地模型

#### 方案1：修改 local_llm.py

```python
# 原始
MODEL_NAME = "qwen2.5:1.5b"

# 改为
MODEL_NAME = "qwen2.5:7b"      # 更大的模型
MODEL_NAME = "mistral:latest"   # 其他模型
```

#### 方案2：参数化模型名

```python
def chat(messages, temperature=0.7, timeout=30, retries=3, 
         model="qwen2.5:1.5b"):  # 增加参数
    url = "http://localhost:11434/api/chat"
    payload = {"model": model, ...}
```

然后在 MCTSAgent 中：
```python
rollout_sim = UserSimulator(case, 
    chat_fn=lambda m, t: local_llm.chat_json(m, t, model="qwen2.5:7b"))
```

### 5.2 使用远程本地 Ollama

#### 如果 Ollama 在远程机器上

```python
# local_llm.py
OLLAMA_URL = "http://remote-host:11434/api/chat"

# 或者参数化
def chat(messages, temperature=0.7, timeout=30, retries=3,
         ollama_url="http://localhost:11434/api/chat"):
    url = ollama_url
    ...
```

### 5.3 添加模型缓存

```python
# local_llm.py
_model_cache = {}

def chat(messages, temperature=0.7, timeout=30, retries=3,
         cache=True):
    if cache:
        key = (tuple(messages), temperature)
        if key in _model_cache:
            return _model_cache[key]
    
    result = ... # 调用 API
    
    if cache:
        _model_cache[key] = result
    
    return result
```

### 5.4 集成其他本地 LLM 方案

#### LM Studio（兼容 OpenAI API）
```python
def chat_lm_studio(messages, temperature=0.7):
    url = "http://localhost:1234/v1/chat/completions"
    ...
```

#### LLaMA.cpp 
```python
def chat_llama_cpp(messages, temperature=0.7):
    url = "http://localhost:8080/completion"
    ...
```

#### vLLM
```python
def chat_vllm(messages, temperature=0.7):
    url = "http://localhost:8000/v1/chat/completions"
    ...
```

---

## 常见问题解答

### Q1: 为什么不能直接使用 llm.chat_json 而是要包装 local_llm.chat？
**A:** 因为需要 JSON 输出格式，而直接的 HTTP 返回是文本，需要 _extract_json 处理。通过包装实现了兼容的接口。

### Q2: rollout_sim 为 None 时的性能是否会下降？
**A:** 不会，因为 `sim_to_use = self.rollout_sim or self.sim` 是 O(1) 操作，无性能开销。

### Q3: 能否在 rollout 中混合使用多个模拟器？
**A:** 可以，通过修改 _rollout_action 方法的选择逻辑，但增加了复杂度。

### Q4: 本地模型错误率太高怎么办？
**A:** 可以调整 temperature（降低确定性）或增加 retries 次数。

### Q5: 如何监控 Rollout 中的模型调用？
**A:** 可以在 local_llm.chat_json 中添加日志。

---

**文档完成：** 2026-07-13
