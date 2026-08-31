# 本地 Rollout 快速启动指南

## 1. 环境准备（5 分钟）

### 安装 Ollama
```bash
# macOS
brew install ollama

# 启动服务（后台运行）
ollama serve &
```

### 下载模型
```bash
ollama pull qwen2.5:1.5b
```

### 验证环境
```bash
curl http://localhost:11434/api/tags
# 应该看到 qwen2.5:1.5b 在列表中
```

---

## 2. 验证安装（2 分钟）

### 黑箱连通性测试
```bash
cd /Users/liujiaojiao/Downloads/电商智能体框架
python3 -c "
import sys
sys.path.insert(0, 'src')
from local_llm import chat
result = chat([{'role': 'user', 'content': '你好'}])
print('✓ 连接成功' if result else '✗ 连接失败')
print(f'回复：{result[:50]}...')
"
```

**预期输出：**
```
✓ 连接成功
回复：你好！很高兴认识你。有什么我可以帮助你的吗？...
```

---

## 3. 运行单条测试（3 分钟）

### 单条冒烟测试
```bash
python src/runner.py --agent mcts --use-local-rollout \
  --cases exchange_01 --out runs/test_local_1
```

**预期输出：**
```
[1/1] exchange_01: status=ok 完成=True 终点=end_with_satisfaction 轮数=5 严格违规=0 重同步违规=0
```

### 查看结果
```bash
python3 -c "
import json
result = json.load(open('runs/test_local_1/exchange_01.json'))
print(f'状态: {result[\"status\"]}')
print(f'完成: {result[\"completed\"]}')
print(f'轮数: {result[\"num_turns\"]}')
print(f'最终节点: {result[\"final_node_resync\"]}')
"
```

---

## 4. 运行全量对比评估（2-3 小时）

### 执行本地 Rollout 版本
```bash
python src/runner.py --agent mcts --use-local-rollout \
  --workers 6 --out runs/mcts_local_60
```

### 执行云 Rollout 版本（对照组）
```bash
python src/runner.py --agent mcts \
  --workers 6 --out runs/mcts_puct_60
```

### 生成评估报告
```bash
python src/evaluate.py runs/mcts_local_60
python src/evaluate.py runs/mcts_puct_60
```

---

## 5. 对比指标

### 查看 local 版本的评估结果
```bash
cat runs/mcts_local_60/evaluation.json | python3 -m json.tool | head -50
```

### 查看 cloud 版本的评估结果
```bash
cat runs/mcts_puct_60/evaluation.json | python3 -m json.tool | head -50
```

### 关键指标对比
```bash
python3 << 'EOF'
import json

local = json.load(open('runs/mcts_local_60/evaluation.json'))
cloud = json.load(open('runs/mcts_puct_60/evaluation.json'))

print("=" * 60)
print("本地 Rollout vs 云 Rollout 对比")
print("=" * 60)
for key in ['endpoint_match_rate', 'completion_rate', 'avg_turns']:
    local_val = local.get(key, 0)
    cloud_val = cloud.get(key, 0)
    diff = local_val - cloud_val
    print(f"{key:20s}: {local_val:6.2%} vs {cloud_val:6.2%} (差异: {diff:+.2%})")
EOF
```

---

## 6. 常见问题

### Q1: "无法连接 Ollama"
**A:** 确认 Ollama 正在运行
```bash
ps aux | grep ollama
# 若没有结果，执行：ollama serve
```

### Q2: "Model qwen2.5:1.5b not found"
**A:** 下载模型
```bash
ollama pull qwen2.5:1.5b
ollama list  # 验证
```

### Q3: 请求超时
**A:** 
- 方案1：等待模型加载完成（第一次调用较慢）
- 方案2：启用 GPU 加速（编辑 ~/.ollama/config.json）
- 方案3：增加超时时间（修改 src/local_llm.py 中的 timeout 参数）

### Q4: 效果下降太多
**A:**
- 1.5B 模型质量低于 DeepSeek 是预期现象
- 可尝试更大的本地模型（qwen2.5:7b）
- 或调整 temperature 参数（降低为 0.5）

---

## 7. 成本对比

### 单条对话的成本
```
云 Rollout:       ¥0.8  (树内 ¥0.8 + Rollout ¥0.1)
本地 Rollout:     ¥0.8  (树内 ¥0.8 + Rollout ¥0)
节省:            ¥0.1/条 (11% 成本节省)

60 条对话总成本:
云版本:    ¥48 (¥0.8 × 60)
本地版本:  ¥48 (¥0.8 × 60) + Rollout 零成本
```

---

## 8. 关键命令速查表

| 任务 | 命令 |
|------|------|
| 启动 Ollama | `ollama serve` |
| 下载模型 | `ollama pull qwen2.5:1.5b` |
| 连通性测试 | `python3 -c "from src.local_llm import chat; print(chat([...]))"` |
| 单条测试 | `python src/runner.py --agent mcts --use-local-rollout --cases exchange_01 --out runs/test_1` |
| 60 条本地 | `python src/runner.py --agent mcts --use-local-rollout --workers 6 --out runs/mcts_local_60` |
| 60 条云版本 | `python src/runner.py --agent mcts --workers 6 --out runs/mcts_puct_60` |
| 生成报告 | `python src/evaluate.py runs/mcts_local_60` |

---

## 9. 目录结构

```
电商智能体框架/
├── src/
│   ├── local_llm.py          ← 新增：本地 LLM 接口
│   ├── user_simulator.py     ← 修改：支持 chat_fn 参数
│   ├── mcts_env.py           ← 修改：支持 rollout_sim 参数
│   ├── mcts_agent.py         ← 修改：支持 use_local_rollout 参数
│   ├── runner.py             ← 修改：新增 --use-local-rollout 标志
│   └── ...
├── runs/
│   ├── mcts_local_60/        ← 本地 Rollout 结果
│   │   ├── exchange_01.json
│   │   ├── ...
│   │   └── evaluation.json
│   └── mcts_puct_60/         ← 云 Rollout 结果
│       ├── exchange_01.json
│       ├── ...
│       └── evaluation.json
└── 文档/
    ├── LOCAL_ROLLOUT_IMPL_SUMMARY.md     ← 实现总结
    ├── IMPLEMENTATION_CHECKLIST.md       ← 检查清单
    ├── FINAL_IMPLEMENTATION_REPORT.md    ← 最终报告
    └── QUICKSTART.md                     ← 本文件
```

---

## 10. 预期耗时

| 步骤 | 耗时 |
|------|-----|
| 环境准备 | 5 分钟 |
| 模型下载 | 5-10 分钟（取决于网速） |
| 连通性测试 | 2 分钟 |
| 单条冒烟测试 | 3 分钟 |
| 全量 60 条（本地） | 60-90 分钟（6 workers） |
| 全量 60 条（云） | 60-90 分钟（6 workers） |
| 评估报告生成 | 2 分钟 |
| **总计** | **3-4 小时** |

---

## 11. 下一步

1. 按照上述步骤运行验证
2. 收集 runs/ 目录下的 evaluation.json 文件
3. 对比指标（终点符合率、完成率、轮数）
4. 评估是否满足预期要求
5. 决定是否在生产环境启用本地 Rollout

---

**开始前的最后检查清单：**
- [ ] Ollama 已安装并运行
- [ ] qwen2.5:1.5b 模型已下载
- [ ] Python 依赖已安装（requests）
- [ ] 有足够的磁盘空间（>5GB）
- [ ] 预留 3-4 小时进行完整评估

准备好了？开始吧！
