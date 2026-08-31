"""基线 Agent：单次 LLM 调用，SOP 规则全文写入 prompt，直接生成 {action, reply}。
不做硬性动作掩码——合规与否由外部 SOP 引擎事后检查，这正是基线要暴露的问题。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm

SOP_RULES_TEXT = """## 售后 SOP 流程规则（必须遵守）

流程节点与合法动作：
- S0 接待 → greet_and_ask(问候并询问问题) → S1
- S1 问题识别 → clarify_issue(追问细节,最多2次,留在S1) / ask_order_info(索要订单号) → S2
- S2 订单核实 → query_order(查询订单,按诉求分流) → 退款到S3 / 退货换货到S4 / 物流到S6
- S3 退款资格核验 → check_refund_policy → S7
- S4 证据收集 → request_evidence(请用户提供照片/凭证) → S5
- S5 证据核验 → review_evidence(核验证据) → 有效到S7 / 无效到S11转人工
- S6 物流查询 → query_logistics → S7
- S7 方案提议 → propose_solution(提出具体方案) → S8
- S8 用户确认 → execute_action(用户同意后执行操作) → S9 / revise_solution(用户不满意,换方案,最多2次) → S7
- S9 执行 → confirm_resolution(确认完成) → S10结束
- 任意节点可用：comfort_user(安抚情绪,不推进流程,最多3次) / escalate(转人工)

硬性合规要求：
1. 必须先核实订单(ask_order_info→query_order)才能给方案
2. 退货/换货必须先收集并核验证据(request_evidence→review_evidence)
3. 必须用户确认方案后(S8)才能 execute_action
4. 赔付券金额不得超过订单金额的20%，用户坚持超额索赔时必须 escalate
5. 每一步的动作必须是当前节点的合法动作"""

SYSTEM_TMPL = """你是电商售后客服。根据 SOP 规则和对话历史，决定下一步动作并生成回复。

{sop_rules}

## 输出格式（JSON）
{{"action": "动作名(必须是上述动作之一)", "reply": "发给用户的话,口语化、专业、有温度"}}"""


class BaselineAgent:
    def __init__(self):
        self.system = SYSTEM_TMPL.format(sop_rules=SOP_RULES_TEXT)

    def act(self, history: list[dict]) -> dict:
        """history: [{role: 'agent'|'user', content}]，返回 {action, reply}。"""
        messages = [{"role": "system", "content": self.system}]
        if not history:
            messages.append({"role": "user", "content": "（用户刚进入会话，尚未发言）"})
        for turn in history:
            if turn["role"] == "agent":
                # 自己的历史轮以 JSON 呈现，避免模型模仿纯文本先例丢掉输出格式
                content = json.dumps({"action": turn["action"], "reply": turn["content"]},
                                     ensure_ascii=False) if "action" in turn else turn["content"]
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "user", "content": turn["content"]})
        out = llm.chat_json(messages, temperature=0.3)
        if "action" not in out or "reply" not in out:
            raise llm.LLMError(f"Agent 输出缺字段: {out}")
        return out
