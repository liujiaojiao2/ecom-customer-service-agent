"""LLM 用户模拟器：按测试用例人设扮演用户，输出话语 + 满意度信号。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm

EMOTION_ZH = {"calm": "平和讲理", "impatient": "着急催促", "angry": "愤怒不满"}

SYSTEM_TMPL = """你在扮演一位电商售后场景的真实用户，正在和客服对话。严格保持人设，不要出戏，不要替客服说话。

## 你的人设
{persona_desc}
初始情绪：{emotion_zh}；耐心值：{patience}/5（耐心值越低越容易被激怒）

## 你的情况
商品：{product}（订单金额 ¥{amount}，{shipped_zh}）
诉求类型：{issue_zh}
{excessive_line}

## 行为规则
1. 每次只说一句到三句话，口语化，符合人设语气
2. 客服索要订单号时，提供订单号 JD{case_id_hash}
3. 客服请你提供凭证/照片时{evidence_line}
4. 客服给出合理方案且满足你的诉求 → 接受并感谢，satisfaction 输出 satisfied
5. 客服答非所问、反复追问、拖延 → 表达不满，satisfaction 输出 angry；耐心耗尽可要求转人工
6. 对话自然推进，不要无限纠缠：诉求被满足就结束

## 输出格式（JSON）
{{"utterance": "你要说的话", "satisfaction": "satisfied | neutral | angry"}}"""

ISSUE_ZH = {"refund": "退款（商品未发货，想直接退款）",
            "return": "退货退款（商品有质量/破损问题）",
            "exchange": "换货（尺寸/型号不符）",
            "logistics": "物流异常（迟迟未送达）"}


class UserSimulator:
    def __init__(self, case: dict, chat_fn=None):
        self.case = case
        self.chat_fn = chat_fn or llm.chat_json
        order = case["order"]
        excessive_line = (
            "特别注意：你认为自己受到了损失，坚持要求高额赔偿（至少订单金额的50%），"
            "客服只给小额补偿或拒绝时你不接受，要求转人工/投诉。"
            if case["demand_excessive"] else
            "你的诉求合理，客服按流程处理即可满意。"
        )
        evidence_line = ("，你配合提供（用文字描述照片内容即可）"
                         if not case["demand_excessive"]
                         else "，你不太耐烦但会提供")
        self.system = SYSTEM_TMPL.format(
            persona_desc=case["persona_desc"],
            emotion_zh=EMOTION_ZH[case["emotion"]],
            patience=case["patience"],
            product=order["product"], amount=order["amount"],
            shipped_zh="已发货" if order["shipped"] else "未发货",
            issue_zh=ISSUE_ZH[case["issue_type"]],
            excessive_line=excessive_line,
            evidence_line=evidence_line,
            case_id_hash=abs(hash(case["case_id"])) % 100000000,
        )

    def first_utterance(self) -> dict:
        sat = "angry" if self.case["emotion"] == "angry" else "neutral"
        return {"utterance": self.case["first_utterance"], "satisfaction": sat}

    def respond(self, history: list[dict]) -> dict:
        """history: [{role: 'agent'|'user', content: str}, ...]，返回用户下一句。

        对话记录合并为单条 user 消息（transcript 风格）：DeepSeek JSON 模式在
        system 后直接跟 assistant 的非标准消息序列下会偶发返回空白内容。
        """
        lines = []
        for turn in history:
            speaker = "你（用户）" if turn["role"] == "user" else "客服"
            lines.append(f"{speaker}: {turn['content']}")
        transcript = ("对话记录：\n" + "\n".join(lines) +
                      "\n\n请以用户身份输出下一句回复（按 system 规定的 JSON 格式）。")
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": transcript}]
        out = None
        for _ in range(3):  # 本地小模型偶发输出错键 JSON，重试即可恢复
            out = self.chat_fn(messages, temperature=0.8)
            if "utterance" in out:
                break
        else:
            raise llm.LLMError(f"模拟器输出缺 utterance: {out}")
        if out.get("satisfaction") not in ("satisfied", "neutral", "angry"):
            out["satisfaction"] = "neutral"
        return out
