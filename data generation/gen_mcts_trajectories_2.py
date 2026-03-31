import json
import random
import dashscope
from http import HTTPStatus

# ================= 极其重要：填入你的 API KEY =================
dashscope.api_key = "sk-f75659950e574836b6ec32c78ed7636a" # 请替换为你真实的阿里云 API KEY

# ================= 1. 定义极其简化的 SOP 规则库 (MCTS 的掩码基础) =================
# 真实场景中，你会从 json 文件里读取这个。这里为了跑通写在代码里。
SOP_GRAPH = {
    "STATE_识别诉求": ["STATE_安抚并询问单号", "STATE_直接拒赔", "STATE_结束"],
    "STATE_安抚并询问单号": ["STATE_核实并退款", "STATE_要求补充图片", "STATE_结束"],
    "STATE_要求补充图片": ["STATE_核实并退款", "STATE_结束"],
    "STATE_核实并退款": ["STATE_结束"],
    "STATE_直接拒赔": ["STATE_结束"]
}

# ================= 2. 真实的 LLM 调用引擎 =================
def call_llm(prompt):
    """动态路由的大模型调用引擎"""
    if "作为Agent" in prompt:
        model_name = dashscope.Generation.Models.qwen_max # 规划器用聪明的 Max
        system_msg = "你是一个熟知SOP的金牌电商客服。必须严格输出合法的JSON格式，不准包含任何```json等Markdown标记。"
    else:
        model_name = dashscope.Generation.Models.qwen_turbo # 用户用速度快的 Turbo
        system_msg = "你是一个真实的电商买家。说话一定要口语化，不要像机器人。"

    try:
        response = dashscope.Generation.call(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': prompt}
            ],
            result_format='message',
            temperature=0.7 
        )

        if response.status_code == HTTPStatus.OK:
            output_text = response.output.choices[0].message.content
            # 清洗 Agent 的 JSON 输出
            if "作为Agent" in prompt:
                output_text = output_text.replace("```json", "").replace("```", "").strip()
            return output_text
        else:
            return _get_fallback_response(prompt)
    except Exception as e:
        print(f"代码运行异常: {e}")
        return _get_fallback_response(prompt)

def _get_fallback_response(prompt):
    """API 兜底防崩溃机制"""
    if "作为Agent" in prompt:
        return '{"thought": "API请求出错，启动兜底", "action": "STATE_结束", "response": "抱歉亲，系统开了点小差。"}'
    else:
        return "算了，我不弄了。"

# ================= 3. 感知层起点：让 LLM 生成第一句话 =================
def generate_initial_context():
    behaviors = [["查物流", "点退换货"], ["看订单", "点人工客服"]]
    profiles = [{"sentiment": "焦急", "intent": "催退款"}, {"sentiment": "愤怒", "intent": "投诉破损"}]
    
    b = random.choice(behaviors)
    p = random.choice(profiles)
    
    # 【修改点1】：让 Turbo 真实模拟开场白，彻底解决同质化
    user_prompt = f"作为用户。你刚经历了如下操作：{b}。你现在的情绪是：{p['sentiment']}，你的核心诉求是：{p['intent']}。请根据这些条件，输出你发给客服的第一句话（抱怨或质问）。只输出你说的原话。"
    
    first_utterance = call_llm(user_prompt)
    
    return {
        "order_context": {"short_term_behavior": b},
        "situational_profile": p,
        "dialogue_history": [{"role": "user", "content": first_utterance}],
        "current_sop_state": "STATE_识别诉求" # MCTS 规划的绝对起点 S_0
    }

# ================= 4. 规划层博弈：MCTS 左右互搏 =================
def generate_trajectory(initial_data):
    trajectory = [initial_data] # 把短脚本数据作为第一帧存入
    
    current_state = initial_data["current_sop_state"]
    history = initial_data["dialogue_history"]
    
    max_turns = 4 # 防止死循环，最多博弈 4 轮
    for turn in range(max_turns):
        
        # 【修改点2】：动态提取当前状态的合法动作，强制 Agent 遵守 SOP (Action Mask)
        valid_actions = SOP_GRAPH.get(current_state, ["STATE_结束"])
        
        agent_prompt = f"""
作为Agent，当前所处SOP节点为: 【{current_state}】。
当前允许你跳转的合法动作空间有: {valid_actions}。
对话历史: {history}。

请你进行规划推演，严格从上述【合法动作空间】中选择一个action，并输出如下JSON格式：
{{"thought": "你的思考过程，为什么选这个动作", "action": "你选的动作", "response": "你回复用户的话术"}}
"""
        raw_agent_res = call_llm(agent_prompt)
        
        # 【修改点3】：加入 JSON 解析防崩溃保护
        try:
            agent_res = json.loads(raw_agent_res)
        except json.JSONDecodeError:
            print(f"⚠️ JSON 解析失败，跳过本轮。原始输出: {raw_agent_res}")
            break # 如果解析失败，直接终止这条数据的生成，宁缺毋滥
        
        step_data = {
            "turn": turn + 1,
            "agent_thought": agent_res.get("thought", ""),
            "agent_action": agent_res.get("action", "STATE_结束"),
            "agent_response": agent_res.get("response", "")
        }
        
        history.append({"role": "agent", "content": step_data["agent_response"]})
        current_state = step_data["agent_action"] # 状态正式转移
        
        # 提前结束判定
        if current_state == "STATE_结束":
            trajectory.append(step_data)
            break
            
        # User Simulator 给出反应
        user_prompt = f"作为用户，客服刚才对你说: '{step_data['agent_response']}'。请结合你的脾气，给出你的下一句真实回复。"
        user_reply = call_llm(user_prompt)
        history.append({"role": "user", "content": user_reply})
        step_data["user_reply"] = user_reply
        
        trajectory.append(step_data)
            
    return trajectory

# ================= 执行主函数 =================
if __name__ == "__main__":
    print("🚀 开始合成极简多模态售后 MCTS 数据...")
    print("-" * 50)
    
    # 生成一整条完整的博弈数据
    init_data = generate_initial_context()
    full_trajectory = generate_trajectory(init_data)
    
    # 完美打印出来欣赏
    print(json.dumps(full_trajectory, ensure_ascii=False, indent=2))
    print("-" * 50)
    print("✅ 单条数据生成完毕！你可以加上 for 循环和多线程批量生成了。")