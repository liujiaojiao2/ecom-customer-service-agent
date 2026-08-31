"""对话运行器：Agent vs 用户模拟器，SOP 引擎跟踪真实状态并记录违规。"""
import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baseline_agent import BaselineAgent
from mcts_agent import MCTSAgent
from sop import SOPEngine, SOPState
from user_simulator import UserSimulator

MAX_TURNS = 12


def run_case(case: dict, agent, engine: SOPEngine) -> dict:
    """跑一条完整对话，返回轨迹。一轮 = Agent 动作 + 用户回应。"""
    sim = UserSimulator(case)
    # 阶段0：证据默认有效；issue_type 用测试用例真值
    branch_ctx = {"issue_type": case["issue_type"], "evidence_valid": True}

    strict_state = SOPState()   # 严格口径：违规不推进（→合规率）
    resync_state = SOPState()   # 重同步口径：违规也跳转（→完成率/轮数，驱动对话终止）
    first = sim.first_utterance()
    history = [{"role": "user", "content": first["utterance"]}]
    turns = []
    status, completed = "ok", False

    try:
        for turn_idx in range(MAX_TURNS):
            out = agent.act(history)
            action, reply = out["action"], out["reply"]
            strict_result = engine.step(strict_state, action, branch_ctx)
            resync_result = engine.step(resync_state, action, branch_ctx, resync=True)
            record = {
                "turn": turn_idx + 1,
                "action": action,
                "strict_node_before": strict_state.node,
                "strict_violation": strict_result.violation,
                "strict_violation_reason": strict_result.violation_reason,
                "resync_node_before": resync_state.node,
                "resync_violation": resync_result.violation,
                "agent_reply": reply,
            }
            if "search_stats" in out:
                record["search_stats"] = out["search_stats"]
            strict_state = strict_result.state
            resync_state = resync_result.state
            record["strict_node_after"] = strict_state.node
            record["resync_node_after"] = resync_state.node
            history.append({"role": "agent", "content": reply, "action": action})

            if resync_result.terminal:
                turns.append(record)
                completed = True
                break

            user_out = sim.respond(history)
            history.append({"role": "user", "content": user_out["utterance"]})
            record["user_utterance"] = user_out["utterance"]
            record["satisfaction"] = user_out["satisfaction"]
            turns.append(record)
    except Exception as e:
        status = "error"
        turns.append({"error": f"{e}", "traceback": traceback.format_exc()[-500:]})

    return {
        "case_id": case["case_id"],
        "issue_type": case["issue_type"],
        "emotion": case["emotion"],
        "expected_end": case["expected_end"],
        "status": status,
        "completed": completed,
        "final_node_strict": strict_state.node,
        "final_node_resync": resync_state.node,
        "num_turns": len([t for t in turns if "turn" in t]),
        "last_satisfaction": next((t["satisfaction"] for t in reversed(turns)
                                   if "satisfaction" in t), None),
        "turns": turns,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--cases", type=str, default=None, help="逗号分隔的 case_id")
    parser.add_argument("--out", type=str, default="runs/baseline")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--agent", choices=["baseline", "mcts"], default="baseline")
    parser.add_argument("--use-local-rollout", action="store_true", default=False,
                        help="MCTS agent 使用本地小模型进行 rollout")
    parser.add_argument("--reward-mode", choices=["terminal", "dense", "pbrs"],
                        default="terminal",
                        help="MCTS 搜索奖励模式：terminal=仅终局；dense=手写步奖励；pbrs=势能塑形")
    args = parser.parse_args()

    cases = json.loads((ROOT / "data" / "test_cases.json").read_text(encoding="utf-8"))
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["case_id"] in wanted]
    if args.limit:
        cases = cases[:args.limit]

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = SOPEngine()

    def make_agent(case):
        if args.agent == "mcts":
            return MCTSAgent(case, engine, use_local_rollout=args.use_local_rollout,
                             reward_mode=args.reward_mode)
        return BaselineAgent()

    todo = []
    for case in cases:
        if (out_dir / f"{case['case_id']}.json").exists():
            print(f"{case['case_id']} 已存在，跳过")
        else:
            todo.append(case)

    def run_one(case):
        traj = run_case(case, make_agent(case), engine)
        (out_dir / f"{case['case_id']}.json").write_text(
            json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
        return traj

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, c) for c in todo]
        for fut in as_completed(futures):
            traj = fut.result()
            done += 1
            n_strict = sum(1 for t in traj["turns"] if t.get("strict_violation"))
            n_resync = sum(1 for t in traj["turns"] if t.get("resync_violation"))
            print(f"[{done}/{len(todo)}] {traj['case_id']}: status={traj['status']} "
                  f"完成={traj['completed']} 终点={traj['final_node_resync']} "
                  f"轮数={traj['num_turns']} 严格违规={n_strict} 重同步违规={n_resync}")


if __name__ == "__main__":
    main()
