"""多模态端到端：感知层从图文行为推断 branch_ctx → MCTS Agent 对话。"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mcts_agent import MCTSAgent
from mm_dataset import load_samples
from perception import Perception
from runner import run_case
from sop import SOPEngine


def load_case(case_id: str) -> dict:
    cases = json.loads((ROOT / "data" / "test_cases.json").read_text(encoding="utf-8"))
    return next(c for c in cases if c["case_id"] == case_id)


def run_one(mm, out_dir):
    perception = Perception().perceive({
        "text": mm.first_utterance, "image": str(mm.image_path),
        "behaviors": mm.behaviors})
    branch_ctx = {"issue_type": perception["issue_type"],
                  "evidence_valid": perception["has_evidence"]
                                    if perception["issue_type"] in {"return", "exchange"}
                                    else True}
    case = load_case(mm.case_id)
    engine = SOPEngine()
    agent = MCTSAgent(case, engine, branch_ctx=branch_ctx)
    traj = run_case(case, agent, engine)
    traj["perception"] = perception
    traj["gold_issue_type"] = mm.gold_issue_type
    (out_dir / f"{mm.case_id}.json").write_text(
        json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
    return traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id", type=str, nargs="?", default=None,
                        help="单条 case_id；不传则跑全部 20 条")
    parser.add_argument("--out", type=str, default="runs/mm_e2e")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    if args.case_id:
        samples = [s for s in samples if s.case_id == args.case_id]
        if not samples:
            print(f"未找到样本: {args.case_id}"); sys.exit(1)

    todo = [s for s in samples if not (out_dir / f"{s.case_id}.json").exists()]
    for s in samples:
        if s not in todo:
            print(f"{s.case_id} 已存在，跳过")

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, s, out_dir): s for s in todo}
        for fut in as_completed(futures):
            traj = fut.result()
            done += 1
            n_strict = sum(1 for t in traj["turns"] if t.get("strict_violation"))
            print(f"[{done}/{len(todo)}] {traj['case_id']}: status={traj['status']} "
                  f"完成={traj['completed']} 终点={traj['final_node_resync']} "
                  f"轮数={traj['num_turns']} 严格违规={n_strict}")


if __name__ == "__main__":
    main()
