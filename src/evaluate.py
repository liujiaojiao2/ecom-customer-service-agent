"""评估：从轨迹目录计算三类核心指标 + 分场景细分。"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def compute_metrics(trajs: list[dict]) -> dict:
    ok = [t for t in trajs if t["status"] == "ok"]
    n_errors = len(trajs) - len(ok)

    total_actions = strict_v = resync_v = 0
    for t in ok:
        for turn in t["turns"]:
            if "turn" not in turn:
                continue
            total_actions += 1
            strict_v += bool(turn["strict_violation"])
            resync_v += bool(turn["resync_violation"])

    completed = [t for t in ok if t["completed"]]
    end_match = sum(
        1 for t in completed
        if (t["final_node_resync"] == "S10") == (t["expected_end"] == "resolution")
    )

    return {
        "n_cases": len(trajs),
        "n_errors": n_errors,
        "total_actions": total_actions,
        "strict_compliance_rate": 1 - strict_v / total_actions if total_actions else 0,
        "resync_compliance_rate": 1 - resync_v / total_actions if total_actions else 0,
        "completion_rate": len(completed) / len(ok) if ok else 0,
        "avg_turns": sum(t["num_turns"] for t in ok) / len(ok) if ok else 0,
        "expected_end_match_rate": end_match / len(completed) if completed else 0,
    }


def load_trajs(run_dir: Path) -> list[dict]:
    return [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(run_dir.glob("*.json"))]


def stratify(trajs: list[dict]) -> list[dict]:
    """分层口径：覆盖率（场景）+ 难度（情绪）+ 期望终点。

    难度与期望终点两层是错配检测器：若总体指标漂亮而 angry / escalation
    子集塌陷，说明 agent 在简单样本上刷分、走了捷径（见 主线 G-2）。
    场景层沿用裸名以兼容历史 metrics.csv。
    """
    rows = []
    for field, values, prefix in [
        ("issue_type", ["refund", "return", "exchange", "logistics"], ""),
        ("emotion", ["calm", "impatient", "angry"], "emotion="),
        ("expected_end", ["resolution", "escalation"], "expected="),
    ]:
        for v in values:
            subset = [t for t in trajs if t.get(field) == v]
            if subset:
                rows.append({"scope": prefix + v, **compute_metrics(subset)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    args = parser.parse_args()
    run_dir = ROOT / args.run_dir if not Path(args.run_dir).is_absolute() else Path(args.run_dir)

    trajs = load_trajs(run_dir)
    rows = [{"scope": "overall", **compute_metrics(trajs)}] + stratify(trajs)

    print(f"{'范围':20s} {'条数':>4s} {'错误':>4s} {'严格合规率':>10s} {'重同步合规率':>11s} "
          f"{'完成率':>7s} {'平均轮数':>8s} {'终点符合率':>9s}")
    prev_prefix = None
    for r in rows:
        prefix = r["scope"].split("=")[0] if "=" in r["scope"] else ""
        if prev_prefix is not None and prefix != prev_prefix:
            print()
        prev_prefix = prefix
        print(f"{r['scope']:20s} {r['n_cases']:4d} {r['n_errors']:4d} "
              f"{r['strict_compliance_rate']:10.1%} {r['resync_compliance_rate']:11.1%} "
              f"{r['completion_rate']:7.1%} {r['avg_turns']:8.2f} "
              f"{r['expected_end_match_rate']:9.1%}")

    out_csv = run_dir / "metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n指标已写入 {out_csv}")


if __name__ == "__main__":
    main()
