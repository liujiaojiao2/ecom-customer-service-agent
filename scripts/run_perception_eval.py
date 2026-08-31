"""跑全量多模态感知评估：并发调用 Qwen-VL，产出准确率与混淆矩阵。"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mm_dataset import load_samples
from mm_eval import compute_metrics, format_report
from perception import Perception

OUT = ROOT / "runs" / "perception"
OUT.mkdir(parents=True, exist_ok=True)


def run_one(sample):
    p = Perception()
    try:
        result = p.perceive({"text": sample.first_utterance,
                             "image": str(sample.image_path),
                             "behaviors": sample.behaviors})
        return {"case_id": sample.case_id,
                "gold": {"issue_type": sample.gold_issue_type,
                         "has_evidence": sample.gold_has_evidence},
                "pred": result, "error": None}
    except Exception as e:
        return {"case_id": sample.case_id,
                "gold": {"issue_type": sample.gold_issue_type,
                         "has_evidence": sample.gold_has_evidence},
                "pred": None, "error": str(e)}


def main():
    samples = load_samples()
    print(f"加载 {len(samples)} 条样本，开始评估…\n")
    records = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(run_one, s): s for s in samples}
        for fut in as_completed(futures):
            r = fut.result()
            records.append(r)
            if r["error"]:
                print(f"[ERR] {r['case_id']}: {r['error'][:80]}")
            else:
                p = r["pred"]
                it_ok = "✓" if p["issue_type"] == r["gold"]["issue_type"] else "✗"
                ev_ok = "✓" if p["has_evidence"] == r["gold"]["has_evidence"] else "✗"
                print(f"[{r['case_id']:14s}] issue={p['issue_type']:10s}{it_ok} "
                      f"ev={str(p['has_evidence']):5s}{ev_ok} conf={p['confidence']}")

    (OUT / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = [r for r in records if not r["error"]]
    metrics = compute_metrics(ok)
    print("\n" + "=" * 60)
    print(format_report(metrics))
    print(f"\n错误 {len(records) - len(ok)} 条")

    (OUT / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n轨迹与指标：{OUT}")


if __name__ == "__main__":
    main()
