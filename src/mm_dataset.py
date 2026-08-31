"""多模态样本 loader：读取 samples.json，校验图片存在，产出可迭代的样本对象。"""
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_PATH = ROOT / "data" / "mm_samples" / "samples.json"

ISSUE_TYPES = {"refund", "return", "exchange", "logistics"}


@dataclass
class MMSample:
    case_id: str
    image_path: Path
    first_utterance: str
    behaviors: list[str]
    product: str
    gold_issue_type: str
    gold_has_evidence: bool


def _validate_record(rec: dict) -> None:
    for f in ("case_id", "image_path", "first_utterance", "behaviors", "gold"):
        if f not in rec:
            raise ValueError(f"{rec.get('case_id', '?')}: 缺字段 {f}")
    gold = rec["gold"]
    if gold["issue_type"] not in ISSUE_TYPES:
        raise ValueError(f"{rec['case_id']}: issue_type 非法 {gold['issue_type']}")
    if not isinstance(gold["has_evidence"], bool):
        raise ValueError(f"{rec['case_id']}: has_evidence 必须为 bool")


def load_samples(path: Path = SAMPLES_PATH, require_images: bool = True) -> list[MMSample]:
    """加载并校验样本。require_images=True 时任何图片缺失都会报错。"""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    samples, missing = [], []
    for rec in records:
        _validate_record(rec)
        raw = Path(rec["image_path"])
        img = raw if raw.is_absolute() else ROOT / raw
        if not img.exists():
            # 兼容扩展名不一致：同名不同后缀（png/jpg/jpeg/webp）也认
            for cand in img.parent.glob(img.stem + ".*"):
                if cand.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    img = cand
                    break
        if not img.exists():
            missing.append(rec["case_id"])
            if require_images:
                continue
        samples.append(MMSample(
            case_id=rec["case_id"], image_path=img,
            first_utterance=rec["first_utterance"],
            behaviors=list(rec["behaviors"]),
            product=rec.get("product", ""),
            gold_issue_type=rec["gold"]["issue_type"],
            gold_has_evidence=rec["gold"]["has_evidence"],
        ))
    if require_images and missing:
        raise FileNotFoundError(
            f"缺少 {len(missing)} 张图片: {', '.join(missing[:5])}"
            + ("..." if len(missing) > 5 else ""))
    return samples


def summary(samples: list[MMSample]) -> dict:
    dist = {}
    for s in samples:
        dist[s.gold_issue_type] = dist.get(s.gold_issue_type, 0) + 1
    return {"total": len(samples), "distribution": dist}
