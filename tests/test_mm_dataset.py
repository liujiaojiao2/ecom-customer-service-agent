"""M14 多模态 loader：schema 校验、图片存在检查、分布统计。"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mm_dataset import ISSUE_TYPES, MMSample, load_samples, summary


def _write(tmp_path, records):
    p = tmp_path / "samples.json"
    p.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return p


def _rec(case_id, image_path, issue_type="return", has_evidence=True):
    return {
        "case_id": case_id, "image_path": str(image_path),
        "first_utterance": "…", "behaviors": ["a", "b"], "product": "p",
        "gold": {"issue_type": issue_type, "has_evidence": has_evidence},
    }


def test_load_ok(tmp_path):
    img = tmp_path / "a.jpg"; img.write_bytes(b"x")
    p = _write(tmp_path, [_rec("c1", img)])
    s = load_samples(p)
    assert len(s) == 1 and s[0].case_id == "c1"
    assert s[0].gold_issue_type == "return" and s[0].gold_has_evidence is True


def test_missing_image_raises(tmp_path):
    p = _write(tmp_path, [_rec("c1", tmp_path / "missing.jpg")])
    with pytest.raises(FileNotFoundError, match="缺少 1 张"):
        load_samples(p)


def test_missing_image_soft(tmp_path):
    """软模式：即使图片缺失也返回样本，方便用户查看清单。"""
    img_ok = tmp_path / "a.jpg"; img_ok.write_bytes(b"x")
    p = _write(tmp_path, [_rec("c1", img_ok), _rec("c2", tmp_path / "missing.jpg")])
    s = load_samples(p, require_images=False)
    assert [x.case_id for x in s] == ["c1", "c2"]


def test_bad_issue_type(tmp_path):
    img = tmp_path / "a.jpg"; img.write_bytes(b"x")
    p = _write(tmp_path, [_rec("c1", img, issue_type="bogus")])
    with pytest.raises(ValueError, match="issue_type 非法"):
        load_samples(p)


def test_bad_has_evidence_type(tmp_path):
    img = tmp_path / "a.jpg"; img.write_bytes(b"x")
    rec = _rec("c1", img); rec["gold"]["has_evidence"] = "yes"
    p = _write(tmp_path, [rec])
    with pytest.raises(ValueError, match="has_evidence"):
        load_samples(p)


def test_missing_field(tmp_path):
    img = tmp_path / "a.jpg"; img.write_bytes(b"x")
    rec = _rec("c1", img); del rec["first_utterance"]
    p = _write(tmp_path, [rec])
    with pytest.raises(ValueError, match="缺字段"):
        load_samples(p)


def test_summary_distribution():
    samples = [
        MMSample("c1", Path("/a"), "", [], "", "refund", False),
        MMSample("c2", Path("/b"), "", [], "", "refund", False),
        MMSample("c3", Path("/c"), "", [], "", "return", True),
    ]
    assert summary(samples) == {"total": 3, "distribution": {"refund": 2, "return": 1}}


def test_issue_types_constant():
    assert ISSUE_TYPES == {"refund", "return", "exchange", "logistics"}
