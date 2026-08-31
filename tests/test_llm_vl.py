"""M13 DashScope 客户端：不依赖真实 key，用 mock 验证消息装配。"""
import base64
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import llm_vl


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "test.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
    return p


def test_no_key_raises(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(llm_vl.VLError, match="DASHSCOPE_API_KEY"):
        llm_vl.chat_vl("hi")


def test_encode_missing_image_raises():
    with pytest.raises(llm_vl.VLError, match="图片不存在"):
        llm_vl._encode_image("/nonexistent.jpg")


def test_encode_image_base64_prefix(img):
    url = llm_vl._encode_image(img)
    assert url.startswith("data:image/jpeg;base64,")
    payload = url.split(",", 1)[1]
    assert base64.b64decode(payload) == img.read_bytes()


def test_build_multimodal_content_shape(img):
    parts = llm_vl._build_multimodal_content("describe", [img, img])
    assert len(parts) == 3
    assert parts[0]["type"] == "image_url" and parts[1]["type"] == "image_url"
    assert parts[2] == {"type": "text", "text": "describe"}


@patch("llm_vl._client")
def test_chat_vl_assembles_messages_and_returns(mock_client, img):
    fake = MagicMock()
    fake.chat.completions.create.return_value.choices[0].message.content = "分类=退货"
    mock_client.return_value = fake
    out = llm_vl.chat_vl("图里是什么？", images=[img])
    assert out == "分类=退货"
    call = fake.chat.completions.create.call_args
    msg = call.kwargs["messages"][0]
    assert msg["role"] == "user"
    assert len(msg["content"]) == 2  # 1 图 + 1 文本
    assert msg["content"][0]["type"] == "image_url"


@patch("llm_vl._client")
def test_chat_vl_json_parses(mock_client, img):
    fake = MagicMock()
    fake.chat.completions.create.return_value.choices[0].message.content = (
        '这是一个 JSON：{"issue_type": "return", "has_evidence": true}')
    mock_client.return_value = fake
    out = llm_vl.chat_vl_json("分类", images=[img])
    assert out == {"issue_type": "return", "has_evidence": True}


@patch("llm_vl._client")
def test_chat_vl_blank_retries_then_raises(mock_client):
    fake = MagicMock()
    fake.chat.completions.create.return_value.choices[0].message.content = "   "
    mock_client.return_value = fake
    with pytest.raises(llm_vl.VLError, match="空白"):
        llm_vl.chat_vl("x", retries=2)
    assert fake.chat.completions.create.call_count == 2
