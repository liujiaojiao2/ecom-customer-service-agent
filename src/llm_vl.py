"""DashScope Qwen-VL 客户端（OpenAI 兼容接口）。

调用真实 API 需要 DASHSCOPE_API_KEY，未设置时抛出明确异常。骨架先行、单测用 mock。
"""
import base64
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from llm import _extract_json  # 复用阶段0 的 JSON 抽取

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-vl-plus"


class VLError(RuntimeError):
    pass


def _client() -> OpenAI:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise VLError("DASHSCOPE_API_KEY 未设置，请在 .env 中补充")
    return OpenAI(api_key=key, base_url=DASHSCOPE_BASE_URL)


def _encode_image(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise VLError(f"图片不存在: {p}")
    ext = p.suffix.lstrip(".").lower() or "jpeg"
    mime = "jpeg" if ext == "jpg" else ext
    return f"data:image/{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def _build_multimodal_content(text: str, images: list[str | Path]) -> list[dict]:
    parts = [{"type": "image_url", "image_url": {"url": _encode_image(p)}}
             for p in images]
    parts.append({"type": "text", "text": text})
    return parts


def chat_vl(text: str, images: list[str | Path] | None = None,
            model: str = DEFAULT_MODEL, temperature: float = 0.2,
            max_tokens: int = 1024, retries: int = 3) -> str:
    """多模态文本回复。"""
    images = images or []
    if not images:
        messages = [{"role": "user", "content": text}]
    else:
        messages = [{"role": "user", "content": _build_multimodal_content(text, images)}]
    last_err = None
    for attempt in range(retries):
        try:
            resp = _client().chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                raise VLError("Qwen-VL 返回空白")
            return content
        except VLError as e:
            last_err = e
        except Exception as e:
            last_err = VLError(f"Qwen-VL 调用失败: {e}")
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))
    raise last_err


def chat_vl_json(text: str, images: list[str | Path] | None = None,
                 model: str = DEFAULT_MODEL, temperature: float = 0.2,
                 max_tokens: int = 1024, retries: int = 3) -> dict:
    """多模态 JSON 回复：prompt 约束 + 鲁棒抽取。"""
    prompt = text + "\n\n只输出一个 JSON 对象，不要输出任何其他文字。"
    text_out = ""
    last_err = None
    for attempt in range(retries):
        try:
            text_out = chat_vl(prompt, images, model=model,
                               temperature=temperature, max_tokens=max_tokens,
                               retries=1)
            return _extract_json(text_out)
        except json.JSONDecodeError:
            last_err = VLError(f"JSON 解析失败: {text_out[:200]}")
        except VLError as e:
            last_err = e
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))
    raise last_err
