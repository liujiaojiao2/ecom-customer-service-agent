"""DeepSeek LLM 客户端（OpenAI 兼容接口）。"""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


class LLMError(RuntimeError):
    pass


def _client() -> OpenAI:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise LLMError("DEEPSEEK_API_KEY 未设置，请检查 .env")
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)


def chat(messages: list[dict], model: str = DEFAULT_MODEL,
         temperature: float = 0.7, max_tokens: int = 1024) -> str:
    try:
        resp = _client().chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
    except Exception as e:
        raise LLMError(f"LLM 调用失败: {e}") from e
    return resp.choices[0].message.content


def _extract_json(text: str) -> dict:
    """从模型输出中鲁棒抽取 JSON 对象（容忍代码围栏与前后缀文字）。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise json.JSONDecodeError("未找到 JSON 对象", text, 0)
    return json.loads(text[start:end + 1])


def chat_json(messages: list[dict], model: str = DEFAULT_MODEL,
              temperature: float = 0.7, max_tokens: int = 1024,
              retries: int = 3) -> dict:
    """要求模型输出 JSON 并鲁棒解析。

    不用 DeepSeek 的 response_format=json_object——它会不定期返回纯空白内容
    （已在多种消息结构下稳定复现），改用 prompt 约束 + 文本抽取。
    """
    last_err = None
    text = ""
    for attempt in range(retries):
        try:
            resp = _client().chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise LLMError("模型返回空白内容")
            return _extract_json(text)
        except json.JSONDecodeError:
            last_err = LLMError(f"JSON 解析失败: {text[:200]}")
        except LLMError as e:
            last_err = e
        except Exception as e:
            last_err = LLMError(f"LLM 调用失败: {e}")
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))
    raise last_err
