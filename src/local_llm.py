"""本地小模型接口（通过 Ollama HTTP API）。"""
import json
import time
from typing import Optional

try:
    import requests
except ImportError:
    requests = None


class LocalLLMError(RuntimeError):
    pass


def _extract_json(text: str) -> dict:
    """从模型输出中鲁棒抽取 JSON 对象（容忍代码围栏与前后缀文字）。

    参考 llm.py 中的实现逻辑。
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise json.JSONDecodeError("未找到 JSON 对象", text, 0)
    return json.loads(text[start:end + 1])


def chat(messages: list[dict], temperature: float = 0.7,
         timeout: int = 30, retries: int = 3) -> str:
    """调用本地 ollama 模型（qwen2.5:1.5b）。

    Args:
        messages: 对话消息列表，格式同 llm.py
        temperature: 生成温度
        timeout: 请求超时（秒）
        retries: 重试次数

    Returns:
        模型输出文本

    Raises:
        LocalLLMError: 连接失败或模型返回异常
    """
    if requests is None:
        raise LocalLLMError("requests 库未安装，运行 pip install requests")

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "qwen2.5:1.5b",
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }

    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code != 200:
                raise LocalLLMError(
                    f"Ollama 返回 {resp.status_code}: {resp.text[:200]}"
                )
            data = resp.json()
            if "message" not in data:
                raise LocalLLMError(f"Ollama 返回缺 message: {data}")
            return data["message"].get("content", "").strip()
        except requests.exceptions.Timeout:
            last_err = LocalLLMError(f"请求超时 ({timeout}s)")
        except requests.exceptions.ConnectionError:
            last_err = LocalLLMError(
                "无法连接 Ollama (http://localhost:11434)，"
                "请确保已运行 'ollama serve'"
            )
        except json.JSONDecodeError as e:
            last_err = LocalLLMError(f"Ollama 返回无效 JSON: {e}")
        except Exception as e:
            last_err = LocalLLMError(f"本地模型调用失败: {e}")

        if attempt < retries - 1:
            time.sleep(0.5 * (attempt + 1))

    raise last_err


def chat_json(messages: list[dict], temperature: float = 0.7,
              timeout: int = 30, retries: int = 3) -> dict:
    """调用本地模型并解析 JSON 响应。

    兼容 llm.chat_json 的接口，用于需要 JSON 输出的场景（如用户模拟器）。

    Args:
        messages: 对话消息列表
        temperature: 生成温度
        timeout: 请求超时（秒）
        retries: 重试次数

    Returns:
        解析后的 JSON dict

    Raises:
        LocalLLMError: 调用失败或 JSON 解析失败
    """
    last_err = None
    for attempt in range(retries):
        try:
            text = chat(messages, temperature, timeout, retries=1)
            if not text:
                raise LocalLLMError("模型返回空白内容")
            try:
                return _extract_json(text)
            except json.JSONDecodeError:
                # 后备方案：纯文本 → 自动包装为 JSON
                # 用于本地小模型无法保证 JSON 格式的情况
                text = text.strip()
                if "satisfaction" not in text.lower() and "utterance" not in text.lower():
                    # 纯文本响应，包装为 user_simulator 期望的格式
                    return {
                        "utterance": text,
                        "satisfaction": "neutral"
                    }
                raise  # 如果看起来像 JSON 但解析失败，就重新抛错
        except LocalLLMError as e:
            last_err = e
        except Exception as e:
            last_err = LocalLLMError(f"本地模型调用失败: {e}")

        if attempt < retries - 1:
            time.sleep(1.0 * (attempt + 1))

    raise last_err
