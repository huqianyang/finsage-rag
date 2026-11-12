"""大模型客户端：硅基流动 OpenAI 兼容接口。"""
import base64
import json
import logging
import re
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.siliconflow_api_key, base_url=s.siliconflow_base_url,
                  timeout=s.llm_timeout_seconds, max_retries=2)


def _extra() -> dict[str, Any]:
    """硅基流动扩展参数：关闭思考模式可显著降低首字延迟（实测 2.4s → 0.5s）。"""
    return {"extra_body": {"enable_thinking": get_settings().llm_enable_thinking}}


def chat(messages: list[dict[str, Any]], *, temperature: float = 0.2, max_tokens: int = 2048,
         json_mode: bool = False, model: str | None = None) -> str:
    kwargs: dict[str, Any] = _extra()
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = get_client().chat.completions.create(
        model=model or get_settings().llm_model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, stream=False, **kwargs,
    )
    return resp.choices[0].message.content or ""


def chat_json(messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
    """要求模型输出 JSON，并做容错解析（去掉 ```json 包裹）。"""
    text = chat(messages, json_mode=True, **kwargs)
    return parse_json(text)


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"模型未返回 JSON：{text[:200]}")
    return json.loads(text[start:end + 1])


def chat_stream(messages: list[dict[str, Any]], *, temperature: float = 0.3, max_tokens: int = 2048) -> Iterator[str]:
    stream = get_client().chat.completions.create(
        model=get_settings().llm_model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, stream=True, **_extra(),
    )
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
    finally:
        stream.close()


def describe_image(image_path: Path, context: str) -> str:
    """调用 VLM 生成图片摘要。"""
    suffix = image_path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    prompt = (
        "这是一份金融资料中的图片。请用中文简要描述图片内容（不超过 150 字）：如果是表格或图表，"
        "提取其中的关键数字、指标名称和趋势；如果是印章、logo、装饰图，只回答“装饰性图片”。"
        "只描述图中能直接看到的内容，不要做评价、推断或补充图中没有的信息。"
        f"\n图片上下文：{context[:500]}"
    )
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]}]
    return chat(messages, model=get_settings().vlm_model, max_tokens=400).strip()
