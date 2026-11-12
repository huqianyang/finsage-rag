"""大模型客户端：调用硅基流动的 OpenAI 兼容接口。

提供四种调用方式：
- chat()           普通对话，一次性返回完整文本
- chat_json()      要求模型输出 JSON 并解析成字典（问题改写、元数据抽取用）
- chat_stream()    流式对话，逐段返回文本（回答生成用，前端可边生成边显示）
- describe_image() 调用视觉模型给图片生成文字摘要（导入时处理文档图片用）
"""
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
    """创建（并缓存）OpenAI SDK 客户端，base_url 指向硅基流动。失败时 SDK 自动重试 2 次。"""
    s = get_settings()
    return OpenAI(api_key=s.siliconflow_api_key, base_url=s.siliconflow_base_url,
                  timeout=s.llm_timeout_seconds, max_retries=2)


def _extra() -> dict[str, Any]:
    """硅基流动扩展参数：关闭思考模式可显著降低首字延迟（实测 2.4s → 0.5s）。

    extra_body 中的字段会原样合并进请求体，用来传 OpenAI 标准参数之外的厂商参数。
    """
    return {"extra_body": {"enable_thinking": get_settings().llm_enable_thinking}}


def chat(messages: list[dict[str, Any]], *, temperature: float = 0.2, max_tokens: int = 2048,
         json_mode: bool = False, model: str | None = None) -> str:
    """普通（非流式）对话，返回模型输出的完整文本。

    Args:
        messages: OpenAI 格式的消息列表，如 [{"role": "user", "content": "..."}]
        temperature: 随机性，越低越稳定；抽取类任务建议 0
        max_tokens: 最多生成的 token 数
        json_mode: True 时要求模型只输出 JSON 对象
        model: 使用的模型，默认取配置 LLM_MODEL
    """
    kwargs: dict[str, Any] = _extra()
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = get_client().chat.completions.create(
        model=model or get_settings().llm_model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, stream=False, **kwargs,
    )
    return resp.choices[0].message.content or ""


def chat_json(messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
    """要求模型输出 JSON，并做容错解析（去掉 ```json 包裹）。解析失败抛 ValueError。"""
    text = chat(messages, json_mode=True, **kwargs)
    return parse_json(text)


def parse_json(text: str) -> dict[str, Any]:
    """从模型输出中提取 JSON 对象。

    模型有时会在 JSON 外面包一层 ```json ... ``` 代码块，或在前后多说几句话，
    这里先去掉代码块标记，再截取第一个 "{" 到最后一个 "}" 之间的内容解析。
    """
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"模型未返回 JSON：{text[:200]}")
    return json.loads(text[start:end + 1])


def chat_stream(messages: list[dict[str, Any]], *, temperature: float = 0.3, max_tokens: int = 2048) -> Iterator[str]:
    """流式对话：每收到一段新文本就 yield 出去。

    调用方提前停止迭代（例如用户点了"停止生成"）时，finally 会关闭 HTTP 流，
    服务端随即停止生成，不再消耗 token。
    """
    stream = get_client().chat.completions.create(
        model=get_settings().llm_model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, stream=True, **_extra(),
    )
    try:
        for chunk in stream:
            # 最后一个分片可能只带用量统计、没有 choices
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # 只要正文 content；思考模式下的 reasoning_content 不展示给用户
            if delta and delta.content:
                yield delta.content
    finally:
        stream.close()


def describe_image(image_path: Path, context: str) -> str:
    """调用视觉模型（VLM）生成图片摘要。

    图片以 base64 data URL 的形式直接放进消息里发送，不需要公网可访问的图片地址。

    Args:
        image_path: 本地图片路径
        context: 图片在文档中前后的文字，帮助模型理解图片含义（最多取 500 字）

    Returns:
        摘要文本；装饰性图片（印章、logo 等）会返回"装饰性图片"，调用方据此跳过
    """
    suffix = image_path.suffix.lower().lstrip(".") or "png"
    # MIME 类型里 jpg 要写成 jpeg
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    prompt = (
        "这是一份金融资料中的图片。请用中文简要描述图片内容（不超过 150 字）：如果是表格或图表，"
        "提取其中的关键数字、指标名称和趋势；如果是印章、logo、装饰图，只回答“装饰性图片”。"
        "只描述图中能直接看到的内容，不要做评价、推断或补充图中没有的信息。"
        f"\n图片上下文：{context[:500]}"
    )
    # 多模态消息：content 是一个列表，同时包含图片和文字
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]}]
    return chat(messages, model=get_settings().vlm_model, max_tokens=400).strip()
