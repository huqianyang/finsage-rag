"""联网搜索：阿里云百炼 WebSearch MCP（Streamable HTTP），工具 bailian_web_search。

未配置 DASHSCOPE_API_KEY 或调用失败时返回空列表（降级，不影响问答）。
"""
import asyncio
import json
import logging
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from app.config import get_settings

logger = logging.getLogger(__name__)
TOOL_NAME = "bailian_web_search"


async def _call(query: str, count: int) -> list[dict[str, Any]]:
    s = get_settings()
    headers = {"Authorization": f"Bearer {s.dashscope_api_key}"}
    # mcp>=2.x：请求头通过自建 http 客户端传入；总超时由外层 wait_for 控制
    async with create_mcp_http_client(headers=headers) as http_client:
        async with streamable_http_client(s.web_search_mcp_url, http_client=http_client) as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(TOOL_NAME, {"query": query, "count": count})
    if result.is_error:
        raise RuntimeError(f"联网搜索工具返回错误：{result.content}")
    texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
    if not texts:
        return []
    data = json.loads(texts[0])
    pages = data.get("pages", []) if isinstance(data, dict) else []
    return [_normalize(p) for p in pages if isinstance(p, dict)]


def _normalize(page: dict[str, Any]) -> dict[str, Any]:
    """不同版本字段名可能不同，统一成 title / url / snippet。"""
    return {
        "title": page.get("title") or page.get("hostname") or "网络来源",
        "url": page.get("url") or page.get("link") or "",
        "snippet": page.get("snippet") or page.get("content") or page.get("summary") or "",
    }


def search(query: str) -> list[dict[str, Any]]:
    s = get_settings()
    if not s.dashscope_api_key:
        logger.info("未配置 DASHSCOPE_API_KEY，跳过联网搜索")
        return []
    try:
        return asyncio.run(asyncio.wait_for(_call(query, s.web_search_top_k), timeout=40))
    except Exception as e:  # 降级
        logger.warning("联网搜索失败，已跳过：%s", e)
        return []
