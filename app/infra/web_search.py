"""联网搜索：调用阿里云百炼 WebSearch MCP 服务（Streamable HTTP 传输），工具名 bailian_web_search。

MCP（Model Context Protocol）是一种让应用调用外部"工具"的标准协议。这里作为 MCP 客户端：
建立会话 → 初始化 → 调用工具 → 解析返回的文本。

联网搜索是问答的第三路召回（另外两路是知识库稠密检索和 BM25 检索）。
未配置 DASHSCOPE_API_KEY 或调用失败时返回空列表（降级），不影响问答。
"""
import asyncio
import json
import logging
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from app.config import get_settings

logger = logging.getLogger(__name__)
TOOL_NAME = "bailian_web_search"  # 百炼联网搜索 MCP 提供的工具名，参数：query（搜索词）、count（条数）


async def _call(query: str, count: int) -> list[dict[str, Any]]:
    """建立 MCP 会话并调用搜索工具，返回统一格式的结果列表。"""
    s = get_settings()
    headers = {"Authorization": f"Bearer {s.dashscope_api_key}"}
    # mcp>=2.x：请求头通过自建 http 客户端传入；总超时由外层 wait_for 控制
    async with create_mcp_http_client(headers=headers) as http_client:
        async with streamable_http_client(s.web_search_mcp_url, http_client=http_client) as streams:
            read, write = streams[0], streams[1]  # 读流、写流
            async with ClientSession(read, write) as session:
                await session.initialize()  # MCP 握手
                result = await session.call_tool(TOOL_NAME, {"query": query, "count": count})
    if result.is_error:
        raise RuntimeError(f"联网搜索工具返回错误：{result.content}")
    # 工具返回的是文本内容，文本本身是 JSON：{"pages": [...], "request_id": ..., "status": ...}
    texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
    if not texts:
        return []
    data = json.loads(texts[0])
    pages = data.get("pages", []) if isinstance(data, dict) else []
    return [_normalize(p) for p in pages if isinstance(p, dict)]


def _normalize(page: dict[str, Any]) -> dict[str, Any]:
    """把一条搜索结果统一成 title / url / snippet 三个字段。

    百炼当前返回 title / url / snippet / hostname / hostlogo；这里兼容其他可能的字段名，
    防止接口升级后字段改名导致取不到内容。
    """
    return {
        "title": page.get("title") or page.get("hostname") or "网络来源",
        "url": page.get("url") or page.get("link") or "",
        "snippet": page.get("snippet") or page.get("content") or page.get("summary") or "",
    }


def search(query: str) -> list[dict[str, Any]]:
    """同步入口：执行一次联网搜索。任何异常都吞掉并返回空列表（降级）。

    MCP 客户端是异步的，这里用 asyncio.run 在当前线程新建事件循环执行；
    调用方（检索节点）运行在普通线程中，没有正在运行的事件循环，可以安全使用。
    """
    s = get_settings()
    if not s.dashscope_api_key:
        logger.info("未配置 DASHSCOPE_API_KEY，跳过联网搜索")
        return []
    try:
        # 40 秒总超时，防止网络异常时一直卡住整个问答
        return asyncio.run(asyncio.wait_for(_call(query, s.web_search_top_k), timeout=40))
    except Exception as e:  # 降级：联网搜索失败不影响知识库检索
        logger.warning("联网搜索失败，已跳过：%s", e)
        return []
