"""MinerU 云 API（v4 精准解析）：把本地 PDF / DOC / DOCX 解析成 Markdown + 图片。

调用流程（官方"本地文件批量上传"方式）：
1. POST /file-urls/batch            申请上传地址，得到 batch_id 和预签名上传 URL
2. PUT  <上传 URL>                   上传文件内容；上传完成后 MinerU 自动开始解析，无需再提交任务
3. GET  /extract-results/batch/{id}  轮询解析状态，直到 state 为 done 或 failed
4. 下载 full_zip_url 并解压，得到 full.md 和 images/ 目录
"""
import io
import logging
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class MinerUError(Exception):
    """MinerU 调用失败（未配置 Token、接口报错、解析失败、超时等），message 可直接展示给用户。"""


@dataclass
class ParseResult:
    """解析结果。"""
    markdown_path: Path  # 解压后的 Markdown 文件路径
    output_dir: Path  # Markdown 所在目录；文中的图片是相对这个目录的路径
    total_pages: int | None  # 文档总页数（接口未返回时为 None）


def _headers() -> dict[str, str]:
    """MinerU 接口的请求头（Bearer Token 鉴权）。"""
    token = get_settings().mineru_api_token
    if not token:
        raise MinerUError("未配置 MINERU_API_TOKEN")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _check(resp: httpx.Response) -> dict:
    """检查 MinerU 响应：HTTP 状态码正常且业务 code == 0 才算成功，返回其中的 data 部分。"""
    resp.raise_for_status()
    body = resp.json()
    if body.get("code") != 0:
        raise MinerUError(f"MinerU 错误 {body.get('code')}：{body.get('msg')}")
    return body["data"]


def parse_file(file_path: Path, output_dir: Path, data_id: str) -> ParseResult:
    """解析单个本地文件，结果解压到 output_dir。

    Args:
        file_path: 本地文件路径（文件名必须带正确的扩展名，MinerU 据此判断文件类型）
        output_dir: 解析结果的解压目录
        data_id: 业务方自定义 ID（这里传 doc_id），便于在 MinerU 控制台对照

    Raises:
        MinerUError: 任一环节失败
    """
    s = get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60) as client:
        # 1. 申请上传地址
        data = _check(client.post(
            f"{s.mineru_base_url}/file-urls/batch", headers=_headers(),
            # enable_table：识别表格（输出为 HTML 表格）；enable_formula：识别公式；language=ch：中文 OCR
            json={"files": [{"name": file_path.name, "data_id": data_id}],
                  "model_version": s.mineru_model_version, "language": "ch",
                  "enable_table": True, "enable_formula": True},
        ))
        batch_id, upload_url = data["batch_id"], data["file_urls"][0]
        logger.info("MinerU batch_id=%s 上传 %s", batch_id, file_path.name)

        # 2. 上传文件。注意不能带 Content-Type 请求头，否则预签名 URL 校验会失败
        with file_path.open("rb") as f:
            put = client.put(upload_url, content=f.read(), timeout=300)
        put.raise_for_status()

        # 3. 轮询解析结果，直到完成、失败或超时
        deadline = time.monotonic() + s.mineru_timeout_seconds
        result = None
        while time.monotonic() < deadline:
            time.sleep(s.mineru_poll_interval_seconds)
            data = _check(client.get(f"{s.mineru_base_url}/extract-results/batch/{batch_id}", headers=_headers()))
            items = data.get("extract_result") or []
            if not items:  # 文件还没被系统扫描到，继续等
                continue
            item = items[0]
            state = item.get("state")  # waiting-file / pending / running / converting / done / failed
            if state == "done":
                result = item
                break
            if state == "failed":
                raise MinerUError(f"MinerU 解析失败：{item.get('err_msg') or '未知原因'}")
        if result is None:
            raise MinerUError(f"MinerU 解析超时（>{s.mineru_timeout_seconds} 秒）")

        # 4. 下载结果压缩包（在内存中处理，不落盘）
        zip_resp = client.get(result["full_zip_url"], timeout=300, follow_redirects=True)
        zip_resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
        zf.extractall(output_dir)

    # 优先使用约定的 full.md；万一文件名不同，退而取任意 .md 文件
    md_files = list(output_dir.rglob("full.md")) or list(output_dir.rglob("*.md"))
    if not md_files:
        raise MinerUError("MinerU 结果中没有 Markdown 文件")
    progress = result.get("extract_progress") or {}
    return ParseResult(markdown_path=md_files[0], output_dir=md_files[0].parent,
                       total_pages=progress.get("total_pages"))
