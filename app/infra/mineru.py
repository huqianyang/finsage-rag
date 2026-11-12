"""MinerU 云 API（v4 精准解析）：本地文件 → Markdown + 图片。

流程：POST /file-urls/batch 取上传地址 → PUT 文件 → 轮询 GET /extract-results/batch/{batch_id}
      → state=done 时下载 full_zip_url 并解压。
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
    pass


@dataclass
class ParseResult:
    markdown_path: Path
    output_dir: Path
    total_pages: int | None


def _headers() -> dict[str, str]:
    token = get_settings().mineru_api_token
    if not token:
        raise MinerUError("未配置 MINERU_API_TOKEN")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _check(resp: httpx.Response) -> dict:
    resp.raise_for_status()
    body = resp.json()
    if body.get("code") != 0:
        raise MinerUError(f"MinerU 错误 {body.get('code')}：{body.get('msg')}")
    return body["data"]


def parse_file(file_path: Path, output_dir: Path, data_id: str) -> ParseResult:
    s = get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60) as client:
        # 1. 申请上传地址
        data = _check(client.post(
            f"{s.mineru_base_url}/file-urls/batch", headers=_headers(),
            json={"files": [{"name": file_path.name, "data_id": data_id}],
                  "model_version": s.mineru_model_version, "language": "ch",
                  "enable_table": True, "enable_formula": True},
        ))
        batch_id, upload_url = data["batch_id"], data["file_urls"][0]
        logger.info("MinerU batch_id=%s 上传 %s", batch_id, file_path.name)

        # 2. 上传文件（不设置 Content-Type）
        with file_path.open("rb") as f:
            put = client.put(upload_url, content=f.read(), timeout=300)
        put.raise_for_status()

        # 3. 轮询结果
        deadline = time.monotonic() + s.mineru_timeout_seconds
        result = None
        while time.monotonic() < deadline:
            time.sleep(s.mineru_poll_interval_seconds)
            data = _check(client.get(f"{s.mineru_base_url}/extract-results/batch/{batch_id}", headers=_headers()))
            items = data.get("extract_result") or []
            if not items:
                continue
            item = items[0]
            state = item.get("state")
            if state == "done":
                result = item
                break
            if state == "failed":
                raise MinerUError(f"MinerU 解析失败：{item.get('err_msg') or '未知原因'}")
        if result is None:
            raise MinerUError(f"MinerU 解析超时（>{s.mineru_timeout_seconds} 秒）")

        # 4. 下载并解压
        zip_resp = client.get(result["full_zip_url"], timeout=300, follow_redirects=True)
        zip_resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
        zf.extractall(output_dir)

    md_files = list(output_dir.rglob("full.md")) or list(output_dir.rglob("*.md"))
    if not md_files:
        raise MinerUError("MinerU 结果中没有 Markdown 文件")
    progress = result.get("extract_progress") or {}
    return ParseResult(markdown_path=md_files[0], output_dir=md_files[0].parent,
                       total_pages=progress.get("total_pages"))
