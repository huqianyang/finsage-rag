"""导入流程的 6 个节点（架构设计 §4）。"""
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.common.content_types import label_of
from app.common.errors import StepError
from app.config import get_settings
from app.infra import llm, local_models, mineru, object_store, vector_store
from app.pipeline.importing.markdown_splitter import html_tables_to_text, split_markdown
from app.pipeline.importing.progress import step
from app.pipeline.importing.state import ImportState
from app.prompts.importing import METADATA_SYSTEM, METADATA_USER

logger = logging.getLogger(__name__)

IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
MAX_VLM_IMAGES = 30  # 单文档最多调用 VLM 的图片数
MIN_IMAGE_BYTES = 8 * 1024  # 过小的图片多为图标/印章，跳过摘要
# 文档自称类泛称，作为简称没有区分度，会干扰实体匹配
GENERIC_ALIASES = {"本公司", "公司", "本行", "我行", "本基金", "基金", "本产品", "产品", "本理财产品",
                   "理财产品", "本集团", "集团", "管理人", "本办法", "本报告"}
METADATA_FIELDS = ["document_title", "product_name", "product_code", "institution_name", "risk_level",
                   "industry", "market", "publish_date"]


# ---------- 1. 解析 ----------
@step("parse")
def parse_node(state: ImportState) -> dict:
    path = Path(state["local_path"])
    if state["file_ext"] == "md":
        text = path.read_text(encoding="utf-8", errors="replace")
        md_dir = path.parent
        msg = "Markdown 文件，直接读取"
    else:
        try:
            result = mineru.parse_file(path, Path(state["work_dir"]) / "mineru", data_id=state["doc_id"])
        except mineru.MinerUError as e:
            raise StepError("parse", f"解析文档失败：{e}") from e
        text = result.markdown_path.read_text(encoding="utf-8")
        md_dir = result.output_dir
        msg = f"共 {result.total_pages} 页" if result.total_pages else ""
    if not text.strip():
        raise StepError("parse", "解析文档失败：未提取到任何文字内容")
    text = html_tables_to_text(text)
    msg = "，".join(filter(None, [msg, f"约 {len(text)} 字"]))
    return {"markdown": text, "markdown_dir": str(md_dir), "_message": msg}


# ---------- 2. 图片 ----------
@step("image")
def image_node(state: ImportState) -> dict:
    md, md_dir, doc_id = state["markdown"], Path(state["markdown_dir"]), state["doc_id"]
    matches = list(IMAGE_RE.finditer(md))
    if not matches:
        return {"_message": "无图片"}

    # 本地存在的图片
    images: dict[str, Path] = {}
    for m in matches:
        rel = m.group(2)
        p = (md_dir / rel).resolve()
        if p.is_file():
            images[rel] = p

    # 上传 MinIO
    uploaded: dict[str, str] = {}
    for rel, p in images.items():
        name = object_store.image_object_name(doc_id, p.name)
        try:
            object_store.upload_file(name, p, content_type=f"image/{p.suffix.lstrip('.') or 'png'}")
            uploaded[rel] = f"minio://{get_settings().minio_bucket}/{name}"
        except Exception as e:
            logger.warning("图片上传失败 %s：%s", p.name, e)

    # VLM 摘要（并发 4，失败跳过）
    targets = [(rel, p) for rel, p in images.items() if p.stat().st_size >= MIN_IMAGE_BYTES][:MAX_VLM_IMAGES]

    def summarize(item):
        rel, p = item
        pos = md.find(rel)
        context = md[max(0, pos - 300): pos + 300]
        try:
            return rel, llm.describe_image(p, context)
        except Exception as e:
            logger.warning("图片摘要失败 %s：%s", p.name, e)
            return rel, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        summaries = dict(pool.map(summarize, targets))

    failed = sum(1 for v in summaries.values() if v is None)

    def replace(m: re.Match) -> str:
        rel = m.group(2)
        url = uploaded.get(rel, rel)
        summary = summaries.get(rel)
        if summary and "装饰性图片" not in summary:
            return f"![图片]({url})\n> 图片内容：{summary}\n"
        return f"![图片]({url})"

    new_md = IMAGE_RE.sub(replace, md)
    msg = f"{len(images)} 张图片，{len(targets)} 张生成摘要"
    if failed:
        msg += f"，{failed} 张摘要失败已跳过"
    return {"markdown": new_md, "_message": msg}


# ---------- 3. 切分 ----------
@step("split")
def split_node(state: ImportState) -> dict:
    s = get_settings()
    # 切分前去掉图片链接本身（摘要文字保留）
    text = IMAGE_RE.sub("", state["markdown"])
    chunks = split_markdown(text, target=s.chunk_target_chars, min_chars=s.chunk_min_chars,
                            max_chars=s.chunk_max_chars)
    chunks = [c for c in chunks if len(c.content.strip()) >= 10]
    if not chunks:
        raise StepError("split", "切分文档失败：没有有效文本片段")
    return {"chunks": [{"entry_name": c.entry_name, "content": c.content} for c in chunks],
            "_message": f"共 {len(chunks)} 个切片"}


# ---------- 4. 元数据 ----------
@step("metadata")
def metadata_node(state: ImportState) -> dict:
    head = state["markdown"][:4000]
    try:
        data = llm.chat_json([
            {"role": "system", "content": METADATA_SYSTEM},
            {"role": "user", "content": METADATA_USER.format(
                file_name=state["file_name"], content_type=label_of(state["content_type"]), head=head)},
        ], temperature=0, max_tokens=800)
    except Exception as e:
        raise StepError("metadata", f"抽取元数据失败：{e}") from e
    meta = {k: _clean(data.get(k)) for k in METADATA_FIELDS}
    if not meta["document_title"]:
        meta["document_title"] = Path(state["file_name"]).stem
    if meta["publish_date"] and not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", meta["publish_date"]):
        meta["publish_date"] = ""
    aliases = data.get("aliases") or []
    meta["aliases"] = [a for a in (_clean(x) for x in aliases if isinstance(x, str))
                       if a and a not in GENERIC_ALIASES][:5]
    filled = sum(1 for k in METADATA_FIELDS if meta[k])
    return {"metadata": meta, "_message": f"识别 {filled}/{len(METADATA_FIELDS)} 个字段"}


def _clean(v) -> str:
    return str(v).strip() if v not in (None, "null", "None") else ""


def _search_text(meta: dict, chunk: dict) -> str:
    return f"{meta['document_title']}\n{chunk['entry_name']}\n{chunk['content']}"


# ---------- 5. 向量 ----------
@step("embed")
def embed_node(state: ImportState) -> dict:
    meta = state["metadata"]
    texts = [_search_text(meta, c) for c in state["chunks"]]
    vectors = local_models.embed(texts)
    return {"vectors": vectors, "_message": f"{len(vectors)} 条向量"}


# ---------- 6. 入库 ----------
@step("index")
def index_node(state: ImportState) -> dict:
    meta, doc_id = state["metadata"], state["doc_id"]
    common = {
        "doc_id": doc_id, "content_type": state["content_type"],
        **{k: meta[k][:_LIMITS[k]] for k in METADATA_FIELDS},
        "source_file": state["file_name"][:512], "source_path": state["source_path"][:1024],
    }
    rows = []
    for i, (chunk, vec) in enumerate(zip(state["chunks"], state["vectors"])):
        rows.append({
            **common,
            "chunk_id": f"{doc_id}_{i:05d}", "chunk_index": i,
            "content": _truncate_bytes(chunk["content"], 16000),
            "search_text": _truncate_bytes(_search_text(meta, chunk), 16000),
            "entry_name": _truncate_bytes(chunk["entry_name"], 1000),
            "dense_vector": vec,
        })

    entity_text = " ".join(dict.fromkeys(filter(None, [
        meta["document_title"], meta["product_name"], meta["product_code"], meta["institution_name"],
        *meta.get("aliases", []),
    ])))
    doc_row = {
        "doc_id": doc_id, "content_type": state["content_type"],
        **{k: meta[k][:_LIMITS[k]] for k in ["document_title", "product_name", "product_code",
                                             "institution_name", "publish_date"]},
        "entity_text": _truncate_bytes(entity_text, 2000),
        "dense_vector": local_models.embed([entity_text])[0],
    }

    vector_store.delete_doc(doc_id)  # 覆盖导入：先删旧数据
    vector_store.insert_chunks(rows)
    vector_store.upsert_doc(doc_row)
    return {"_message": f"写入 {len(rows)} 个切片"}


_LIMITS = {"document_title": 500, "product_name": 250, "product_code": 60, "institution_name": 250,
           "risk_level": 30, "industry": 60, "market": 60, "publish_date": 30}


def _truncate_bytes(text: str, max_bytes: int) -> str:
    """Milvus VARCHAR 长度按 UTF-8 字节计算。"""
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", errors="ignore")
