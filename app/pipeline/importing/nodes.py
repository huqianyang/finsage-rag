"""导入流程的 6 个节点（架构设计文档 §4）。

每个节点：接收 ImportState，返回需要更新的字段（字典）。
都用 `@step("编码")` 装饰，开始/完成/失败会自动记录到导入任务进度中；
返回值里的 `_message` 是给导入页显示的步骤说明（如"共 18 个切片"），会被装饰器取走。
"""
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

# 匹配 Markdown 图片：![说明文字](图片路径)，group(2) 是图片路径
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
MAX_VLM_IMAGES = 30  # 单文档最多调用 VLM 的图片数（控制费用与耗时）
MIN_IMAGE_BYTES = 8 * 1024  # 过小的图片多为图标/印章，跳过摘要
# 文档自称类泛称，作为简称没有区分度，会干扰实体匹配
GENERIC_ALIASES = {"本公司", "公司", "本行", "我行", "本基金", "基金", "本产品", "产品", "本理财产品",
                   "理财产品", "本集团", "集团", "管理人", "本办法", "本报告"}
# 元数据字段（对应需求说明 §5），会冗余写到每个切片上
METADATA_FIELDS = ["document_title", "product_name", "product_code", "institution_name", "risk_level",
                   "industry", "market", "publish_date"]


# ==================== 1. 解析：文件 → Markdown ====================
@step("parse")
def parse_node(state: ImportState) -> dict:
    """把上传文件转成 Markdown 文本。

    - .md 文件直接读取；
    - .pdf / .doc / .docx 调用 MinerU 云 API 解析；
    - 解析完成后把 HTML 表格转成逐行纯文本，避免后续切分时把表格切断。
    """
    path = Path(state["local_path"])
    if state["file_ext"] == "md":
        # errors="replace"：遇到非 UTF-8 字节用替换字符代替，不因个别乱码导致整个导入失败
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
        # 例如纯扫描件识别失败、空文件
        raise StepError("parse", "解析文档失败：未提取到任何文字内容")
    text = html_tables_to_text(text)
    msg = "，".join(filter(None, [msg, f"约 {len(text)} 字"]))
    return {"markdown": text, "markdown_dir": str(md_dir), "_message": msg}


# ==================== 2. 图片：上传 MinIO + 生成文字摘要 ====================
@step("image")
def image_node(state: ImportState) -> dict:
    """处理 Markdown 中的图片。

    文档里的图表（如资产配置饼图、净值走势图）包含重要信息，但向量检索只能处理文字，
    所以调用视觉模型给图片生成文字摘要，写在图片下方，让这些信息也能被检索到。

    任何一张图片上传或摘要失败都只记日志、跳过该图，不会让整个导入失败。
    """
    md, md_dir, doc_id = state["markdown"], Path(state["markdown_dir"]), state["doc_id"]
    matches = list(IMAGE_RE.finditer(md))
    if not matches:
        return {"_message": "无图片"}

    # 1) 找出本地确实存在的图片：相对路径 → 本地绝对路径（同一张图被引用多次只处理一次）
    images: dict[str, Path] = {}
    for m in matches:
        rel = m.group(2)
        p = (md_dir / rel).resolve()
        if p.is_file():
            images[rel] = p

    # 2) 上传 MinIO：相对路径 → minio:// 地址（上传失败的保留原相对路径）
    uploaded: dict[str, str] = {}
    for rel, p in images.items():
        name = object_store.image_object_name(doc_id, p.name)
        try:
            object_store.upload_file(name, p, content_type=f"image/{p.suffix.lstrip('.') or 'png'}")
            uploaded[rel] = f"minio://{get_settings().minio_bucket}/{name}"
        except Exception as e:
            logger.warning("图片上传失败 %s：%s", p.name, e)

    # 3) VLM 摘要：跳过过小的图片，最多处理 MAX_VLM_IMAGES 张；4 个线程并发调用以缩短耗时
    targets = [(rel, p) for rel, p in images.items() if p.stat().st_size >= MIN_IMAGE_BYTES][:MAX_VLM_IMAGES]

    def summarize(item):
        """为一张图片生成摘要，返回 (相对路径, 摘要或 None)。"""
        rel, p = item
        # 取图片引用位置前后各 300 字作为上下文，帮助模型理解图片说的是什么
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
        """把原图片引用替换成 MinIO 地址，并在下方追加"图片内容：摘要"（装饰性图片不追加）。"""
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


# ==================== 3. 切分：Markdown → 切片 ====================
@step("split")
def split_node(state: ImportState) -> dict:
    """按标题结构切分（规则见 markdown_splitter.py），并过滤掉过短的无效切片。"""
    s = get_settings()
    # 切分前去掉图片链接本身（摘要文字保留）：链接对检索没有意义
    text = IMAGE_RE.sub("", state["markdown"])
    chunks = split_markdown(text, target=s.chunk_target_chars, min_chars=s.chunk_min_chars,
                            max_chars=s.chunk_max_chars)
    # 不足 10 个字的切片（如孤立的页码、符号）没有检索价值
    chunks = [c for c in chunks if len(c.content.strip()) >= 10]
    if not chunks:
        raise StepError("split", "切分文档失败：没有有效文本片段")
    return {"chunks": [{"entry_name": c.entry_name, "content": c.content} for c in chunks],
            "_message": f"共 {len(chunks)} 个切片"}


# ==================== 4. 元数据：大模型抽取 ====================
@step("metadata")
def metadata_node(state: ImportState) -> dict:
    """用大模型从文档开头抽取元数据：资料名、产品名、代码、机构、风险等级、发布日期、简称等。

    只取前 4000 字：这些信息通常都在文档开头（标题、产品概况表），取全文既慢又贵。
    抽取结果会做清洗：日期格式不对的置空，泛称简称（"本行"等）去掉。
    """
    head = state["markdown"][:4000]
    try:
        data = llm.chat_json([
            {"role": "system", "content": METADATA_SYSTEM},
            {"role": "user", "content": METADATA_USER.format(
                file_name=state["file_name"], content_type=label_of(state["content_type"]), head=head)},
        ], temperature=0, max_tokens=800)  # temperature=0：抽取任务要求输出稳定
    except Exception as e:
        raise StepError("metadata", f"抽取元数据失败：{e}") from e
    meta = {k: _clean(data.get(k)) for k in METADATA_FIELDS}
    if not meta["document_title"]:
        # 模型没给出标题时，用去掉扩展名的文件名兜底
        meta["document_title"] = Path(state["file_name"]).stem
    # 只接受 YYYY-MM-DD 或 YYYY-MM，其他格式（如"2026年一季度"）宁可置空，也不存不规范的值
    if meta["publish_date"] and not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", meta["publish_date"]):
        meta["publish_date"] = ""
    meta["aliases"] = _parse_aliases(data.get("aliases"))
    filled = sum(1 for k in METADATA_FIELDS if meta[k])
    return {"metadata": meta, "_message": f"识别 {filled}/{len(METADATA_FIELDS)} 个字段"}


def _clean(v) -> str:
    """把模型输出的字段值规范成字符串；None 以及字符串 "null"/"None" 一律当作空。"""
    return str(v).strip() if v not in (None, "null", "None") else ""


def _parse_aliases(raw) -> list[str]:
    """把模型返回的 aliases 规整成字符串列表（最多 5 个，去掉空值和"本行"之类的泛称）。

    提示词要求返回数组；万一模型返回成一个字符串（如 "招商银行,招行"），就按常见分隔符拆开，
    而不是让 for 循环把它拆成一个个单字；其他类型一律当作没有。
    """
    if isinstance(raw, str):
        raw = re.split(r"[,，、;；]+", raw)
    if not isinstance(raw, list):
        return []
    items = (_clean(x) for x in raw if isinstance(x, str))
    return [a for a in items if a and a not in GENERIC_ALIASES][:5]


def _search_text(meta: dict, chunk: dict) -> str:
    """切片的检索文本 = 资料名 + 章节路径 + 正文。

    把资料名和章节名拼进去，让"华夏债券C 的赎回费"这类问题能同时匹配到资料和章节，
    而不只是依赖正文里有没有出现产品名。稠密向量和 BM25 都基于这段文本。
    """
    return f"{meta['document_title']}\n{chunk['entry_name']}\n{chunk['content']}"


# ==================== 5. 向量：本地 bge-m3 编码 ====================
@step("embed")
def embed_node(state: ImportState) -> dict:
    """把所有切片的检索文本编码成稠密向量（顺序与 chunks 一致）。"""
    meta = state["metadata"]
    texts = [_search_text(meta, c) for c in state["chunks"]]
    vectors = local_models.embed(texts)
    return {"vectors": vectors, "_message": f"{len(vectors)} 条向量"}


# ==================== 6. 入库：写入 Milvus ====================
@step("index")
def index_node(state: ImportState) -> dict:
    """写入两个集合：fin_chunks（全部切片）和 fin_docs（本资料的一条实体记录）。

    覆盖导入时先删除该 doc_id 的旧数据再写入，保证不会出现新旧切片混在一起。
    字段长度按 Milvus 集合定义截断，避免超长导致整批写入失败。
    """
    meta, doc_id = state["metadata"], state["doc_id"]
    # 每个切片都相同的字段（资料级元数据）
    common = {
        "doc_id": doc_id, "content_type": state["content_type"],
        **{k: _fit(k, meta[k]) for k in METADATA_FIELDS},
        "source_file": _fit("source_file", state["file_name"]),
        "source_path": _fit("source_path", state["source_path"]),
    }
    rows = []
    for i, (chunk, vec) in enumerate(zip(state["chunks"], state["vectors"])):
        rows.append({
            **common,
            # 切片 ID：资料 ID + 5 位序号，如 9302fe..._00003
            "chunk_id": f"{doc_id}_{i:05d}", "chunk_index": i,
            "content": _truncate_bytes(chunk["content"], 16000),
            "search_text": _truncate_bytes(_search_text(meta, chunk), 16000),
            "entry_name": _truncate_bytes(chunk["entry_name"], 1000),
            "dense_vector": vec,
        })

    # 实体文本：资料名 + 产品名 + 代码 + 机构 + 简称，去重并保持顺序（dict.fromkeys 去重保序）
    entity_text = " ".join(dict.fromkeys(filter(None, [
        meta["document_title"], meta["product_name"], meta["product_code"], meta["institution_name"],
        *meta.get("aliases", []),
    ])))
    doc_row = {
        "doc_id": doc_id, "content_type": state["content_type"],
        **{k: _fit(k, meta[k]) for k in ["document_title", "product_name", "product_code",
                                         "institution_name", "publish_date"]},
        "entity_text": _truncate_bytes(entity_text, 2000),
        "dense_vector": local_models.embed([entity_text])[0],
    }

    vector_store.delete_doc(doc_id)  # 覆盖导入：先删旧数据
    vector_store.insert_chunks(rows)
    vector_store.upsert_doc(doc_row)
    return {"_message": f"写入 {len(rows)} 个切片"}


# 各字段写入 Milvus 前的最大 UTF-8 字节数，与 vector_store 中集合定义的 VARCHAR max_length 一致。
# 注意是字节而不是字符：一个汉字占 3 字节，按字符截断的话，170 个汉字以上的资料名就会超长导致整批写入失败
_LIMITS = {"document_title": 512, "product_name": 256, "product_code": 64, "institution_name": 256,
           "risk_level": 32, "industry": 64, "market": 64, "publish_date": 32,
           "source_file": 512, "source_path": 1024}


def _fit(field: str, value: str) -> str:
    """把字段值截断到该字段在 Milvus 中允许的字节长度内。"""
    return _truncate_bytes(value, _LIMITS[field])


def _truncate_bytes(text: str, max_bytes: int) -> str:
    """按 UTF-8 字节数截断文本（Milvus VARCHAR 长度按字节计算，一个汉字 3 字节）。

    直接按字节截断可能切在一个汉字中间，decode 时用 errors="ignore" 丢掉不完整的半个字。
    """
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", errors="ignore")
