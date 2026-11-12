from typing import Any, TypedDict


class ImportState(TypedDict, total=False):
    # 输入
    task_id: str
    doc_id: str
    file_name: str
    file_ext: str  # pdf / doc / docx / md
    local_path: str  # 本地临时文件
    work_dir: str  # 本任务工作目录
    content_type: str
    source_path: str  # MinIO 对象路径
    # 中间结果
    markdown: str
    markdown_dir: str  # Markdown 所在目录（图片相对路径基准）
    chunks: list[dict[str, Any]]  # {entry_name, content}
    metadata: dict[str, Any]
    vectors: list[list[float]]
