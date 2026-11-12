"""导入流程的状态定义。

LangGraph 中每个节点接收当前状态、返回需要更新的字段，框架把返回值合并进状态后
交给下一个节点。total=False 表示所有字段都是可选的：输入阶段只有前面一组字段，
中间结果由各节点逐步填充。
"""
from typing import Any, TypedDict


class ImportState(TypedDict, total=False):
    # ========== 输入（由导入服务在启动流程时提供） ==========
    task_id: str  # 导入任务 ID（import_tasks._id），节点用它更新进度
    doc_id: str  # 资料 ID（documents._id），同名覆盖导入时沿用旧 ID
    file_name: str  # 原始文件名
    file_ext: str  # 扩展名：pdf / doc / docx / md
    local_path: str  # 上传文件在本地临时目录中的路径
    work_dir: str  # 本任务的临时工作目录（解析结果也放在这里），流程结束后删除
    content_type: str  # 资料类型代码，上传时由管理员选择
    source_path: str  # 原始文件在 MinIO 中的对象路径

    # ========== 中间结果（由各节点依次填充） ==========
    markdown: str  # 解析出的 Markdown 全文（parse 生成，image 会改写其中的图片部分）
    markdown_dir: str  # Markdown 所在目录（Markdown 里的图片路径相对于这个目录）
    chunks: list[dict[str, Any]]  # 切片列表，每项 {entry_name: 章节路径, content: 正文}（split 生成）
    metadata: dict[str, Any]  # 资料元数据：产品名、代码、机构、风险等级等（metadata 生成）
    vectors: list[list[float]]  # 与 chunks 一一对应的稠密向量（embed 生成）
