"""批量导入：按子目录对应资料类型，调用导入服务的上传接口，并等待全部任务完成。

用法：uv run python -m scripts.batch_import <数据根目录> [--only 子目录名]
前提：导入服务（8000 端口）已启动。

为什么用脚本而不是 curl：Windows 上 curl 发送的中文文件名是 GBK 编码，服务端会解析成乱码；
httpx 按 UTF-8 发送文件名，和浏览器上传一致。
"""
import argparse
import sys
import time
from pathlib import Path

import httpx

API = "http://127.0.0.1:8000"  # 导入服务地址
# 首批资料目录 → 资料类型（数据库设计 §1）；数据根目录下必须是这几个子目录
DIR_TYPES = {
    "基金产品": "fund_product",
    "银行理财&风险揭示书": "risk_disclosure",
    "上市公司年报": "company_report",
    "宏观经济&政策资料": "macro_policy",
    "用户FAQ": "faq",
}
EXTS = {".pdf", ".doc", ".docx", ".md"}  # 只上传这些格式，其他文件忽略


def upload(client: httpx.Client, files: list[Path], content_type: str) -> list[dict]:
    """一次请求上传同一类型的多个文件，返回每个文件的受理结果。"""
    handles = [("files", (f.name, f.open("rb"))) for f in files]
    try:
        r = client.post(f"{API}/api/documents", data={"content_type": content_type}, files=handles, timeout=300)
    finally:
        # 无论成功与否都关闭文件句柄
        for _, (_, fh) in handles:
            fh.close()
    r.raise_for_status()
    return r.json()["items"]


def wait(client: httpx.Client, tasks: dict[str, str]) -> dict[str, dict]:
    """每 5 秒轮询一次所有未完成的任务，每完成一个打印一行结果（含各步骤说明），全部完成后返回。"""
    done: dict[str, dict] = {}
    while len(done) < len(tasks):
        for task_id, name in tasks.items():
            if task_id in done:
                continue
            t = client.get(f"{API}/api/import-tasks/{task_id}").json()
            if t["status"] in ("completed", "failed"):
                done[task_id] = t
                steps = " / ".join(f"{s['label']}:{s['message']}" for s in t["steps"] if s["message"])
                flag = "OK  " if t["status"] == "completed" else "FAIL"
                print(f"[{flag}] {name}  {steps}  {(t.get('error') or {}).get('message', '')}", flush=True)
        time.sleep(5)
    return done


def main():
    """解析命令行参数，按目录依次上传，然后等待全部完成并汇总；有失败时退出码为 1。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--only", help="只导入某个子目录")
    args = ap.parse_args()
    root = Path(args.root)
    tasks: dict[str, str] = {}  # task_id → 文件名
    with httpx.Client() as client:
        for sub, ctype in DIR_TYPES.items():
            if args.only and args.only != sub:
                continue
            files = sorted(p for p in (root / sub).glob("*") if p.suffix.lower() in EXTS)
            if not files:
                continue
            for item in upload(client, files, ctype):
                if item["accepted"]:
                    tasks[item["task_id"]] = item["file_name"]
                    print(f"已提交 {item['file_name']}（{ctype}）{' 覆盖' if item['overwrite'] else ''}", flush=True)
                else:
                    print(f"[拒绝] {item['file_name']}：{item['message']}", flush=True)
        results = wait(client, tasks)
    failed = [t for t in results.values() if t["status"] == "failed"]
    print(f"\n完成 {len(results) - len(failed)} / {len(results)}，失败 {len(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
