"""批量导入：按目录对应资料类型，调用导入服务的上传接口并等待完成。

用法：uv run python -m scripts.batch_import <数据根目录> [--only 子目录名]
"""
import argparse
import sys
import time
from pathlib import Path

import httpx

API = "http://127.0.0.1:8000"
# 首批资料目录 → 资料类型（数据库设计 §1）
DIR_TYPES = {
    "基金产品": "fund_product",
    "银行理财&风险揭示书": "risk_disclosure",
    "上市公司年报": "company_report",
    "宏观经济&政策资料": "macro_policy",
    "用户FAQ": "faq",
}
EXTS = {".pdf", ".doc", ".docx", ".md"}


def upload(client: httpx.Client, files: list[Path], content_type: str) -> list[dict]:
    handles = [("files", (f.name, f.open("rb"))) for f in files]
    try:
        r = client.post(f"{API}/api/documents", data={"content_type": content_type}, files=handles, timeout=300)
    finally:
        for _, (_, fh) in handles:
            fh.close()
    r.raise_for_status()
    return r.json()["items"]


def wait(client: httpx.Client, tasks: dict[str, str]) -> dict[str, dict]:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--only", help="只导入某个子目录")
    args = ap.parse_args()
    root = Path(args.root)
    tasks: dict[str, str] = {}
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
