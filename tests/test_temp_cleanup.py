"""临时目录清理的单元测试（不依赖外部服务：用 tmp_path 和假的 Mongo）。"""
import logging
from types import SimpleNamespace

from app.services import import_service

TASK_ID = "a" * 32  # 形如 task_id（32 位十六进制）的目录名
ACTIVE_ID = "b" * 32


def _make_task_dir(base, name):
    """造一个像导入工作目录的目录树：{name}/mineru/images/x.jpg。"""
    images = base / name / "mineru" / "images"
    images.mkdir(parents=True)
    (images / "x.jpg").write_bytes(b"x")
    return base / name


def test_remove_dir_deletes_whole_tree(tmp_path):
    d = _make_task_dir(tmp_path, TASK_ID)
    assert import_service._remove_dir(d) is True
    assert not d.exists()


def test_remove_dir_missing_path_is_ok(tmp_path):
    """目录本来就不存在：不报错，算删除成功。"""
    assert import_service._remove_dir(tmp_path / "nope") is True


def test_remove_dir_warns_instead_of_silently_ignoring(tmp_path, monkeypatch, caplog):
    """有文件删不掉（如被占用）时：返回 False 并记一条警告，而不是像 ignore_errors=True 那样悄悄吞掉。"""
    d = _make_task_dir(tmp_path, TASK_ID)

    def fake_rmtree(path, *, onexc):
        onexc(None, str(path / "mineru" / "x.jpg"), PermissionError("被占用"))

    monkeypatch.setattr(import_service.shutil, "rmtree", fake_rmtree)
    with caplog.at_level(logging.WARNING):
        assert import_service._remove_dir(d) is False
    assert "没能删除" in caplog.text and "PermissionError" in caplog.text


def test_sweep_temp_removes_only_inactive_task_dirs(tmp_path, monkeypatch):
    """清扫：只删"名字像 task_id 且不属于进行中任务"的目录；进行中任务的目录和别的东西都保留。"""
    leftover = _make_task_dir(tmp_path, TASK_ID)
    running = _make_task_dir(tmp_path, ACTIVE_ID)
    other = tmp_path / "keep_me"
    other.mkdir()
    monkeypatch.setattr(import_service, "get_settings", lambda: SimpleNamespace(temp_dir=tmp_path))
    fake_tasks = SimpleNamespace(find=lambda *a, **k: [{"_id": ACTIVE_ID}])
    monkeypatch.setattr(import_service.mongo, "import_tasks", lambda: fake_tasks)

    assert import_service.sweep_temp() == 1
    assert not leftover.exists()
    assert running.exists() and other.exists()
