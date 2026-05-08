from pathlib import Path

def get_project_root():
    """向上查找名为 DistServe 的目录作为项目根目录"""
    current = Path(__file__).resolve().parent
    for parent in current.parents:
        if parent.name == "DistServe":
            return parent
    raise RuntimeError("找不到项目根目录 DistServe，请确认目录结构")

REPO_ROOT = get_project_root()