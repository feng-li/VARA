from pathlib import Path
import subprocess
import sys
import shutil
import importlib.util

try:
    from ._bootstrap import REPO_ROOT
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[1]


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def extract_req_name(line: str) -> str:
    s = line.strip()
    for sep in ["==", ">=", "<=", ">", "<", "~=", "["]:
        if sep in s:
            s = s.split(sep)[0]
    return normalize_name(s)


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def run_pipreqs(project_dir: str = ".") -> bool:
    if not has_module("pipreqs"):
        print(
            "Error: pipreqs is not installed in the current Python environment.\n"
            "Install it with:\n"
            f"  {sys.executable} -m pip install pipreqs"
        )
        return False

    pipreqs_exe = shutil.which("pipreqs")
    if pipreqs_exe is None:
        print(
            "Error: pipreqs is installed, but the 'pipreqs' executable is not on PATH.\n"
            "Try running:\n"
            f"  {sys.executable} -m pip install --force-reinstall pipreqs"
        )
        return False

    try:
        subprocess.run(
            [pipreqs_exe, project_dir, "--force"],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: pipreqs failed with exit code {e.returncode}")
        return False


def get_pip_freeze_lines() -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def merge_with_freeze(requirements_file: str = "requirements.txt") -> None:
    req_path = Path(requirements_file)

    if not req_path.exists():
        raise FileNotFoundError(f"{requirements_file} not found")

    reqs = req_path.read_text(encoding="utf-8").splitlines()
    frozen = get_pip_freeze_lines()

    freeze_map = {}
    for line in frozen:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "==" in s:
            pkg_name = normalize_name(s.split("==", 1)[0])
            freeze_map[pkg_name] = s

    merged = []
    for line in reqs:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        pkg = extract_req_name(s)
        merged.append(freeze_map.get(pkg, s))

    req_path.write_text("\n".join(merged) + "\n", encoding="utf-8")


def main():
    if not run_pipreqs(str(REPO_ROOT)):
        sys.exit(1)
    merge_with_freeze(str(REPO_ROOT / "requirements.txt"))
    print("requirements.txt updated successfully.")


if __name__ == "__main__":
    main()
