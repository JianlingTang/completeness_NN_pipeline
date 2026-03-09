#!/usr/bin/env python3
"""
CI script: static checks, tests, schema validation, parquet alignment, cluster_id join check.
Run from repo root. Use only changed files when git is available.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CLUSTER_PIPELINE = REPO_ROOT / "cluster_pipeline"
TESTS_DIR = REPO_ROOT / "tests"
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"
FAILED = []


def get_changed_python_files():
    """Files changed in latest commit or modified in working tree (exclude .venv)."""
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        names = [n.strip() for n in out.stdout.splitlines() if n.strip().endswith(".py")]
        out2 = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out2.returncode == 0:
            for n in out2.stdout.splitlines():
                n = n.strip()
                if n.endswith(".py") and n not in names:
                    names.append(n)
        paths = [REPO_ROOT / n for n in names if (REPO_ROOT / n).exists() and ".venv" not in n]
        return paths if paths else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def run(cmd, check=True, capture=False):
    if capture:
        r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
        return r.returncode, r.stdout, r.stderr
    r = subprocess.run(cmd, cwd=REPO_ROOT, timeout=300)
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)
    return r.returncode


def step_flake8():
    print("\n--- 1. Flake8 ---")
    code, out, err = run(
        [
            str(VENV_PY), "-m", "flake8",
            str(CLUSTER_PIPELINE), str(TESTS_DIR),
            "--exclude=.venv", "--max-line-length=120", "--extend-ignore=E203,W503",
        ],
        check=False,
        capture=True,
    )
    if code != 0:
        FAILED.append("flake8")
        print(out or err)
        return False
    print("OK")
    return True


def step_mypy():
    print("\n--- 2. Mypy ---")
    code, out, err = run(
        [str(VENV_PY), "-m", "mypy", str(CLUSTER_PIPELINE), "--no-error-summary"],
        check=False,
        capture=True,
    )
    if code != 0:
        FAILED.append("mypy")
        print(out or err)
        return False
    print("OK")
    return True


def step_pytest():
    print("\n--- 3. Pytest ---")
    code = run([str(VENV_PY), "-m", "pytest", str(TESTS_DIR), "-v", "--tb=short"], check=False)
    if code != 0:
        FAILED.append("pytest")
        return False
    print("OK")
    return True


def step_schema_validation():
    print("\n--- 4. Dataset schemas (injected_clusters, matched_clusters, photometry, labels) ---")
    sys.path.insert(0, str(REPO_ROOT))
    from cluster_pipeline.data.schemas import get_required_columns
    for name in ["injected_clusters", "matched_clusters", "photometry", "labels"]:
        cols = get_required_columns(name)
        if not cols or "cluster_id" not in cols:
            FAILED.append("schema_validation")
            print(f"FAIL: schema '{name}' missing or missing cluster_id")
            return False
    print("OK")
    return True


def step_parquet_schemas():
    print("\n--- 5. Parquet schemas match expected ---")
    code = run(
        [str(VENV_PY), "-m", "pytest", str(TESTS_DIR / "test_schemas.py"), "-v", "--tb=short"],
        check=False,
    )
    if code != 0:
        FAILED.append("parquet_schemas")
        return False
    print("OK")
    return True


def step_cluster_id_joins():
    print("\n--- 6. cluster_id in joins ---")
    for path in [CLUSTER_PIPELINE / "catalogue" / "catalogue_filters.py", CLUSTER_PIPELINE / "dataset" / "dataset_builder.py"]:
        if not path.exists():
            continue
        text = path.read_text()
        if ".merge(" not in text:
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if ".merge(" in line:
                # Look at a larger block: merge key may be on=base or on=[...] a few lines above/below
                start = max(0, i - 8)
                block = "\n".join(lines[start : i + 5])
                if "cluster_id" not in block and "on=" in block:
                    FAILED.append("cluster_id_joins")
                    print(f"FAIL: possible merge without cluster_id at {path}:{i+1}")
                    return False
    print("OK (all merges use cluster_id in join keys)")
    return True


def main():
    print("CI: static checks, tests, schemas, parquet, cluster_id joins")
    changed = get_changed_python_files()
    if changed:
        print("Changed/modified .py files:", [str(p.relative_to(REPO_ROOT)) for p in changed[:20]])
    else:
        print("No git changes (or not a git repo); running on full cluster_pipeline + tests")

    step_flake8()
    step_mypy()
    step_pytest()
    step_schema_validation()
    step_parquet_schemas()
    step_cluster_id_joins()

    if FAILED:
        print("\n*** CI FAILED ***", FAILED)
        sys.exit(1)
    print("\n*** CI PASSED ***")
    sys.exit(0)


if __name__ == "__main__":
    main()
