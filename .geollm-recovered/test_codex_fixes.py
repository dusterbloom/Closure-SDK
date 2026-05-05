#!/usr/bin/env python3
"""TDD coverage for the three Codex P2/P3 review items.

Each test maps to one Codex finding:

  1. fisher_sweep.load_real_corpus reads bundled corpora at
     ./corpora/, not /home/claude/...
  2. fisher_sweep main()'s save path resolves to a writable location next
     to the script (creating the parent dir if needed), not to a hardcoded
     /mnt/user-data/... that doesn't exist on a normal checkout.
  3. The repo .gitignore allowlist exposes .geollm-recovered/*.py but
     KEEPS *.out and corpora/ ignored, matching the repo's
     "ignore everything; allowlist what ships" policy.
"""
import importlib.util
import os
import subprocess
from pathlib import Path

THIS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = THIS_DIR.parent
SCRIPT_PATH = THIS_DIR / "01_fisher_sweep.py"


def _load_fisher_sweep():
    """Import 01_fisher_sweep.py despite the leading digit."""
    spec = importlib.util.spec_from_file_location("fisher_sweep_mod", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Test 1: corpus loader uses bundled local path ─────────────────────
def test_load_real_corpus_uses_bundled_corpora():
    """load_real_corpus must successfully read ./corpora/*.txt and return
    a non-empty lowercased string. Should NOT depend on /home/claude/... ."""
    corpora_dir = THIS_DIR / "corpora"
    assert (corpora_dir / "shakespeare.txt").exists(), \
        "test prerequisite: corpora/shakespeare.txt must exist"
    assert (corpora_dir / "pride_prejudice.txt").exists(), \
        "test prerequisite: corpora/pride_prejudice.txt must exist"

    mod = _load_fisher_sweep()
    text = mod.load_real_corpus()

    assert isinstance(text, str) and len(text) > 100_000, \
        f"expected substantial corpus text, got {len(text) if isinstance(text,str) else type(text)} chars"
    assert text == text.lower(), "corpus must be lowercased"
    # Confirm both sources are present (Shakespeare-only would mean P&P silently dropped).
    # Use markers unique to each work that survive the lowercasing.
    assert "thou" in text, "Shakespeare corpus appears to be missing (no 'thou')"
    assert "elizabeth" in text or "darcy" in text or "bennet" in text, \
        "Pride and Prejudice corpus appears to be missing"


# ── Test 2: save path resolves locally and is writable ────────────────
def test_save_path_is_local_and_creates_parent():
    """The fisher_sweep module must expose a results path that lives
    next to the script (not /mnt/user-data/...) and whose parent can be
    created with mkdir(parents=True, exist_ok=True)."""
    mod = _load_fisher_sweep()
    # We expect a function or attribute that resolves the output path.
    # The minimal contract: a callable `results_path()` returning Path
    # under THIS_DIR; OR a module attribute `RESULTS_PATH` that is a Path
    # under THIS_DIR.
    if hasattr(mod, "results_path") and callable(mod.results_path):
        out = Path(mod.results_path())
    elif hasattr(mod, "RESULTS_PATH"):
        out = Path(mod.RESULTS_PATH)
    else:
        raise AssertionError(
            "fisher_sweep must expose results_path() or RESULTS_PATH so "
            "callers and tests can override the destination instead of "
            "writing to a hardcoded environment-specific absolute path."
        )

    out = out.resolve()
    # Must live under .geollm-recovered (the script's directory tree).
    assert THIS_DIR in out.parents or out.parent == THIS_DIR, \
        f"results path {out} must be under {THIS_DIR}"
    # The parent must be createable; mkdir is idempotent so this is safe.
    out.parent.mkdir(parents=True, exist_ok=True)
    assert out.parent.exists() and out.parent.is_dir(), \
        f"parent dir {out.parent} should be createable"
    # And must be writable (touch + delete the file path).
    try:
        out.write_text("{}")
        assert out.exists()
    finally:
        if out.exists():
            out.unlink()


# ── Test 3: .gitignore allowlist is narrow ────────────────────────────
def _git_check_ignore(path):
    """Returns True if `path` is currently ignored by git."""
    rel = str(path.relative_to(REPO_ROOT))
    res = subprocess.run(
        ["git", "check-ignore", "-q", rel],
        cwd=REPO_ROOT, capture_output=True
    )
    # exit 0 = ignored, exit 1 = not ignored, other = error
    return res.returncode == 0


def test_gitignore_allows_python_but_not_outputs_or_corpora():
    """In the .geollm-recovered/ directory:
       - *.py files must be tracked (NOT ignored)
       - *.out files must remain ignored
       - corpora/ raw text files must remain ignored
    This matches the repo's allowlist policy: only ship code.
    """
    repo_dir = REPO_ROOT / ".geollm-recovered"
    py_file  = repo_dir / "01_fisher_sweep.py"
    out_file = repo_dir / "world_model.out"
    corp_file = repo_dir / "corpora" / "shakespeare.txt"

    assert py_file.exists(),  "test prerequisite: 01_fisher_sweep.py must exist"
    assert out_file.exists(), "test prerequisite: world_model.out must exist"
    assert corp_file.exists(),"test prerequisite: corpora/shakespeare.txt must exist"

    assert not _git_check_ignore(py_file), \
        f"{py_file.relative_to(REPO_ROOT)} should be tracked, but is ignored"
    assert _git_check_ignore(out_file), \
        f"{out_file.relative_to(REPO_ROOT)} should be ignored (it's a generated output)"
    assert _git_check_ignore(corp_file), \
        f"{corp_file.relative_to(REPO_ROOT)} should be ignored (raw corpus, not project code)"


if __name__ == "__main__":
    # Allow running without pytest; report each failure inline.
    failures = []
    for name, fn in list(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failures.append((name, str(e)))
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    if failures:
        print(f"\n{len(failures)} failure(s)")
        raise SystemExit(1)
    print("\nAll tests passed")
