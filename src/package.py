"""Build the code ZIP for upload: source, run instructions, dependencies, licence, results.

    ./venv/bin/python -m src.package                                  # code only
    ./venv/bin/python -m src.package --predictions v4_9fam_stack_selftrain   # + that run's CSV/JSON

The organisers ask for predictions and validation results, code, model details, run instructions
and a written report alongside it. Data files, caches and the virtual environment are left out.
"""
import argparse
import zipfile

from src.load import ROOT

INCLUDE = ["README.md", "LICENSE", "requirements.txt", "docs/TEAM_BRIEF.md", "notes/leaderboard.md"]
INCLUDE_DIRS = ["src", "colab", "notes"]
LIMIT = 8 * 1024 * 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", help="run name under submissions/ to include (CSV + JSON)")
    a = ap.parse_args()
    out = ROOT / "submissions" / "code.zip"
    out.parent.mkdir(exist_ok=True)
    files = [ROOT / f for f in INCLUDE if (ROOT / f).exists()]
    for d in INCLUDE_DIRS:
        files += sorted(f for f in (ROOT / d).rglob("*")
                        if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc"
                        and ".ipynb_checkpoints" not in f.parts)
    if a.predictions:
        for ext in (".csv", ".json"):
            f = ROOT / "submissions" / f"{a.predictions}{ext}"
            if f.exists():
                files.append(f)
    files = list(dict.fromkeys(files))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=False) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT))
    size = out.stat().st_size
    raw = sum(f.stat().st_size for f in files)
    assert size < LIMIT and raw < 64 * 1024 * 1024 and len(files) < 2000
    print(f"wrote {out}: {len(files)} files, {size / 1e3:.0f} kB zipped, {raw / 1e3:.0f} kB raw")


if __name__ == "__main__":
    main()
