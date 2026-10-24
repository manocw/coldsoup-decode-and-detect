"""Build the code ZIP for upload: source, docs, requirements, licence. No data, cache or venv.

    ./venv/bin/python -m src.package            # -> submissions/code.zip
"""
import zipfile

from src.load import ROOT

INCLUDE = ["README.md", "LICENSE", "requirements.txt", "CLAUDE.md"]
INCLUDE_DIRS = ["src"]
LIMIT = 8 * 1024 * 1024


def main():
    out = ROOT / "submissions" / "code.zip"
    out.parent.mkdir(exist_ok=True)
    files = [ROOT / f for f in INCLUDE if (ROOT / f).exists()]
    for d in INCLUDE_DIRS:
        files += sorted(f for f in (ROOT / d).rglob("*")
                        if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=False) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT))
    size = out.stat().st_size
    raw = sum(f.stat().st_size for f in files)
    assert size < LIMIT and raw < 64 * 1024 * 1024 and len(files) < 2000
    print(f"wrote {out}: {len(files)} files, {size / 1e3:.0f} kB zipped, {raw / 1e3:.0f} kB raw")


if __name__ == "__main__":
    main()
