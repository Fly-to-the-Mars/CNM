"""Verify the release inventory and preserved runtime using only the standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path, manifest: dict) -> list[str]:
    errors = []
    for entry in manifest["files"]:
        relative = entry["path"]
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            errors.append(f"Path escapes release: {relative}")
            continue
        if not path.is_file():
            errors.append(f"Missing file: {relative}")
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            errors.append(f"SHA-256 mismatch: {relative}")
        if "bytes" in entry and len(data) != entry["bytes"]:
            errors.append(f"Size mismatch: {relative}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    release = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    runtime = json.loads((root / "docs/RUNTIME_BASELINE.json").read_text(encoding="utf-8"))
    errors = verify(root, release) + verify(root, runtime)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Release inventory: {len(release['files'])} files verified")
    print(f"Preserved runtime: {len(runtime['files'])} files verified")


if __name__ == "__main__":
    main()
