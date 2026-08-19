from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).resolve().parents[1] / "config" / "smoke.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))["dataset"]
    target = ROOT / config["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".partial")
        urllib.request.urlretrieve(config["url"], temporary)
        temporary.replace(target)
    actual_bytes = target.stat().st_size
    actual_sha256 = sha256_file(target)
    if actual_bytes != int(config["bytes"]):
        raise RuntimeError(
            f"byte-count mismatch: expected {config['bytes']}, got {actual_bytes}"
        )
    if actual_sha256.lower() != str(config["sha256"]).lower():
        raise RuntimeError(
            f"SHA-256 mismatch: expected {config['sha256']}, got {actual_sha256}"
        )
    print(f"verified {target}")
    print(f"bytes={actual_bytes}")
    print(f"sha256={actual_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

