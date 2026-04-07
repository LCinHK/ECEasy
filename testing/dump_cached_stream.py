"""Inspect cached backend stream payloads stored in shelve (KV_NAME).

Usage examples:
  python testing/dump_cached_stream.py --list
  python testing/dump_cached_stream.py --key <search_uuid>
  python testing/dump_cached_stream.py --key <search_uuid> --out raw_stream.txt
"""

from __future__ import annotations

import argparse
import shelve
from pathlib import Path


DEFAULT_KV = "eceasy-chat-local.kv"
SCRIPT_DIR = Path(__file__).resolve().parent
SIDECAR_EXTENSIONS = (".dat", ".db", ".dir", ".bak")


def resolve_kv_path(raw_path: str) -> Path:
    path = Path(raw_path)

    # Accept either the shelve prefix itself (e.g. eceasy-chat-local.kv)
    # or any of the physical sidecar files. On Windows, `shelve` may store
    # data in .dat/.dir/.bak files without a standalone base file.
    prefixes = [path, SCRIPT_DIR / path]

    def sidecar_candidates(prefix: Path):
        yield prefix
        for ext in SIDECAR_EXTENSIONS:
            yield Path(f"{prefix}{ext}")

    for prefix in prefixes:
        if any(candidate.exists() for candidate in sidecar_candidates(prefix)):
            return prefix

    return path


def has_shelve_store(prefix: Path) -> bool:
    if prefix.exists():
        return True
    return any(Path(f"{prefix}{ext}").exists() for ext in SIDECAR_EXTENSIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump cached ECEasy stream payload by search_uuid")
    parser.add_argument("--kv", default=DEFAULT_KV, help="Path to shelve KV file (default: eceasy-chat-local.kv)")
    parser.add_argument("--list", action="store_true", help="List available keys and exit")
    parser.add_argument("--key", help="search_uuid key to dump")
    parser.add_argument("--out", help="Optional output file path for raw stream text")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kv_path = resolve_kv_path(args.kv)

    if not has_shelve_store(kv_path):
        print(f"KV file not found: {args.kv}")
        return 1

    with shelve.open(str(kv_path)) as db:
        keys = list(db.keys())
        if args.list:
            print(f"Total keys: {len(keys)}")
            for k in keys:
                print(k)
            return 0

        if not args.key:
            print("Please provide --key <search_uuid>, or use --list")
            return 1

        if args.key not in db:
            print(f"Key not found: {args.key}")
            return 1

        raw = "".join(db[args.key])
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(raw, encoding="utf-8")
            print(f"Wrote raw stream to: {out_path}")
        else:
            print(raw)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())






