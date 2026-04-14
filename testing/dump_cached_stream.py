"""Inspect cached backend stream payloads stored in shelve (KV_NAME).

Usage examples:
  python testing/dump_cached_stream.py --list
  python testing/dump_cached_stream.py --key <search_uuid>
  python testing/dump_cached_stream.py --key <search_uuid> --out raw_stream.txt
  python testing/dump_cached_stream.py --search <word_or_phrase>
  python testing/dump_cached_stream.py --search <word_or_phrase> --case-sensitive
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
    parser.add_argument("--search", help="Search for a word or phrase in all cached values")
    parser.add_argument("--case-sensitive", action="store_true", help="Make search case-sensitive (default: case-insensitive)")
    parser.add_argument("--out", help="Optional output file path for raw stream text")
    return parser.parse_args()


def search_kv(db: dict, search_term: str, case_sensitive: bool = False) -> list[tuple[str, int]]:
    """Search for a term in all values in the shelve database.
    
    Args:
        db: Shelve database object
        search_term: Term to search for
        case_sensitive: Whether to use case-sensitive search
        
    Returns:
        List of tuples (key, match_count) for keys containing the search term
    """
    results = []
    search_term_normalized = search_term if case_sensitive else search_term.lower()
    
    for key in db.keys():
        try:
            raw_value = "".join(db[key])
            value_normalized = raw_value if case_sensitive else raw_value.lower()
            match_count = value_normalized.count(search_term_normalized)
            if match_count > 0:
                results.append((key, match_count))
        except Exception as e:
            print(f"  [Warning] Error reading key {key}: {e}")
    
    return results


def main() -> int:
    args = parse_args()
    kv_path = resolve_kv_path(args.kv)

    if not has_shelve_store(kv_path):
        print(f"KV file not found: {args.kv}")
        return 1

    with shelve.open(str(kv_path)) as db:
        keys = list(db.keys())
        
        # List mode
        if args.list:
            print(f"Total keys: {len(keys)}")
            for k in keys:
                print(k)
            return 0

        # Search mode
        if args.search:
            print(f"Searching for: '{args.search}' (case-sensitive: {args.case_sensitive})")
            results = search_kv(db, args.search, args.case_sensitive)
            if not results:
                print("No matches found.")
                return 0
            print(f"Found {len(results)} matching key(s):\n")
            for key, count in sorted(results, key=lambda x: x[1], reverse=True):
                print(f"  {key} ({count} match{'es' if count > 1 else ''})")
            return 0

        # Dump specific key
        if args.key:
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

        # No action specified
        print("Please provide one of: --list, --search <term>, or --key <search_uuid>")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())






