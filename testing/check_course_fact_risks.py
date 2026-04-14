from pathlib import Path
import sys
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import faiss_rag

CODE_RE = re.compile(r"\[([A-Za-z]{4}\d{4}[A-Za-z]?)]")


def extract_code(name: str) -> str:
    m = CODE_RE.search(name or "")
    return m.group(1).upper() if m else ""


def main() -> int:
    queries = [
        "i want to work on robotic and ai in future, can you recommend a list of elective courses for me",
        "Which course is ELEC3130?",
        "Is ELEC2200 still offered?",
    ]

    print("Official course codes detected from concrete files:", len(faiss_rag.OFFICIAL_COURSE_CODES))
    for q in queries:
        print("\n" + "-" * 80)
        print("Query:", q)
        ctx = faiss_rag.get_rag_context(q, k=12)
        print("Returned contexts:", len(ctx))

        risky = []
        for i, c in enumerate(ctx, start=1):
            code = extract_code(c.get("name", ""))
            src = c.get("source_relpath", "")
            if code and faiss_rag.OFFICIAL_COURSE_CODES and code not in faiss_rag.OFFICIAL_COURSE_CODES:
                risky.append((i, code, src, c.get("name", "")))

        if risky:
            print("Potentially risky contexts (code not in official file list):")
            for i, code, src, name in risky:
                print(f"  {i}. {code} | {name}")
                print(f"     source: {src}")
        else:
            print("No risky contexts flagged by official-code guardrail.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


