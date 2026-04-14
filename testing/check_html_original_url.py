from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ingest_FAISS

FILES = [
    Path("ECEknowledge/academic_policies/Academic Regulations Governing UG Studies, 2025-26 _ HKUST - Academic Registry.html"),
    Path("ECEknowledge/academic_policies/Assessment & Progress _ HKUST - Academic Registry_files/Assessment & Progress _ HKUST - Academic Registry.html"),
    Path("ECEknowledge/course_syllabus/common core courese/HKUST - Common Core Program.html"),
    Path("ECEknowledge/student_guides/undergraduate_student_guide/Courses & Enrollment _ HKUST - Academic Registry/Courses & Enrollment _ HKUST - Academic Registry.html"),
]


def main() -> int:
    print("Checking original_url extraction from saved HTML files...\n")
    for path in FILES:
        print(f"FILE: {path}")
        if not path.exists():
            print("  status: MISSING FILE\n")
            continue

        docs = ingest_FAISS._load_html_as_document(path)
        metadata = docs[0].metadata if docs else {}
        original_url = str(metadata.get("original_url", "")).strip()
        title = str(metadata.get("title", "")).strip()

        print(f"  original_url: {original_url if original_url else '<not found>'}")
        print(f"  title: {title[:100] if title else '<none>'}\n")

    base = Path("ECEknowledge")
    all_html = sorted(list(base.rglob("*.html")) + list(base.rglob("*.htm")))
    ok = 0
    missing: list[str] = []
    for path in all_html:
        docs = ingest_FAISS._load_html_as_document(path)
        metadata = docs[0].metadata if docs else {}
        if str(metadata.get("original_url", "")).strip():
            ok += 1
        else:
            missing.append(str(path.relative_to(base)))

    print("Coverage summary:")
    print(f"  Total HTML files: {len(all_html)}")
    print(f"  With original_url: {ok}")
    print(f"  Without original_url: {len(missing)}")
    if missing:
        print("  First 5 missing:")
        for rel in missing[:5]:
            print(f"    - {rel}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



