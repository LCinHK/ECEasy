from __future__ import annotations

from pathlib import Path
import re
import sys

from docx2txt import process as docx_process

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import faiss_rag

SYLLABUS_DOC = PROJECT_ROOT / "ECEknowledge" / "course_syllabus" / "ELEC" / "ELEC_Syllabus_25-26_fall_spring.docx"
ELEC_SYLLABUS_DIR = PROJECT_ROOT / "ECEknowledge" / "course_syllabus" / "ELEC"


def extract_heading_title(text: str, code: str) -> str:
    patterns = [
        rf"#\s*{re.escape(code)}\s*-\s*([^\n(]+)",
        rf"{re.escape(code)}\s*##\s*Course\s*Title\s*([^#\n]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def official_course_codes() -> set[str]:
    codes = set()
    for p in ELEC_SYLLABUS_DIR.glob("*.pdf"):
        m = re.search(r"([A-Z]{4}\d{4}[A-Z]?)", p.stem.upper())
        if m:
            codes.add(m.group(1))
    return codes


def main() -> int:
    failures: list[str] = []
    syllabus_text = docx_process(str(SYLLABUS_DOC))

    # 1) Authoritative source facts.
    title_3130 = extract_heading_title(syllabus_text, "ELEC 3130")
    if title_3130 != "Digital Image Processing":
        failures.append(f"ELEC3130 title expected 'Digital Image Processing' but got '{title_3130 or '<missing>'}'")

    codes = official_course_codes()
    if "ELEC2200" in codes:
        failures.append("ELEC2200 unexpectedly present as a standalone syllabus PDF in course_syllabus/ELEC")

    # 2) Retrieval behavior for risky queries.
    q3130 = faiss_rag.get_rag_context("Which course is ELEC3130?", k=8)
    text3130 = "\n".join((c.get("name", "") + "\n" + c.get("snippet", "")) for c in q3130)
    if "Digital Image Processing" not in text3130:
        failures.append("Retrieval for ELEC3130 did not surface the authoritative title Digital Image Processing")
    if "AI in Robotics" in text3130:
        failures.append("Retrieval for ELEC3130 still surfaced the incorrect AI in Robotics title")

    q2200 = faiss_rag.get_rag_context("Is ELEC2200 still offered?", k=8)
    text2200 = "\n".join((c.get("name", "") + "\n" + c.get("snippet", "")) for c in q2200)
    if "verification note" not in text2200.lower() and "verify" not in text2200.lower():
        failures.append("Retrieval for ELEC2200 did not fall back to a verification note")

    print("Course fact check summary")
    print("- ELEC3130 official title:", title_3130 or "<missing>")
    print("- Standalone ELEC2200 syllabus PDF present:", "ELEC2200" in codes)
    print("- ELEC3130 retrieval contexts:", len(q3130))
    print("- ELEC2200 retrieval contexts:", len(q2200))

    if failures:
        print("\nFAILURES:")
        for item in failures:
            print("-", item)
        return 1

    print("\nAll course-fact checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

