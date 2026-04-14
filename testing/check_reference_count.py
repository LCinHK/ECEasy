from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eceasy_server.config import resolve_reference_count

TESTS = [
    "ELEC2400 prerequisites?",
    "What is ELEC1100",
    "Please compare ECE major roadmap, internship, exchange options, and graduation requirements with detailed step by step plan?",
]

for query in TESTS:
    print(f"{resolve_reference_count(query)} :: {query}")

