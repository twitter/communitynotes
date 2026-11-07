"""
List this writer's most recent test-mode notes and the evaluator scores for each.

uv run src/check_scores.py
uv run src/check_scores.py --max-notes 200
uv run src/check_scores.py --json > scores.json

Pages through GET /2/notes/search/notes_written (test_mode=true), newest first, until it has
`--max-notes` notes (default 50, the window admission is based on), prints one block per note,
then a count of score buckets per evaluator.
"""

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from typing import Dict, List

from cnapi.xurl_util import run_xurl
import dotenv
from main import GUIDE_URL, check_xurl_installed, xurl_error_details


def get_recent_test_notes_written(max_notes: int = 50) -> List[Dict]:
    notes: List[Dict] = []
    pagination_token = None
    while len(notes) < max_notes:
        path = (
            "/2/notes/search/notes_written"
            "?test_mode=true"
            f"&max_results={min(100, max_notes - len(notes))}"
            "&note.fields=id,info,status,test_result"
        )
        if pagination_token:
            path += f"&pagination_token={pagination_token}"
        resp = run_xurl(["xurl", path])
        notes.extend(resp.get("data", []))
        pagination_token = (resp.get("meta") or {}).get("next_token")
        if not pagination_token:
            break
    return notes[:max_notes]


def _evaluation_outcomes(note: Dict) -> List[Dict]:
    return (note.get("test_result") or {}).get("evaluation_outcome") or []


def print_report(notes: List[Dict]) -> None:
    print(f"Showing your {len(notes)} most recent test-mode notes\n")
    buckets: Dict[str, Counter] = defaultdict(Counter)
    for note in notes:
        info = note.get("info") or {}
        print(f"Note {note.get('id')}  post {info.get('post_id')}  status {note.get('status')}")
        print(f"  {info.get('text', '')}")
        outcomes = _evaluation_outcomes(note)
        if not outcomes:
            print("  (no evaluator results yet)")
        for outcome in outcomes:
            print(f"  {outcome['evaluator_type']}: {outcome['evaluator_score_bucket']}")
            buckets[outcome["evaluator_type"]][outcome["evaluator_score_bucket"]] += 1
        print()

    if buckets:
        print("Summary")
        for evaluator, counts in sorted(buckets.items()):
            print(f"  {evaluator}: " + ", ".join(f"{b}={n}" for b, n in counts.most_common()))
    pending = sum(1 for note in notes if not _evaluation_outcomes(note))
    if pending:
        print(f"  {pending} notes have no evaluator results yet")


if __name__ == "__main__":
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="Show evaluator scores for test-mode notes.")
    parser.add_argument("--json", action="store_true", help="Print the raw notes as JSON")
    parser.add_argument(
        "--max-notes", type=int, default=50, help="Number of most recent notes to show (default 50)"
    )
    args = parser.parse_args()
    check_xurl_installed()
    try:
        notes = get_recent_test_notes_written(args.max_notes)
    except subprocess.CalledProcessError as e:
        sys.exit(
            "Could not fetch your notes. The X API said:\n"
            f"{xurl_error_details(e)}\n\n"
            "Check that xurl is logged in (`xurl auth status`) and that your account is enrolled as an "
            f"AI Note Writer (steps 2.2 and A.1 of the AI Note Writer guide: {GUIDE_URL})."
        )
    if args.json:
        print(json.dumps(notes, indent=2))
    else:
        print_report(notes)
