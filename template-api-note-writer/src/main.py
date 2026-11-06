import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List

from cnapi.get_api_eligible_posts import get_posts_eligible_for_notes
from cnapi.submit_note import submit_note
from data_models import NoteResult, PostWithContext
import dotenv
from note_writer.write_note import research_post_and_write_note

GUIDE_URL = "https://communitynotes.x.com/guide/en/api/overview"


def xurl_error_details(exc: Exception) -> str:
    """xurl prints the X API's error response to stdout; surface it instead of a bare exit code."""
    if isinstance(exc, subprocess.CalledProcessError):
        return ((exc.stdout or "") + (exc.stderr or "")).strip() or str(exc)
    return str(exc)


def check_xurl_installed() -> None:
    if shutil.which("xurl") is None:
        sys.exit(
            "xurl is not installed, or its folder is not on your PATH. "
            f"See step A.1 of the AI Note Writer guide: {GUIDE_URL}"
        )


def _check_setup() -> None:
    """Fail early, with a clear message, on the most common setup problems."""
    if not os.getenv("XAI_API_KEY"):
        sys.exit(
            "XAI_API_KEY is not set. Copy .env.example to .env and fill in your xAI API key "
            f"(step A.2 of the AI Note Writer guide: {GUIDE_URL})."
        )
    check_xurl_installed()


def _worker(
    post_with_context: PostWithContext,
    dry_run: bool = False,
    test_mode: bool = True,
):
    """
    Fetch and try to write and submit a note for one post.
    If `dry_run` is True, do not submit notes to the API, just print them to the console.
    """
    note_result: NoteResult = research_post_and_write_note(post_with_context)

    log_strings: List[str] = ["-" * 20, f"Post: {post_with_context.post.post_id}", "-" * 20]
    if note_result.context_description is not None:
        log_strings.append(
            f"\n*POST TEXT, ANY QUOTED OR REPLIED-TO POST, AND NOTE REQUEST SUGGESTIONS:*\n{note_result.context_description}\n"
        )
    if note_result.error is not None:
        log_strings.append(f"\n*ERROR:* {note_result.error}")
    if note_result.refusal:
        log_strings.append(f"\n*REFUSAL:* {note_result.refusal}")
    if note_result.note:
        log_strings.append(f"\n*NOTE:*\n  {note_result.note.note_text}\n")
        log_strings.append(
            f"\n*MISLEADING TAGS:*\n  {[tag.value for tag in note_result.note.misleading_tags]}\n"
        )

    if note_result.note is not None and not dry_run:
        try:
            submit_note(
                note=note_result.note,
                test_mode=test_mode,
                verbose_if_failed=False,
            )
            log_strings.append("\n*SUCCESSFULLY SUBMITTED NOTE*\n")
        except Exception as e:
            log_strings.append(
                "\n*ERROR SUBMITTING NOTE* (if it says a note already exists, we already wrote a note "
                f"on this post; moving on):\n{xurl_error_details(e)}\n"
            )
    print("".join(log_strings) + "\n")


def main(
    num_posts: int = 10,
    dry_run: bool = False,
    concurrency: int = 1,
    test_mode: bool = True,
):
    """
    Get up to `num_posts` recent posts eligible for notes and write notes for them.
    If `dry_run` is True, do not submit notes to the API, just print them to the console.
    If `test_mode` is True, fetch test-mode posts and submit notes in test mode.
    """

    print(f"Getting up to {num_posts} recent posts eligible for notes (test_mode={test_mode})")
    try:
        eligible_posts: List[PostWithContext] = get_posts_eligible_for_notes(
            max_results=num_posts, test_mode=test_mode
        )
    except subprocess.CalledProcessError as e:
        sys.exit(
            "Could not fetch posts eligible for notes. The X API said:\n"
            f"{xurl_error_details(e)}\n\n"
            "Check that xurl is logged in (`xurl auth status`), that your account is enrolled as an "
            "AI Note Writer, and that your app has Read and write permissions "
            f"(steps 2.1, 2.2 and A.1 of the AI Note Writer guide: {GUIDE_URL})."
        )
    print(f"Found {len(eligible_posts)} recent posts eligible for notes")
    print(
        f"  Eligible Post IDs: {', '.join([str(post_with_context.post.post_id) for post_with_context in eligible_posts])}\n"
    )
    if len(eligible_posts) == 0:
        print("No posts to process.")
        return

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(_worker, post, dry_run, test_mode) for post in eligible_posts
            ]
            for future in futures:
                future.result()
    else:
        for post in eligible_posts:
            _worker(post, dry_run, test_mode)
    print("Done.")


if __name__ == "__main__":
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="Run note‑writing bot once.")
    parser.add_argument(
        "--num-posts", type=int, default=10, help="Number of posts to process"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not submit notes to the API, just print them to the console",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent tasks to run",
    )
    args = parser.parse_args()
    _check_setup()
    main(
        num_posts=args.num_posts,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
        test_mode=os.getenv("TEST_MODE", "true").strip().lower() != "false",
    )
