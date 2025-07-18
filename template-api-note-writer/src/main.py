import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import List

from cnapi.get_api_eligible_posts import get_posts_eligible_for_notes
from cnapi.submit_note import submit_note
from data_models import NoteResult, Post, PostWithContext
import dotenv
from note_writer.write_note import research_post_and_write_note


def _worker(
    post_with_context: PostWithContext,
    dry_run: bool = False,
):
    """
    Fetch and try to write and submit a note for one post.
    If `dry_run` is True, do not submit notes to the API, just print them to the console.
    """
    note_result: NoteResult = research_post_and_write_note(post_with_context)

    log_strings: List[str] = ["-" * 20, f"Post: {post_with_context.post.post_id}", "-" * 20]
    if note_result.context_description is not None:
        log_strings.append(
            f"\n*POST TEXT, IMAGE SUMMARIES, AND ANY QUOTED OR REPLIED-TO POST:*\n{note_result.context_description}\n"
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
                test_mode=True,
                verbose_if_failed=False,
            )
            log_strings.append("\n*SUCCESSFULLY SUBMITTED NOTE*\n")
        except Exception:
            log_strings.append(
                "\n*ERROR SUBMITTING NOTE*: likely we already wrote a note on this post; moving on.\n"
            )
    print("".join(log_strings) + "\n")


def main(
    num_posts: int = 10,
    dry_run: bool = False,
    concurrency: int = 1,
):
    """
    Get up to `num_posts` recent posts eligible for notes and write notes for them.
    If `dry_run` is True, do not submit notes to the API, just print them to the console.
    """

    print(f"Getting up to {num_posts} recent posts eligible for notes")
    eligible_posts: List[PostWithContext] = get_posts_eligible_for_notes(max_results=num_posts)
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
                executor.submit(_worker, post, dry_run) for post in eligible_posts
            ]
            for future in futures:
                future.result()
    else:
        for post in eligible_posts:
            _worker(post, dry_run)
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
    main(
        num_posts=args.num_posts,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
    )
