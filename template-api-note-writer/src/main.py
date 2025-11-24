"""
To run the note writing bot, complete env_sample and rename it to ".env".
Then run the bot with:
  $ uv run src/main.py
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from typing import List

from requests_oauthlib import OAuth1Session  # type: ignore
from cnapi.get_api_eligible_posts import get_posts_eligible_for_notes
from cnapi.submit_note import submit_note
from cnapi.evaluate_note import evaluate_note
from cnapi.get_notes_written import get_notes_written
from data_models import EnvironmentVariables, NoteResult, PostWithContext
import dotenv
from note_writer.write_note import research_post_and_write_note, check_for_unsupported_media_in_post_with_context


def _parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the note-writing bot.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - num_posts (int): Number of posts to process (default: 10)
            - dry_run (bool): Whether to skip API submission (default: False)
            - concurrency (int): Number of concurrent tasks (default: 1)
            - co_submission_threshold (float): Minimum CO score to submit (default: 0)
            - co_reject_threshold (float): Minimum CO score to continue trying (default: -2)
            - max_attempts (int): Maximum number of note writing attempts (default: 3)
    """
    parser = argparse.ArgumentParser(description="Run note‑writing bot once.")
    _ = parser.add_argument(
        "--num-posts", type=int, default=10, help="Number of posts to process"
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not submit notes to the API, just print them to the console",
    )
    _ = parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent tasks to run",
    )
    _ = parser.add_argument(
        "--co-submission-threshold",
        type=float,
        default=0.0,
        help="Minimum CO score required to submit a note (default: 0)",
    )
    _ = parser.add_argument(
        "--co-reject-threshold",
        type=float,
        default=-2.0,
        help="Minimum CO score required to continue attempting to write a note (default: -2)",
    )
    _ = parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum number of attempts at writing a note (default: 3)",
    )
    args = parser.parse_args()

    return args


def _load_environment_variables() -> EnvironmentVariables:
    """
    Load and validate environment variables.
    
    Returns:
        EnvironmentVariables: An object containing all environment variables
        
    Raises:
        ValueError: If any required environment variables are missing
    """
    dotenv.load_dotenv()
    api_key = os.getenv("X_API_KEY")
    api_secret_key = os.getenv("X_API_KEY_SECRET")
    access_token = os.getenv("X_ACCESS_TOKEN")
    access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")
    xai_api_key = os.getenv("XAI_API_KEY")
    
    # Validate that all credentials are present
    missing_vars = []
    if not api_key:
        missing_vars.append("X_API_KEY")
    if not api_secret_key:
        missing_vars.append("X_API_KEY_SECRET")
    if not access_token:
        missing_vars.append("X_ACCESS_TOKEN")
    if not access_token_secret:
        missing_vars.append("X_ACCESS_TOKEN_SECRET")
    if not xai_api_key:
        missing_vars.append("XAI_API_KEY")
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please set them in your .env file or environment."
        )
    
    return EnvironmentVariables(
        api_key=api_key,
        api_secret_key=api_secret_key,
        access_token=access_token,
        access_token_secret=access_token_secret,
        xai_api_key=xai_api_key,
    )


def _create_oauth_session(env_vars: EnvironmentVariables) -> OAuth1Session:
    """
    Creates an OAuth 1.0a session for authenticating with the X API.
    
    Args:
        env_vars: Environment variables object containing OAuth credentials
    
    Returns:
        OAuth1Session: An authenticated OAuth 1.0a session object
    """
    return OAuth1Session(
        env_vars.api_key,
        client_secret=env_vars.api_secret_key,
        resource_owner_key=env_vars.access_token,
        resource_owner_secret=env_vars.access_token_secret,
    )


def _worker(
    oauth: OAuth1Session,
    post_with_context: PostWithContext,
    xai_api_key: str,
    dry_run: bool = False,
    co_submission_threshold: float = 0.0,
    co_reject_threshold: float = -2.0,
    max_attempts: int = 3,
):
    """
    Process a single post by researching and writing a note, then optionally submitting it.
    
    Args:
        oauth: OAuth1Session for authenticating with the X API
        post_with_context: Post with additional context (replies, quotes, etc.)
        xai_api_key: xAI API key for Grok models
        dry_run: If True, skip API submission and only print notes to console (default: False)
        co_submission_threshold: Minimum CO score required to submit a note (default: 0)
        co_reject_threshold: Minimum CO score required to continue attempting (default: -2)
        max_attempts: Maximum number of attempts at writing a note (default: 3)
    
    Notes:
        This function prints detailed information about the processing result, including
        post text, any errors, refusals, the generated note, and submission status.
        It will retry writing notes up to max_attempts times based on CO score thresholds.
    """
    # Initialize log with post and context
    log_strings: List[str] = ["-" * 20 + f"Post: {post_with_context.post.post_id}" + "-" * 20]
    if check_for_unsupported_media_in_post_with_context(post_with_context):
        log_strings.append(f"\n*UNSUPPORTED MEDIA TYPE FOUND IN POST OR REFERENCED POST*")
        return
    assert post_with_context.post.text is not None
    log_strings.append(f"\n*POST TEXT:* \n  {post_with_context.post.text}")
    if post_with_context.quoted_post is not None and post_with_context.quoted_post.text is not None:
        log_strings.append(f"\n*QUOTED POST TEXT:* \n  {post_with_context.quoted_post.text}")
    if post_with_context.in_reply_to_post is not None and post_with_context.in_reply_to_post.text is not None:
        log_strings.append(f"\n*IN-REPLY-TO POST TEXT:* \n  {post_with_context.in_reply_to_post.text}")

    note_result: NoteResult | None = None
    co_score: float | None = None
    
    for attempt in range(1, max_attempts + 1):
        log_strings.append(f"\n*ATTEMPT {attempt}/{max_attempts}*")
        
        note_result = research_post_and_write_note(post_with_context, xai_api_key, log_strings, verbose=(attempt == 1))

        if note_result.note:
            log_strings.append(f"\n*NOTE:*\n  {note_result.note.note_text}")
            log_strings.append(
                f"\n*MISLEADING TAGS:*\n  {[tag.value for tag in note_result.note.misleading_tags]}"
            )
        
        co_score = None
        if note_result.note is not None:
            co_score = evaluate_note(
                oauth=oauth,
                note_text=note_result.note.note_text,
                post_id=note_result.note.post_id,
            )
            log_strings.append(f"\n*CLAIM OPINION SCORE:* \n  {co_score}")
            
            # Check if score meets submission threshold
            if co_score >= co_submission_threshold:
                log_strings.append(f"\n*CO score {co_score} >= submission threshold {co_submission_threshold}, proceeding to submit*")
                break
            # Check if score is below reject threshold
            elif co_score < co_reject_threshold:
                log_strings.append(f"\n*CO score {co_score} < reject threshold {co_reject_threshold}, stopping attempts*")
                break
            # Score is between thresholds, try again if attempts remain
            else:
                if attempt < max_attempts:
                    log_strings.append(f"\n*CO score {co_score} between thresholds [{co_reject_threshold}, {co_submission_threshold}), retrying*")
                else:
                    log_strings.append(f"\n*CO score {co_score} between thresholds but max attempts reached*")
        else:
            log_strings.append(f"\n*No note generated on attempt {attempt}*")
            if attempt < max_attempts:
                log_strings.append("\n*Retrying...*")

    if (
        (note_result is not None)
        and (note_result.note is not None)
        and (co_score is not None)
        and (co_score >= co_submission_threshold)
        and (not dry_run)
    ):
        try:
            submit_note(
                oauth=oauth,
                note=note_result.note,
                test_mode=True,
                log_strings=log_strings,
            )
            log_strings.append("\n*SUCCESSFULLY SUBMITTED NOTE*")
        except:
            # Logging occurs in submit_note
            pass
    print("\n".join(log_strings) + "\n")


def main(
    oauth: OAuth1Session,
    xai_api_key: str,
    num_posts: int = 10,
    dry_run: bool = False,
    concurrency: int = 1,
    co_submission_threshold: float = 0.0,
    co_reject_threshold: float = -2.0,
    max_attempts: int = 3,
):
    """
    Main function to fetch eligible posts and write Community Notes for them.
    
    Args:
        oauth: OAuth1Session for authenticating with the X API
        xai_api_key: xAI API key for Grok models
        num_posts: Maximum number of posts to process (default: 10)
        dry_run: If True, skip API submission and only print notes (default: False)
        concurrency: Number of posts to process concurrently (default: 1)
        co_submission_threshold: Minimum CO score required to submit a note (default: 0)
        co_reject_threshold: Minimum CO score required to continue attempting (default: -2)
        max_attempts: Maximum number of attempts at writing a note (default: 3)
    
    Notes:
        Posts are processed either sequentially (concurrency=1) or concurrently
        using a ThreadPoolExecutor. The function prints progress information
        including eligible post IDs and processing results.
    """
    # Log recently submitted notes
    print("=" * 60)
    print("RECENTLY SUBMITTED NOTES SUMMARY")
    print("=" * 60)
    try:
        recent_notes = get_notes_written(oauth=oauth, test_mode=True, max_results=None)
        if recent_notes:
            print(f"\nFound {len(recent_notes)} recent note(s):\n")
            print("     NOTE_ID              POST_ID                TEST_RESULT")
            #print("1    1993006473949946282  1992676776057917735    (HarassmentAbuse: High) (UrlValidity: High) (ClaimOpinion: High)")                
            for i, note in enumerate(recent_notes, 1):
                test_result_str = ""
                if note.test_result:
                    test_result_str = ", ".join([f"{result.evaluator_type}: {result.evaluator_score_bucket}" for result in note.test_result])
                print(f"{i:<3}  {note.note_id}  {note.post_id}    {{{test_result_str}}}")
        else:
            print("\nNo notes found.\n")
    except Exception as e:
        print(f"\nError fetching notes: {e}\n")
    print("=" * 60)
    print()

    print("Getting posts eligible for notes")
    eligible_posts: List[PostWithContext] = get_posts_eligible_for_notes(
        oauth=oauth,
    )
    print(f"Found {len(eligible_posts)} recent posts eligible for notes")
    eligible_posts = eligible_posts[100:]
    eligible_posts = eligible_posts[:num_posts]
    print(f"Pruned to {len(eligible_posts)} posts")
    print(
        f"  Eligible Post IDs: \n{'\n'.join([str(post_with_context.post.post_id) for post_with_context in eligible_posts])}\n"
    )
    if len(eligible_posts) == 0:
        print("No posts to process.")
        return

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(_worker, oauth, post, xai_api_key, dry_run, co_submission_threshold, co_reject_threshold, max_attempts) for post in eligible_posts
            ]
            for future in futures:
                future.result()
    else:
        for post in eligible_posts:
            _worker(oauth, post, xai_api_key, dry_run, co_submission_threshold, co_reject_threshold, max_attempts)
    print("Done.")


if __name__ == "__main__":
    # Load environment variables and parse arguments
    args = _parse_args()
    env_vars = _load_environment_variables()
    
    # Create OAuth session
    oauth = _create_oauth_session(env_vars)

    # Run the main function
    main(
        oauth=oauth,
        xai_api_key=env_vars.xai_api_key,
        num_posts=args.num_posts,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
        co_submission_threshold=args.co_submission_threshold,
        co_reject_threshold=args.co_reject_threshold,
        max_attempts=args.max_attempts,
    )
