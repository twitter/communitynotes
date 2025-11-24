from data_models import NoteResult, Post, PostWithContext, ProposedMisleadingNote
from note_writer.llm_util import LLMClient
from note_writer.misleading_tags import get_misleading_tags
from utils.url_utils import extract_and_validate_urls


def _get_prompt_for_note_writing(post_with_context: PostWithContext):
    user_name = post_with_context.post.username
    post_id = post_with_context.post.post_id
    post_link = f"https://x.com/{user_name}/status/{post_id}"
    return f"""\
Please write a Community Note for this post
{post_link}

Follow these guidelines:
- Good community notes are often 1-2 sentences, and should never be longer than 4 sentences.
- Concise notes avoid unnecessary words, opinion, speculation, and any unusual formatting (e.g. no bolding/underlining/emojis).
- The note should display source URLs without any additional markup, brackets or labels.
- The full response must end after the proposed note and any associated source URLs (i.e. there should be no additional explanation or justification following the note).

If the post does not need a community note (either because the original post is not misleading, \
or does not contain any concrete fact-checkable claims, \
then your response should simply be "NO NOTE NEEDED.".

If the post may need a community note, but you weren't able to find enough concrete evidence to \
write an ironclad community note, then your response should be \
"NOT ENOUGH EVIDENCE TO WRITE A GOOD COMMUNITY NOTE.".
"""

def _check_for_unsupported_media(post: Post) -> bool:
    """Check if the post contains unsupported media types."""
    for media in post.media:
        if media.media_type not in ["photo"]:
            return True
    return False


def check_for_unsupported_media_in_post_with_context(post_with_context: PostWithContext) -> bool:
    """Check if the post or any referenced posts contain unsupported media types."""
    if _check_for_unsupported_media(post_with_context.post):
        return True
    if post_with_context.quoted_post and _check_for_unsupported_media(post_with_context.quoted_post):
        return True
    if post_with_context.in_reply_to_post and _check_for_unsupported_media(post_with_context.in_reply_to_post):
        return True
    return False


def research_post_and_write_note(
    post_with_context: PostWithContext,
    xai_api_key: str,
    log_strings: list[str],
    enable_web_image_understanding: bool = True,
    enable_x_image_understanding: bool = True,
    enable_x_video_understanding: bool = False,
    verbose: bool = True,
) -> NoteResult:
    """
    Research a post and write a Community Note if needed.
    
    Args:
        post_with_context: Post with additional context (replies, quotes, etc.)
        xai_api_key: xAI API key for Grok models
        
    Returns:
        NoteResult containing the note, refusal, or error
    """
    # Create LLM client instance
    llm_client = LLMClient(
        api_key=xai_api_key,
        model="grok-4-fast",
        enable_web_image_understanding=enable_web_image_understanding,
        enable_x_image_understanding=enable_x_image_understanding,
        enable_x_video_understanding=enable_x_video_understanding,
    )    
    writing_prompt = _get_prompt_for_note_writing(post_with_context)
    if verbose:
        log_strings.append(f"\n*WRITING PROMPT:*\n  {writing_prompt}")
    note_or_refusal_str, tool_calls, citations = llm_client.get_grok_response(writing_prompt)

    if ("NO NOTE NEEDED" in note_or_refusal_str) or (
        "NOT ENOUGH EVIDENCE TO WRITE A GOOD COMMUNITY NOTE" in note_or_refusal_str
    ):
        log_strings.append(f"\n*REFUSAL:*\n  {note_or_refusal_str}")
        return NoteResult(post=post_with_context, refusal=note_or_refusal_str, writing_prompt=writing_prompt)

    misleading_tags = get_misleading_tags(post_with_context, note_or_refusal_str, llm_client)

    failed_urls = extract_and_validate_urls(note_or_refusal_str, citations)
    if failed_urls:
        error_details = "\n".join([f"  {status_code:<4} {url}" for url, status_code in failed_urls])
        error_message = f"One or more URLs returned non-2xx/3xx status codes:\n{error_details}"
        log_strings.append(f"\n*ERROR (URL validation failed):*\n  {error_message}")
        log_strings.append(f"\n*CITATIONS:*\n  {'\n  '.join(citations)}")
        return NoteResult(post=post_with_context, error=error_message, citations=citations, tool_calls=tool_calls)

    return NoteResult(
        post=post_with_context,
        note=ProposedMisleadingNote(
            post_id=post_with_context.post.post_id,
            note_text=note_or_refusal_str,
            misleading_tags=misleading_tags,
        ),
        writing_prompt=writing_prompt,
        citations=citations,
        tool_calls=tool_calls,
    )
