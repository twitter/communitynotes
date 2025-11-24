from data_models import NoteResult, Post, PostWithContext, ProposedMisleadingNote
from note_writer.llm_util import LLMClient
from note_writer.misleading_tags import get_misleading_tags


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
"""

def _check_for_unsupported_media(post: Post) -> bool:
    """Check if the post contains unsupported media types."""
    for media in post.media:
        if media.media_type not in ["photo"]:
            return True
    return False


def _check_for_unsupported_media_in_post_with_context(post_with_context: PostWithContext) -> bool:
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
    enable_web_image_understanding: bool = True,
    enable_x_image_understanding: bool = True,
    enable_x_video_understanding: bool = False,
) -> NoteResult:
    """
    Research a post and write a Community Note if needed.
    
    Args:
        post_with_context: Post with additional context (replies, quotes, etc.)
        xai_api_key: xAI API key for Grok models
        
    Returns:
        NoteResult containing the note, refusal, or error
    """
    if _check_for_unsupported_media_in_post_with_context(post_with_context):
        return NoteResult(post=post_with_context, error="Unsupported media type (e.g. video) found in post or in referenced post.")
    
    # Create LLM client instance
    llm_client = LLMClient(
        api_key=xai_api_key,
        model="grok-4-fast",
        enable_web_image_understanding=enable_web_image_understanding,
        enable_x_image_understanding=enable_x_image_understanding,
        enable_x_video_understanding=enable_x_video_understanding,
    )    
    writing_prompt = _get_prompt_for_note_writing(post_with_context)
    note_or_refusal_str = llm_client.get_grok_response(writing_prompt)

    if ("NO NOTE NEEDED" in note_or_refusal_str) or (
        "NOT ENOUGH EVIDENCE TO WRITE A GOOD COMMUNITY NOTE" in note_or_refusal_str
    ):
        return NoteResult(post=post_with_context, refusal=note_or_refusal_str, writing_prompt=writing_prompt)

    misleading_tags = get_misleading_tags(post_with_context, note_or_refusal_str, llm_client)

    return NoteResult(
        post=post_with_context,
        note=ProposedMisleadingNote(
            post_id=post_with_context.post.post_id,
            note_text=note_or_refusal_str,
            misleading_tags=misleading_tags,
        ),
        writing_prompt=writing_prompt,
    )
