from data_models import NoteResult, Post, PostWithContext, ProposedMisleadingNote
from note_writer.llm_util import (
    get_grok_live_search_response,
    get_grok_response,
    grok_describe_image,
)
from note_writer.misleading_tags import get_misleading_tags


def _get_prompt_for_note_writing(post_with_context_description: str, search_results: str):
    return f"""Below will be a post on X, and live search results from the web. \
If the post is misleading and needs a community note, \
then your response should be the proposed community note itself. \
If the post is not misleading, or does not contain any concrete fact-checkable claims, or there \
is not strong enough evidence to write a 100%-supported-by-evidence \
community note, then do not write a note, and instead respond with "NO NOTE NEEDED." or \
"NOT ENOUGH EVIDENCE TO WRITE A GOOD COMMUNITY NOTE.".

If a note is justified, then please write \
a very good community note, which is concise (tweet-length at most), interesting and clear to read, \
contains no unnecessary words, is backed by very solid evidence from sources that would be most likely \
to be found trustworthy by people on both sides of the political spectrum (citing URL(s)), and is \
written in a way that would be most likely to be found trustworthy by people on both sides of the \
political spectrum. \

The answer MUST be short, like a post on X (280 characters maximum, not counting URLs). \
The note should not include any sort of wasted characters e.g. [Source] when listing a URL. Just state the URL directly in the text. \
Each note MUST include at least one URL/link source. Nothing else counts as a source other than a URL/link. \
Each note MUST NOT use any hashtags (#). Keep a professional tone with no hashtags, emojis, etc. \

If the post does not need a community note (either because the original post is not misleading, \
or does not contain any concrete fact-checkable claims, \
then your response should simply be "NO NOTE NEEDED.".

If the post may need a community note, but you weren't able to find enough concrete evidence to \
write an ironclad community note, then your response should be \
"NOT ENOUGH EVIDENCE TO WRITE A GOOD COMMUNITY NOTE.".

If you are writing a note, don't preface it with anything like "Community Note:". Just write the note.

    {post_with_context_description}

    Live search results:
    ```
    {search_results}
    ```

    If you aren't sure whether the post is misleading and warrants a note, then err on the side of \
not writing a note (instead, say "NO NOTE NEEDED" or "NOT ENOUGH EVIDENCE TO WRITE A GOOD COMMUNITY NOTE"). \
Only write a note if you are extremely confident that the post is misleading, \
that the evidence is strong, and that the note will be found helpful by the community. \
For example, if a post is just making a prediction about the future, don't write a note \
saying that the prediction is uncertain or likely to be wrong. \
    """


def _get_post_with_context_description_for_prompt(post_with_context: PostWithContext) -> str:
    description = f"""Post text:
```
{post_with_context.post.text}
```
"""
    images_summary = _summarize_images_for_post(post_with_context.post)
    if images_summary is not None and len(images_summary) > 0:
        description += f"""Summary of images in the post:
```
{images_summary}
```
"""
    if post_with_context.quoted_post:
        description += f"""The post of interest had quoted (referenced) another post. Here is the quoted post's text:
```
{post_with_context.quoted_post.text}
```
"""
        quoted_images_summary = _summarize_images_for_post(post_with_context.quoted_post)
        if quoted_images_summary is not None and len(quoted_images_summary) > 0:
            description += f"""Summary of images in the quoted post:
```
{quoted_images_summary}
```
"""
    
    if post_with_context.in_reply_to_post:
        description += f"""The post of interest was a reply to another post. Here is the replied-to post's text:
```
{post_with_context.in_reply_to_post.text}
```
"""
        replied_to_images_summary = _summarize_images_for_post(post_with_context.in_reply_to_post)
        if replied_to_images_summary is not None and len(replied_to_images_summary) > 0:
            description += f"""Summary of images in the replied-to post:
```
{replied_to_images_summary}
```
"""

    return description


def _get_prompt_for_live_search(post_with_context_description: str) -> str:
    return f"""Below will be a post on X. Do research on the post to determine if the post is potentially misleading. \
Your response MUST include URLs/links directly in the text, next to the claim it supports. Don't include any sort \
of wasted characters e.g. [Source] when listing a URL. Just state the URL directly in the text. \

    {post_with_context_description}
    """

def _summarize_images_for_post(post: Post) -> str:
    """
    Summarize images, if they exist. Abort if video or other unsupported media type.
    """
    images_summary = ""
    for i, media in enumerate(post.media):
        assert media.media_type == "photo" # remove assert when video support is added
        image_description = grok_describe_image(media.url)
        images_summary += f"Image {i}: {image_description}\n"
    return images_summary


def _check_for_unsupported_media(post: Post) -> bool:
    """Check if the post contains unsupported media types."""
    for media in post.media:
        if media.media_type not in ["photo"]:
            return True


def _check_for_unsupported_media_in_post_with_context(post_with_context: PostWithContext) -> None:
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
) -> NoteResult:
    if _check_for_unsupported_media_in_post_with_context(post_with_context):
        return NoteResult(post=post_with_context, error="Unsupported media type (e.g. video) found in post or in referenced post.")
    
    post_with_context_description = _get_post_with_context_description_for_prompt(post_with_context)

    search_prompt = _get_prompt_for_live_search(post_with_context_description)
    search_results = get_grok_live_search_response(search_prompt)

    note_prompt = _get_prompt_for_note_writing(post_with_context_description, search_results)
    note_or_refusal_str = get_grok_response(note_prompt)

    if ("NO NOTE NEEDED" in note_or_refusal_str) or (
        "NOT ENOUGH EVIDENCE TO WRITE A GOOD COMMUNITY NOTE" in note_or_refusal_str
    ):
        return NoteResult(post=post_with_context, refusal=note_or_refusal_str, context_description=post_with_context_description)

    misleading_tags = get_misleading_tags(post_with_context_description, note_or_refusal_str)

    return NoteResult(
        post=post_with_context,
        note=ProposedMisleadingNote(
            post_id=post_with_context.post.post_id,
            note_text=note_or_refusal_str,
            misleading_tags=misleading_tags,
        ),
        context_description=post_with_context_description,
    )
