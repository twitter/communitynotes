from data_models import NoteResult, Post, ProposedMisleadingNote
from note_writer.llm_util import (
    get_grok_live_search_response,
    get_grok_response,
    grok_describe_image,
)
from note_writer.misleading_tags import get_misleading_tags


def _get_prompt_for_note_writing(post: Post, images_summary: str, search_results: str):
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

    ```
    Post text:
    ```
    {post.text}
    ```

    Summary of images in the post:
    ```
    {images_summary}
    ```

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


def _get_prompt_for_live_search(post: Post, images_summary: str = ""):
    return f"""Below will be a post on X. Do research on the post to determine if the post is potentially misleading. \
Your response MUST include URLs/links directly in the text, next to the claim it supports. Don't include any sort \
of wasted characters e.g. [Source] when listing a URL. Just state the URL directly in the text. \

    Post text:
    ```
    {post.text}
    ```

    Summary of images in the post:
    ```
    {images_summary}
    ```
    """


def _summarize_images(post: Post) -> str:
    """
    Summarize images, if they exist. Abort if video or other unsupported media type.
    """
    images_summary = ""
    for i, media in enumerate(post.media):
        if media.media_type == "photo":
            image_description = grok_describe_image(media.url)
            images_summary += f"Image {i}: {image_description}\n\n"
        elif media.media_type == "video":
            raise ValueError("Video not supported yet")
        else:
            raise ValueError(f"Unsupported media type: {media.media_type}")
    return images_summary


def research_post_and_write_note(
    post: Post,
) -> NoteResult:
    try:
        images_summary = _summarize_images(post)
    except ValueError as e:
        return NoteResult(post=post, error=str(e))

    search_prompt = _get_prompt_for_live_search(post, images_summary)
    search_results = get_grok_live_search_response(search_prompt)
    note_prompt = _get_prompt_for_note_writing(post, images_summary, search_results)

    note_or_refusal_str = get_grok_response(note_prompt)

    if ("NO NOTE NEEDED" in note_or_refusal_str) or (
        "NOT ENOUGH EVIDENCE TO WRITE A GOOD COMMUNITY NOTE" in note_or_refusal_str
    ):
        return NoteResult(post=post, refusal=note_or_refusal_str)

    misleading_tags = get_misleading_tags(post, images_summary, note_or_refusal_str)

    return NoteResult(
        post=post,
        note=ProposedMisleadingNote(
            post_id=post.post_id,
            note_text=note_or_refusal_str,
            misleading_tags=misleading_tags,
        ),
    )
