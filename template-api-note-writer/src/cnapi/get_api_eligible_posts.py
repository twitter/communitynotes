from datetime import datetime
from typing import Dict, List

from .xurl_util import run_xurl

from data_models import Media, Post


def _fetch_posts_eligible_for_notes(
    max_results: int = 2, test_mode: bool = True
) -> dict:
    """
    Fetch posts eligible for notes by calling the Community Notes API.
    For more details, see: https://docs.x.com/x-api/community-notes/introduction
    Args:
        max_results: Maximum number of results to return (default is 2).
        test_mode: If True, use test mode for the API (default is True).
    Returns:
        A dictionary containing the API response.
    """
    path = (
        "/2/notes/search/posts_eligible_for_notes"
        f"?test_mode={'true' if test_mode else 'false'}"
        f"&max_results={max_results}"
        "&tweet.fields=author_id,created_at,referenced_tweets,media_metadata,note_tweet"
        "&expansions=attachments.media_keys,referenced_tweets.id,referenced_tweets.id.attachments.media_keys"
        "&media.fields=alt_text,duration_ms,height,media_key,preview_image_url,public_metrics,type,url,width,variants"
    )
    cmd = [
        "xurl",
        path,
    ]
    return run_xurl(cmd)


def _parse_posts_eligible_response(resp: Dict) -> List[Post]:
    """
    Convert the raw JSON dict returned by `fetch_posts_eligible_for_notes`
    into a list of `Post` objects with their associated `Media`.

    For more details, see: https://docs.x.com/x-api/community-notes/introduction
    Args:
        resp: The raw response from the API, as a dictionary.
    Returns:
        A list of `Post` objects, each containing associated `Media` objects if available.
    """
    includes_media = resp.get("includes", {}).get("media", [])
    media_by_key = {m["media_key"]: m for m in includes_media}

    # rename type field to media_type to avoid name conflict with type
    for media_obj in media_by_key.values():
        media_obj["media_type"] = media_obj.pop("type")

    posts: List[Post] = []
    for item in resp.get("data", []):
        media_objs: List[Media] = []
        media_keys = item.get("attachments", {}).get("media_keys", [])

        for key in media_keys:
            if key in media_by_key:
                media_obj = Media(**media_by_key[key])
                media_objs.append(media_obj)

        text = item["text"]
        note_tweet_text = item.get("note_tweet", {}).get("text", "")
        if note_tweet_text:
            text = note_tweet_text

        post = Post(
            post_id=item["id"],
            author_id=item["author_id"],
            created_at=datetime.fromisoformat(
                item["created_at"].replace("Z", "+00:00")
            ),
            text=text,
            media=media_objs,
        )
        posts.append(post)

    return posts


def get_posts_eligible_for_notes(
    max_results: int = 2, test_mode: bool = True
) -> List[Post]:
    """
    Get posts eligible for notes by calling the Community Notes API.
    For more details, see: https://docs.x.com/x-api/community-notes/introduction

    Returns:
        A list of `Post` objects.
    """
    return _parse_posts_eligible_response(
        _fetch_posts_eligible_for_notes(max_results, test_mode)
    )


if __name__ == "__main__":
    eligible_posts = get_posts_eligible_for_notes(max_results=10)
    print(eligible_posts)
