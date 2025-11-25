from datetime import datetime
from typing import Optional

from requests_oauthlib import OAuth1Session  # type: ignore

from data_models import Media, Post, PostWithContext


def _fetch_posts_eligible_for_notes(
    oauth: OAuth1Session,
    max_results: int = 2,
    test_mode: bool = True,
    pagination_token: str | None = None,
) -> dict:
    """
    Fetch posts eligible for notes by calling the Community Notes API.
    For more details, see: https://docs.x.com/x-api/community-notes/introduction
    Args:
        oauth: OAuth1Session object for authenticating with the X API.
        max_results: Maximum number of results to return (default is 2).
        test_mode: If True, use test mode for the API (default is True).
        pagination_token: Token to get the next page of results (default is None).
    Returns:
        A dictionary containing the API response.
    """
    url = (
        "https://api.x.com/2/notes/search/posts_eligible_for_notes"
        f"?test_mode={'true' if test_mode else 'false'}"
        f"&max_results={max_results}"
        "&tweet.fields=author_id,created_at,referenced_tweets,media_metadata,note_tweet"
        "&expansions=author_id,attachments.media_keys,referenced_tweets.id"
        "&user.fields=username"
        "&media.fields=alt_text,duration_ms,height,media_key,preview_image_url,public_metrics,type,url,width,variants"
    )
    if pagination_token:
        url += f"&pagination_token={pagination_token}"
    response = oauth.get(url)
    response.raise_for_status()
    return response.json()

def _parse_individual_post(item: dict, media_by_key: dict[str, dict], users_by_id: dict[str, dict]) -> Post:
    media_objs: list[Media] = []
    media_keys = item.get("attachments", {}).get("media_keys", [])

    for key in media_keys:
        if key in media_by_key:
            media_obj = Media(**media_by_key[key])
            media_objs.append(media_obj)

    text = item["text"]
    note_tweet_text = item.get("note_tweet", {}).get("text", "")
    if note_tweet_text:
        text = note_tweet_text

    # Get username from users_by_id lookup
    author_id = item["author_id"]
    username = users_by_id.get(author_id, {}).get("username", "unknown")

    post = Post(
        post_id=item["id"],
        author_id=author_id,
        username=username,
        created_at=datetime.fromisoformat(
            item["created_at"].replace("Z", "+00:00")
        ),
        text=text,
        media=media_objs,
    )
    return post


def _parse_posts_eligible_response(resp: dict) -> list[PostWithContext]:
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

    includes_posts = resp.get("includes", {}).get("tweets", [])
    posts_by_id = {t["id"]: t for t in includes_posts}

    includes_users = resp.get("includes", {}).get("users", [])
    users_by_id = {u["id"]: u for u in includes_users}

    # rename type field to media_type to avoid name conflict with type
    for media_obj in media_by_key.values():
        media_obj["media_type"] = media_obj.pop("type")

    posts: List[PostWithContext] = []
    for item in resp.get("data", []):
        post = _parse_individual_post(item, media_by_key, users_by_id)

        # Handle quoted and in-reply-to posts ("referenced_tweets")
        quoted_post = None
        in_reply_to_post = None
        if 'referenced_tweets' in item:
            for ref in item["referenced_tweets"]:
                referenced_post_id = ref["id"]
                if referenced_post_id not in posts_by_id:
                    print(f"For post {post.post_id}, referenced post {referenced_post_id} not found in posts_by_id; skipping.")
                    continue
                referenced_post_item = posts_by_id[referenced_post_id]
                referenced_post = _parse_individual_post(referenced_post_item, media_by_key, users_by_id)

                if ref["type"] == "quoted":
                    assert quoted_post is None, "Multiple quoted posts found in a single post"
                    quoted_post = referenced_post
                elif ref["type"] == "replied_to":
                    assert in_reply_to_post is None, "Multiple in-reply-to posts found in a single post"
                    in_reply_to_post = referenced_post
                else:
                    raise ValueError(f"Unknown referenced tweet type: {ref['type']} (expected 'quoted' or 'replied_to')")

        post_with_context = PostWithContext(
            post=post,
            quoted_post=quoted_post,
            in_reply_to_post=in_reply_to_post,
        )
        posts.append(post_with_context)

    return posts


def get_posts_eligible_for_notes(
    oauth: OAuth1Session,
    max_results: int | None = None,
    test_mode: bool = True,
    max_results_per_page: int = 100,
) -> list[PostWithContext]:
    """
    Get posts eligible for notes by calling the Community Notes API.
    For more details, see: https://docs.x.com/x-api/community-notes/introduction

    Args:
        oauth: OAuth1Session object for authenticating with the X API.
        max_results: Maximum number of results to return (default is 2).
        test_mode: If True, use test mode for the API (default is True).

    Returns:
        A list of `Post` objects.
    """
    all_posts: list[PostWithContext] = []
    pagination_token: str | None = None
    
    while True:
        # Determine how many results to fetch in this page
        if max_results is not None:
            remaining = max_results - len(all_posts)
            if remaining <= 0:
                break
            page_size = min(remaining, max_results_per_page)
        else:
            page_size = max_results_per_page
        
        # Fetch a page of results
        response = _fetch_posts_eligible_for_notes(
            oauth, 
            max_results=page_size, 
            test_mode=test_mode,
            pagination_token=pagination_token
        )
        
        # Parse and add posts from this page
        posts = _parse_posts_eligible_response(response)
        all_posts.extend(posts)
        
        # Check if there are more pages
        pagination_token = response.get("meta", {}).get("next_token")
        if not pagination_token:
            break
    
    return all_posts
