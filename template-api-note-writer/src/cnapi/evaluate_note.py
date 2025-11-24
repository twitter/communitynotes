from requests_oauthlib import OAuth1Session  # type: ignore


def evaluate_note(
    oauth: OAuth1Session,
    note_text: str,
    post_id: str,
    verbose_if_failed: bool = False,
) -> float:
    """
    Evaluate a note using the Community Notes API to get the claim opinion score.
    For more details, see:
    https://docs.x.com/x-api/community-notes/evaluate-a-community-note

    Args:
        oauth: OAuth1Session object for authenticating with the X API.
        note_text: Text for the community note.
        post_id: Tweet ID to evaluate the note against.
        verbose_if_failed: If True, print error details if the request fails.

    Returns:
        The claim opinion score for the note (float).
    """
    payload = {
        "note_text": note_text,
        "post_id": post_id,
    }

    url = "https://api.x.com/2/evaluate_note"
    response = oauth.post(url, json=payload)
    
    try:
        response.raise_for_status()
    except Exception as e:
        if verbose_if_failed:
            print(f"Failed to evaluate note: {e}")
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text}")
        raise
    
    response_data = response.json()
    claim_opinion_score = response_data.get("data", {}).get("claim_opinion_score")
    
    if claim_opinion_score is None:
        raise ValueError("Response did not contain claim_opinion_score in expected format")
    
    return claim_opinion_score

