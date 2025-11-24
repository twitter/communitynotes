import json
from typing import Any, Dict

from requests_oauthlib import OAuth1Session  # type: ignore

from data_models import ProposedMisleadingNote


def submit_note(
    oauth: OAuth1Session,
    note: ProposedMisleadingNote,
    test_mode: bool = True,
    verbose_if_failed: bool = False,
) -> Dict[str, Any]:
    """
    Submit a note to the Community Notes API. For more details, see:
    https://docs.x.com/x-api/community-notes/introduction

    Args:
        oauth: OAuth1Session object for authenticating with the X API.
        note: The note to submit.
        test_mode: If True, use test mode for the API (default is True).
        verbose_if_failed: If True, print error details if the request fails.
    """
    payload = {
        "test_mode": test_mode,
        "post_id": note.post_id,
        "info": {
            "text": note.note_text,
            "classification": "misinformed_or_potentially_misleading",
            "misleading_tags": [tag.value for tag in note.misleading_tags],
            "trustworthy_sources": True,
        },
    }

    url = "https://api.x.com/2/notes"
    response = oauth.post(url, json=payload)
    
    try:
        response.raise_for_status()
    except Exception as e:
        if verbose_if_failed:
            print(f"Failed to submit note: {e}")
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text}")
        raise
    
    return response.json()
