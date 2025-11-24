import json
from typing import Any, Dict

from requests_oauthlib import OAuth1Session  # type: ignore

from data_models import ProposedMisleadingNote


def submit_note(
    oauth: OAuth1Session,
    note: ProposedMisleadingNote,
    log_strings: list[str],
    test_mode: bool = True,
) -> Dict[str, Any]:
    """
    Submit a note to the Community Notes API. For more details, see:
    https://docs.x.com/x-api/community-notes/introduction

    Args:
        oauth: OAuth1Session object for authenticating with the X API.
        note: The note to submit.
        log_strings: List of log strings to append to.
        test_mode: If True, use test mode for the API (default is True).
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
        return response.json()
    except Exception as e:
        log_strings.append("\n*ERROR SUBMITTING NOTE*:")
        log_strings.append(f"  Failed to submit note: {e}")
        log_strings.append(f"  Response status: {response.status_code}")
        log_strings.append(f"  Response text: {response.text}")
        raise
    
