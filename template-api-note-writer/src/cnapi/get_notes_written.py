from datetime import datetime
from typing import List, Optional

from requests_oauthlib import OAuth1Session  # type: ignore

from data_models import NoteStatus, TestResult


def get_notes_written(
    oauth: OAuth1Session,
    test_mode: bool = True,
    max_results: Optional[int] = None,
) -> List[NoteStatus]:
    """
    Get all notes written by the authenticated user from the Community Notes API.
    For more details, see: https://docs.x.com/x-api/community-notes/introduction
    
    Args:
        oauth: OAuth1Session object for authenticating with the X API.
        test_mode: If True, use test mode for the API (default is True).
        max_results: Maximum number of results to return. If None, fetch all results (default is None).
    
    Returns:
        A list of NoteStatus objects containing information about written notes.
    """
    all_notes: List[NoteStatus] = []
    pagination_token: Optional[str] = None
    
    while True:
        # Build the URL with query parameters
        url = (
            "https://api.x.com/2/notes/search/notes_written"
            f"?test_mode={'true' if test_mode else 'false'}"
            "&max_results=100"  # API max per page
        )
        
        if pagination_token:
            url += f"&pagination_token={pagination_token}"
        
        # Make the API request
        response = oauth.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Parse the response
        notes_data = data.get("data", [])
        for note_item in notes_data:
            # Parse test_result if present
            test_result_list: Optional[List[TestResult]] = None
            test_result_raw = note_item.get("test_result")
            if test_result_raw and isinstance(test_result_raw, dict):
                evaluation_outcomes = test_result_raw.get("evaluation_outcome", [])
                if evaluation_outcomes:
                    test_result_list = [
                        TestResult(
                            evaluator_score_bucket=outcome["evaluator_score_bucket"],
                            evaluator_type=outcome["evaluator_type"]
                        )
                        for outcome in evaluation_outcomes
                    ]
            
            note_status = NoteStatus(
                note_id=note_item["id"],
                post_id=note_item["info"]["post_id"],
                note_text=note_item["info"]["text"],
                status=note_item["status"],
                test_result=test_result_list,
            )
            all_notes.append(note_status)
            
            # Check if we've reached max_results
            if max_results is not None and len(all_notes) >= max_results:
                return all_notes[:max_results]
        
        # Check for pagination
        meta = data.get("meta", {})
        pagination_token = meta.get("next_token")
        
        # Break if no more pages
        if not pagination_token:
            break
    
    return all_notes

