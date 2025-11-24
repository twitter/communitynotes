from typing import Any

from datetime import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel


class EnvironmentVariables(BaseModel):
    api_key: str
    api_secret_key: str
    access_token: str
    access_token_secret: str
    xai_api_key: str


class ProposedNote(BaseModel):
    post_id: str
    note_text: str
    trustworthy_sources: bool = True


class MisleadingTag(str, Enum):
    factual_error = "factual_error"
    manipulated_media = "manipulated_media"
    outdated_information = "outdated_information"
    missing_important_context = "missing_important_context"
    disputed_claim_as_fact = "disputed_claim_as_fact"
    misinterpreted_satire = "misinterpreted_satire"
    other = "other"


class ProposedMisleadingNote(ProposedNote):
    misleading_tags: List[MisleadingTag]


class Media(BaseModel):
    media_key: str
    media_type: str
    url: str | None = None
    preview_image_url: str | None = None
    height: int | None = None
    width: int | None = None
    duration_ms: int | None = None
    view_count: int | None = None


class Post(BaseModel):
    post_id: str
    author_id: str
    username: str
    created_at: datetime
    text: str
    media: List[Media]


class PostWithContext(BaseModel):
    post: Post
    quoted_post: Post | None = None
    in_reply_to_post: Post | None = None


class NoteResult(BaseModel):
    writing_prompt: str | None = None
    note: ProposedMisleadingNote | None = None
    refusal: str | None = None
    error: str | None = None
    post: PostWithContext | None = None
    citations: list[str] | None = None
    tool_calls: list[Any] | None = None


class TestResult(BaseModel):
    evaluator_score_bucket: str
    evaluator_type: str


class NoteStatus(BaseModel):
    note_id: str
    post_id: str
    status: str
    test_result: List[TestResult] | None = None
    note_text: str | None = None