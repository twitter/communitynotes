from dataclasses import dataclass
from enum import Enum
from typing import Optional


liveNoteClassificationOpinion = "Mostly or entirely a statement of opinion"
liveNoteClassificationOpinionButInaccurate = (
  "Post is substantially opinion, though there might be some inaccuracies"
)
liveNoteClassificationMainPointHoldsButInaccurate = (
  "Main point of post stands, though there might be some inaccuracies"
)
liveNoteClassificationInaccurate = "Post does not appear entirely accurate"
liveNoteClassificationNoInaccuraciesFound = (
  "Haven't yet found inaccuracies but this might change as more information becomes available"
)

liveNoteCategoryMisleading = "M"
liveNoteCategoryNotMisleading = "NM"

noteAuthorUserId = int(1998144127482146816)


class RatingStatus(str, Enum):
  NeedsMoreRatings = "NeedsMoreRatings"
  CurrentlyRatedHelpful = "CurrentlyRatedHelpful"
  CurrentlyRatedNotHelpful = "CurrentlyRatedNotHelpful"
  NmrDueToMinStableCrhTime = "NmrDueToMinStableCrhTime"


def rating_status_from_string(status: Optional[str]) -> RatingStatus:
  if status is None:
    return RatingStatus.NeedsMoreRatings
  return RatingStatus(status)


@dataclass
class ScoringResult:
  intercept: float
  status: RatingStatus
  rating_summary: Optional[dict[str, int]] = None


@dataclass
class NoteContent:
  note_id: int
  summary: str
  classification: str
  created_at_ms: Optional[int] = None
  final_status: Optional[str] = None
  core_intercept: Optional[float] = None


@dataclass
class Suggestion:
  user_id: int
  created_at_ms: int
  suggestion_id: int
  suggestion_text: str


@dataclass
class RejectionDecision:
  should_reject: bool
  rejection_reason: Optional[str] = None
  retryable: Optional[bool] = None


@dataclass
class UpdateDecision:
  should_update: bool
  update_explanation: str
  difference_from_previous: str
  previous_published_version_id: Optional[int] = None


@dataclass
class SuggestionEvaluation:
  is_incorporated: bool
  incorporated_status: Optional[str] = None
  incorporated_explanation: Optional[str] = None
  decision_explanation: Optional[str] = None
  post_hoc_incorporated_status: Optional[str] = None
  post_hoc_incorporated_explanation: Optional[str] = None
  post_hoc_decision_explanation: Optional[str] = None


@dataclass
class NotificationInfo:
  users_who_added_suggestions: list[int]
  users_whose_suggestions_were_accepted: list[int]


@dataclass
class LiveNoteVersion:
  live_note_classification: str
  category: str
  short_live_note: str
  long_live_note: str
  sources_considered: str
  created_at_ms: Optional[int] = None
  version_id: Optional[int] = None
  suggestions: Optional[list[Suggestion]] = None
  update_decision: Optional[UpdateDecision] = None
  rejection_decision: Optional[RejectionDecision] = None
  suggestion_evaluations: Optional[dict[int, SuggestionEvaluation]] = None
  notifications: Optional[NotificationInfo] = None
  scoring_result: Optional[ScoringResult] = None


@dataclass
class ContextForGeneration:
  tweet_id: str
  note_contents: list[NoteContent]
  past_live_note_versions_with_suggestions: list[LiveNoteVersion]
  live_note_version_id: Optional[int] = None
