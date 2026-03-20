from dataclasses import asdict, dataclass
from enum import Enum
import json
import textwrap
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


def format_dataclass(obj) -> str:
  data = asdict(obj)
  # Render long text fields with real newlines for readability in logs
  _TEXT_FIELDS = {"sources_considered", "long_live_note", "short_live_note", "story_assessment"}
  for key in _TEXT_FIELDS:
    if key in data and isinstance(data[key], str) and "\n" in data[key]:
      data[key] = "<<MULTILINE>>\n" + data[key] + "\n<<END>>"
  result = json.dumps(data, default=str, sort_keys=True, indent=2)
  # Unescape the newlines inside our multiline markers
  import re

  result = re.sub(
    r'"<<MULTILINE>>\\n(.*?)\\n<<END>>"',
    lambda m: '"\n' + m.group(1).replace("\\n", "\n").replace('\\"', '"') + '\n"',
    result,
    flags=re.DOTALL,
  )
  return result


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
  rating_tag_summary: Optional[dict] = None  # bucket → {tag: count}
  rating_level_summary: Optional[dict] = None  # bucket → {level: count}


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
class LiveNoteTrackingStats:
  generator_stats: dict[str, int]
  generator_failure: Optional[str] = None
  strato_failure: Optional[str] = None
  tracking_start_ms: Optional[int] = None
  tracking_end_ms: Optional[int] = None
  intended_failure: bool = False


@dataclass
class Source:
  url: Optional[str] = None
  explanation: Optional[str] = None
  created_at_ms: Optional[int] = None
  source_type: Optional[str] = None
  source_detail: Optional[str] = None


@dataclass
class GrokRejectorResult:
  score: Optional[float] = None
  reasoning: Optional[str] = None
  error: Optional[str] = None


@dataclass
class Evaluation:
  grok_rejector_results: Optional[list[GrokRejectorResult]] = None
  mean_grok_rejector_score: Optional[float] = None
  claim_opinion_model_score: Optional[float] = None


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
  story_assessment: Optional[str] = None
  rating_tag_summary: Optional[dict[str, int]] = None
  rating_level_summary: Optional[dict[str, dict[str, int]]] = None  # bucket → {HELPFUL: n, ...}
  total_ratings: Optional[int] = None
  parsed_sources: Optional[list[Source]] = None
  num_rejector_samples: int = 0
  evaluation: Optional[Evaluation] = None


@dataclass
class LiveNoteGenerationResult:
  live_note_version: Optional[LiveNoteVersion]
  tracking_stats: LiveNoteTrackingStats


class MediaMatchVerdict(Enum):
  YES = "YES"
  NO = "NO"
  INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class MediaComparisonVotes:
  yes_votes: int = 0
  no_votes: int = 0
  error_votes: int = 0


@dataclass
class UrlMediaComparisonResult:
  url: str
  same_media: MediaMatchVerdict
  same_incident: MediaMatchVerdict
  same_media_votes: Optional[MediaComparisonVotes] = None
  same_incident_votes: Optional[MediaComparisonVotes] = None


@dataclass
class ContextForGeneration:
  tweet_id: str
  note_contents: list[NoteContent]
  past_live_note_versions_with_suggestions: list[LiveNoteVersion]
  live_note_version_id: Optional[int] = None
  media_comparison_results: Optional[list[UrlMediaComparisonResult]] = None


# =============================================================================
# Logging utilities for formatting prompts and responses
# =============================================================================

_PROMPT_PREFIX = " »   "
_PROMPT_CONTINUATION = " »       "
_RESPONSE_PREFIX = " «   "
_RESPONSE_CONTINUATION = " «       "
_LOG_WRAP_WIDTH = 180


def _wrap_log_line(line: str, prefix: str, continuation: str) -> str:
  """Wrap a single long line, using prefix for first and continuation for rest."""
  if len(prefix + line) <= _LOG_WRAP_WIDTH or not line.strip():
    return prefix + line
  wrapped = textwrap.wrap(line, width=_LOG_WRAP_WIDTH - len(prefix))
  if not wrapped:
    return prefix + line
  return prefix + wrapped[0] + "".join(f"\n{continuation}{w}" for w in wrapped[1:])


def format_prompt_for_logging(prompt_text: str) -> str:
  """Format a prompt for logging with » prefix on each line."""
  return "\n".join(
    _wrap_log_line(line, _PROMPT_PREFIX, _PROMPT_CONTINUATION) for line in prompt_text.split("\n")
  )


def format_response_for_logging(response_text: str) -> str:
  """Format a response for logging with « prefix on each line."""
  return "\n".join(
    _wrap_log_line(line, _RESPONSE_PREFIX, _RESPONSE_CONTINUATION)
    for line in response_text.split("\n")
  )
