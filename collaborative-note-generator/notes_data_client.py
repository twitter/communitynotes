from abc import ABC, abstractmethod
import logging
from typing import Optional

from .constants import LiveNoteVersion, NoteContent, ScoringResult, Suggestion


class NotesDataClient(ABC):
  def __init__(self, logger: logging.Logger):
    self.logger = logger

  @abstractmethod
  def get_new_snowflake_id(self) -> Optional[int]:
    """Get a new snowflake ID for a live note version."""

  @abstractmethod
  def can_write_notes(self, user_id: int, total_misleading_notes_threshold: int = 10) -> bool:
    """Return whether a user can write notes."""

  @abstractmethod
  def fetch_all_note_ids_for_tweet(self, tweet_id: int) -> list[int]:
    """Return a list of note IDs for a tweet."""

  @abstractmethod
  def check_note_character_limit(self, note_text: str) -> bool:
    """Return True if note text fits character limits."""

  @abstractmethod
  def fetch_note_content_from_note_id(self, note_id: int) -> Optional[NoteContent]:
    """Fetch note content for a given note ID."""

  @abstractmethod
  def get_hydrated_previous_live_note_versions(self, tweet_id: int) -> list[LiveNoteVersion]:
    """Fetch previous live note versions, hydrated with suggestions and scoring."""

  @abstractmethod
  def get_previous_live_note_versions(self, tweet_id: int) -> list[LiveNoteVersion]:
    """Fetch previous live note versions for a tweet."""

  @abstractmethod
  def get_current_note_scoring_result(self, note_id: int) -> ScoringResult:
    """Fetch current scoring status and intercept for a note."""

  @abstractmethod
  def get_note_rating_summaries_by_factor_bucket(self, note_id: int) -> Optional[dict[str, int]]:
    """Return rating counts by rater-factor bucket (positive or negative) for Core model."""

  @abstractmethod
  def get_suggestions_for_live_note_version(self, live_note_version: int) -> list[Suggestion]:
    """Fetch suggestions for a given live note version."""

  @abstractmethod
  def get_note_contents(self, tweet_id: int) -> list[NoteContent]:
    """Fetch note contents for a tweet, including status/intercept if available."""
