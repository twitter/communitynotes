from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import pandas as pd


class Scorer(ABC):
  """Base class which all other scorers must extend.

  The Scorer base class defines "score" function which wraps around _score_notes_and_users
  and works with other helper functions defining output columns.  This paradigm is designed
  to improve code readability and decrease bugs by forcing subclasses to be very clear about
  exactly which columns are output and which are dropped.
  """

  def __init__(self, seed: Optional[int] = None) -> None:
    """Configure a new Scorer object.

    Args:
      seed (int, optional): if not None, seed value to ensure deterministic execution
    """
    self._seed = seed

  @abstractmethod
  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""

  @abstractmethod
  def get_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the helpfulnessScores output."""

  @abstractmethod
  def get_auxiliary_note_info_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the auxiliaryNoteInfo output."""

  @abstractmethod
  def _get_dropped_note_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from scoredNotes and auxiliaryNoteInfo."""

  @abstractmethod
  def _get_dropped_user_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from helpfulnessScores output."""

  def _filter_input(
    self, ratings: pd.DataFrame, noteStatusHistory: pd.DataFrame, userEnrollment: pd.DataFrame
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prune the contents of ratings and noteStatusHistory to scope model behavior.

    Args:
      ratings (pd.DataFrame): preprocessed ratings
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
      userEnrollment (pd.DataFrame): one row per user specifying enrollment properties

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame]:
        ratings: ratings filtered to only contain rows of interest
        noteStatusHistory: noteStatusHistory filtered to only contain rows of interest
    """
    return ratings, noteStatusHistory

  def _get_note_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default note column names to custom names for a specific model."""
    return {}

  def _get_user_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default user column names to custom names for a specific model."""
    return {}

  @abstractmethod
  def _score_notes_and_users(
    self, ratings: pd.DataFrame, noteStatusHistory: pd.DataFrame
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process ratings to assign status to notes and optionally compute rater properties.

    Args:
      ratings (pd.DataFrame): preprocessed ratings
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame]:
        noteScores pd.DataFrame: one row per note contained note scores and parameters.
        userScores pd.DataFrame: one row per user containing a column for each helpfulness score.
    """

  def score(
    self, ratings: pd.DataFrame, noteStatusHistory: pd.DataFrame, userEnrollment: pd.DataFrame
  ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Process ratings to assign status to notes and optionally compute rater properties.

    Args:
      ratings (pd.DataFrame): preprocessed ratings
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
      userEnrollment (pd.DataFrame): one row per user specifying enrollment properties

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        scoredNotes pd.DataFrame: one row per note contained note scores and parameters.
        helpfulnessScores pd.DataFrame: one row per user containing a column for each helpfulness score.
        auxiliaryNoteInfo: one row per note containing adjusted and ratio tag values
    """
    # Transform input, run core scoring algorithm, transform output.
    ratings, noteStatusHistory = self._filter_input(ratings, noteStatusHistory, userEnrollment)
    noteScores, userScores = self._score_notes_and_users(ratings, noteStatusHistory)
    noteScores = noteScores.rename(columns=self._get_note_col_mapping())
    userScores = userScores.rename(columns=self._get_user_col_mapping())
    # Process noteScores
    noteScores = noteScores.drop(columns=self._get_dropped_note_cols())
    assert set(noteScores.columns) == set(
      self.get_scored_notes_cols() + self.get_auxiliary_note_info_cols()
    ), f"""all columns must be either dropped or explicitly defined in an output. 
    Extra columns that were in noteScores: {set(noteScores.columns) - set(self.get_scored_notes_cols() + self.get_auxiliary_note_info_cols())}
    Missing expected columns that should've been in noteScores: {set(self.get_scored_notes_cols() + self.get_auxiliary_note_info_cols()) - set(noteScores.columns)}"""
    # Process userScores
    userScores = userScores.drop(columns=self._get_dropped_user_cols())
    assert set(userScores.columns) == set(self.get_helpfulness_scores_cols())
    # Return dataframes with specified columns in specified order
    return (
      noteScores[self.get_scored_notes_cols()],
      userScores[self.get_helpfulness_scores_cols()]
      if self.get_helpfulness_scores_cols()
      else None,
      noteScores[self.get_auxiliary_note_info_cols()]
      if self.get_auxiliary_note_info_cols()
      else None,
    )
