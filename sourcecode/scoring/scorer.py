from abc import ABC, abstractmethod
from contextlib import contextmanager
import time
from typing import Dict, List, Optional, Tuple

from . import constants as c

import pandas as pd
import torch


class Scorer(ABC):
  """Base class which all other scorers must extend.

  The Scorer base class defines "score" function which wraps around _score_notes_and_users
  and works with other helper functions defining output columns.  This paradigm is designed
  to improve code readability and decrease bugs by forcing subclasses to be very clear about
  exactly which columns are output and which are dropped.
  """

  def __init__(self, seed: Optional[int] = None, threads: int = c.defaultNumThreads) -> None:
    """Configure a new Scorer object.

    Args:
      seed (int, optional): if not None, seed value to ensure deterministic execution
    """
    self._seed = seed
    self._threads = threads

  @contextmanager
  def time_block(self, label):
    start = time.time()
    try:
      yield
    finally:
      end = time.time()
      print(
        f"{self.get_name()} {label} elapsed time: {end - start:.2f} secs ({((end-start)/60.0):.2f} mins)"
      )

  def get_name(self):
    return str(type(self))

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

  def _postprocess_output(
    self,
    noteScores: pd.DataFrame,
    userScores: pd.DataFrame,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    userEnrollment: pd.DataFrame,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prune noteScores and userScores and augment with any additional columns as necessary.

    Note that ratings, noteStatusHistory and userEnrollment are expected to be the *raw*
    versions which were supplied to "score", not the version output after filtering.
    Operating on the raw versions allows accurately computing statistics over the entire dataset
    (e.g. fraction of users from a modeling group).

    Args:
      noteScores (pd.DataFrame): scoring output for notes
      userScores (pd.DataFrame): scoirng output for users
      ratings (pd.DataFrame): preprocessed ratings
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
      userEnrollment (pd.DataFrame): one row per user specifying enrollment properties

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame]:
        noteScores: note scoring output from _score_notes_and_users
        userScores: user scoring output from _score_notes_and_users
    """
    return noteScores, userScores

  def _get_note_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default note column names to custom names for a specific model."""
    return {}

  def _get_user_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default user column names to custom names for a specific model."""
    return {}

  @abstractmethod
  def _score_notes_and_users(
    self, ratings: pd.DataFrame, noteStatusHistory: pd.DataFrame, userEnrollmentRaw: pd.DataFrame
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process ratings to assign status to notes and optionally compute rater properties.

    Args:
      ratings (pd.DataFrame): preprocessed ratings
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
      userEnrollmentRaw (pd.DataFrame): one row per user specifying enrollment properties

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame]:
        noteScores pd.DataFrame: one row per note contained note scores and parameters.
        userScores pd.DataFrame: one row per user containing a column for each helpfulness score.
    """

  def score(
    self,
    ratingsRaw: pd.DataFrame,
    noteStatusHistoryRaw: pd.DataFrame,
    userEnrollmentRaw: pd.DataFrame,
  ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Process ratings to assign status to notes and optionally compute rater properties.

    Args:
      ratingsRaw (pd.DataFrame): preprocessed ratings
      noteStatusHistoryRaw (pd.DataFrame): one row per note; history of when note had each status
      userEnrollmentRaw (pd.DataFrame): one row per user specifying enrollment properties

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        scoredNotes pd.DataFrame: one row per note contained note scores and parameters.
        helpfulnessScores pd.DataFrame: one row per user containing a column for each helpfulness score.
        auxiliaryNoteInfo: one row per note containing adjusted and ratio tag values
    """
    torch.set_num_threads(self._threads)
    print(f"Torch intra-op parallelism for {self.get_name()} set to: {torch.get_num_threads()}")
    # Transform input, run core scoring algorithm, transform output.
    with self.time_block("Filter input"):
      ratings, noteStatusHistory = self._filter_input(
        ratingsRaw, noteStatusHistoryRaw, userEnrollmentRaw
      )
      # If there are no ratings left after filtering, then return empty dataframes.
      if len(ratings) == 0:
        return (
          pd.DataFrame(columns=self.get_scored_notes_cols()),
          pd.DataFrame(columns=self.get_helpfulness_scores_cols())
          if self.get_helpfulness_scores_cols()
          else None,
          pd.DataFrame(columns=self.get_auxiliary_note_info_cols())
          if self.get_auxiliary_note_info_cols()
          else None,
        )

    noteScores, userScores = self._score_notes_and_users(
      ratings, noteStatusHistory, userEnrollmentRaw
    )

    with self.time_block("Postprocess output"):
      noteScores, userScores = self._postprocess_output(
        noteScores, userScores, ratingsRaw, noteStatusHistoryRaw, userEnrollmentRaw
      )
      noteScores = noteScores.rename(columns=self._get_note_col_mapping())
      userScores = userScores.rename(columns=self._get_user_col_mapping())
      # TODO: Tolerate unexpcted columns if --nostrict-columns is set.
      # Process noteScores
      noteScores = noteScores.drop(columns=self._get_dropped_note_cols())
      assert set(noteScores.columns) == set(
        self.get_scored_notes_cols() + self.get_auxiliary_note_info_cols()
      ), f"""all columns must be either dropped or explicitly defined in an output. 
      Extra columns that were in noteScores: {set(noteScores.columns) - set(self.get_scored_notes_cols() + self.get_auxiliary_note_info_cols())}
      Missing expected columns that should've been in noteScores: {set(self.get_scored_notes_cols() + self.get_auxiliary_note_info_cols()) - set(noteScores.columns)}"""
      # Process userScores
      userScores = userScores.drop(columns=self._get_dropped_user_cols())
      assert set(userScores.columns) == set(self.get_helpfulness_scores_cols()), f"""all columns must be either dropped or explicitly defined in an output. 
      Extra columns that were in userScores: {set(userScores.columns) - set(self.get_helpfulness_scores_cols())}
      Missing expected columns that should've been in userScores: {set(self.get_helpfulness_scores_cols()) - set(userScores.columns)}"""

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
