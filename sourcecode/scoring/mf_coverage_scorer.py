from typing import Dict, List, Optional, Tuple

from . import constants as c, matrix_factorization
from .mf_core_scorer import MFCoreScorer

import pandas as pd


class MFCoverageScorer(MFCoreScorer):
  def __init__(self, seed: Optional[int] = None) -> None:
    """Configure MFCoverageScorer object.

    Args:
      seed: if not None, seed value to ensure deterministic execution
    """
    super().__init__(seed)
    self._crhThreshold = 0.338
    self._pseudoraters = False
    self._mfRanker = matrix_factorization.MatrixFactorization(l2_intercept_multiplier=7)

  def _get_note_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default note column names to custom names for a specific model."""
    return {
      c.internalNoteInterceptKey: c.coverageNoteInterceptKey,
      c.internalNoteFactor1Key: c.coverageNoteFactor1Key,
      c.internalRatingStatusKey: c.coverageRatingStatusKey,
      c.noteInterceptMinKey: c.coverageNoteInterceptMinKey,
      c.noteInterceptMaxKey: c.coverageNoteInterceptMaxKey,
    }

  def _get_user_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default user column names to custom names for a specific model."""
    return {}

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      c.coverageNoteInterceptKey,
      c.coverageNoteFactor1Key,
      c.coverageRatingStatusKey,
      c.coverageNoteInterceptMinKey,
      c.coverageNoteInterceptMaxKey,
    ]

  def get_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the helpfulnessScores output."""
    return []

  def get_auxiliary_note_info_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the auxiliaryNoteInfo output."""
    return []

  def _get_dropped_note_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from scoredNotes and auxiliaryNoteInfo."""
    return super()._get_dropped_note_cols() + (
      [
        c.internalActiveRulesKey,
        c.activeFilterTagsKey,
        c.ratingWeightKey,
      ]
      + c.notHelpfulTagsAdjustedColumns
      + c.notHelpfulTagsAdjustedRatioColumns
      + c.incorrectFilterColumns
      + c.noteParameterUncertaintyTSVAuxColumns
    )

  def _get_dropped_user_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from helpfulnessScores output."""
    return super()._get_dropped_user_cols() + [
      c.raterParticipantIdKey,
      c.internalRaterInterceptKey,
      c.internalRaterFactor1Key,
      c.crhCrnhRatioDifferenceKey,
      c.meanNoteScoreKey,
      c.raterAgreeRatioKey,
      c.aboveHelpfulnessThresholdKey,
    ]


class MFDummyCoverageScorer(MFCoverageScorer):
  def __init__(self, seed: Optional[int] = None) -> None:
    """Return empty columns from MFCoverageScorer object.

    Args:
      seed: if not None, seed value to ensure deterministic execution
    """
    super().__init__(seed)

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
    noteScores = pd.DataFrame(
      {
        c.noteIdKey: pd.Series(dtype="int"),
        c.internalNoteInterceptKey: pd.Series(dtype="double"),
        c.internalNoteFactor1Key: pd.Series(dtype="double"),
        c.noteInterceptMinKey: pd.Series(dtype="double"),
        c.noteInterceptMaxKey: pd.Series(dtype="double"),
        c.internalRatingStatusKey: pd.Series(dtype="str"),
      }
    )
    noteScores[c.noteIdKey] = ratings[c.noteIdKey].drop_duplicates()
    userScores = pd.DataFrame()
    return noteScores, userScores

  def _get_dropped_note_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from scoredNotes and auxiliaryNoteInfo."""
    return []

  def _get_dropped_user_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from helpfulnessScores output."""
    return []

  def _filter_input(
    self, ratings: pd.DataFrame, noteStatusHistory: pd.DataFrame, userEnrollment: pd.DataFrame
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return ratings, noteStatusHistory
