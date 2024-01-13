from typing import List, Optional, Tuple

from . import constants as c
from .matrix_factorization.matrix_factorization import MatrixFactorization
from .mf_base_scorer import get_ratings_for_stable_init
from .mf_core_scorer import filter_core_input
from .process_data import filter_ratings
from .reputation_matrix_factorization.helpfulness_model import get_helpfulness_reputation_results
from .scorer import Scorer

import pandas as pd
import torch


class ReputationScorer(Scorer):
  """Applies reputation matrix factorization to helpfulness bridging."""

  def __init__(
    self,
    seed: Optional[int] = None,
    threads: int = c.defaultNumThreads,
    minNumRatingsPerRater: int = 10,
    minNumRatersPerNote: int = 5,
    crhThreshold: float = 0.28,
    useStableInitialization: bool = True,
  ):
    """Configure ReputationScorer object.

    Args:
      seed: if not None, seed value to ensure deterministic execution
      minNumRatingsPerRater: Minimum number of ratings which a rater must produce to be
        included in scoring.  Raters with fewer ratings are removed.
      minNumRatersPerNote: Minimum number of ratings which a note must have to be included
        in scoring.  Notes with fewer ratings are removed.
      threads: number of threads to use for intra-op parallelism in pytorch
    """
    super().__init__(seed, threads)
    self._minNumRatingsPerRater = minNumRatingsPerRater
    self._minNumRatersPerNote = minNumRatersPerNote
    self._crhThreshold = crhThreshold
    self._modelingGroupToInitializeForStability = 13 if useStableInitialization else None

  def get_name(self):
    return "ReputationScorer"

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      c.coverageNoteInterceptKey,
      c.coverageNoteFactor1Key,
      c.coverageRatingStatusKey,
    ]

  def get_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the helpfulnessScores output."""
    return [
      c.raterParticipantIdKey,
      c.raterHelpfulnessReputationKey,
    ]

  def get_auxiliary_note_info_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the auxiliaryNoteInfo output."""
    return []

  def _get_dropped_note_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from scoredNotes and auxiliaryNoteInfo."""
    return []

  def _get_dropped_user_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from helpfulnessScores output."""
    return []

  def _filter_input(
    self, ratings: pd.DataFrame, noteStatusHistory: pd.DataFrame, userEnrollment: pd.DataFrame
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings, noteStatusHistory = filter_core_input(ratings, noteStatusHistory, userEnrollment)
    ratings = filter_ratings(ratings, self._minNumRatingsPerRater, self._minNumRatersPerNote)
    return ratings, noteStatusHistory

  def _score_notes_and_users(
    self, ratings: pd.DataFrame, noteStatusHistory: pd.DataFrame, userEnrollmentRaw: pd.DataFrame
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if self._seed is not None:
      print(f"seeding with {self._seed}")
      torch.manual_seed(self._seed)
    # Calculate initialization factors if necessary
    noteParamsInit = None
    raterParamsInit = None
    if self._modelingGroupToInitializeForStability:
      ratingsForStableInitialization = get_ratings_for_stable_init(
        ratings, userEnrollmentRaw, self._modelingGroupToInitializeForStability
      )
      mfRanker = MatrixFactorization()
      noteParamsInit, raterParamsInit, _ = mfRanker.run_mf(ratingsForStableInitialization)
    # Apply model
    noteStats, raterStats = get_helpfulness_reputation_results(
      ratings, noteInitState=noteParamsInit, raterInitState=raterParamsInit
    )
    # Assign rating status
    noteStats[c.coverageRatingStatusKey] = c.needsMoreRatings
    noteStats.loc[
      noteStats[c.coverageNoteInterceptKey] > self._crhThreshold, c.coverageRatingStatusKey
    ] = c.currentlyRatedHelpful
    # Fill in NaN values for any missing notes
    noteStats = noteStats.merge(noteStatusHistory[[c.noteIdKey]].drop_duplicates(), how="outer")
    assert len(noteStats) == len(noteStatusHistory)
    return noteStats, raterStats
