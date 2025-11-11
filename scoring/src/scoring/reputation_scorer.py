import logging
from typing import Dict, List, Optional, Tuple

from . import constants as c
from .matrix_factorization.matrix_factorization import MatrixFactorization
from .mf_base_scorer import get_ratings_for_stable_init
from .process_data import filter_ratings
from .reputation_matrix_factorization.helpfulness_model import (
  get_helpfulness_reputation_results_final,
  get_helpfulness_reputation_results_prescoring,
)
from .scorer import EmptyRatingException, Scorer

import pandas as pd
import torch


logger = logging.getLogger("birdwatch.reputation_scorer")
logger.setLevel(logging.INFO)


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
    super().__init__(
      includedTopics=set(),
      includedGroups=c.coreGroups,
      includeUnassigned=True,
      captureThreshold=0.5,
      seed=seed,
      threads=threads,
    )
    self._minNumRatingsPerRater = minNumRatingsPerRater
    self._minNumRatersPerNote = minNumRatersPerNote
    self._crhThreshold = crhThreshold
    self._modelingGroupToInitializeForStability = 13 if useStableInitialization else None

  def get_name(self):
    return "ReputationScorer"

  def get_prescoring_name(self):
    return "ReputationScorer"

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      c.coverageNoteInterceptKey,
      c.coverageNoteFactor1Key,
      c.coverageRatingStatusKey,
    ]

  def get_internal_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      c.internalNoteInterceptKey,
      c.internalNoteFactor1Key,
      c.internalRatingStatusKey,
    ]

  def get_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the helpfulnessScores output."""
    return [c.raterParticipantIdKey, c.raterHelpfulnessReputationKey]

  def get_internal_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the helpfulnessScores output."""
    return [
      c.raterParticipantIdKey,
      c.internalRaterInterceptKey,
      c.internalRaterFactor1Key,
      c.internalRaterReputationKey,
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

  def _get_user_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default user column names to custom names for a specific model."""
    return {
      c.internalRaterReputationKey: c.raterHelpfulnessReputationKey,
    }

  def _prescore_notes_and_users(
    self,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    userEnrollmentRaw: pd.DataFrame,
  ) -> Tuple[pd.DataFrame, pd.DataFrame, c.PrescoringMetaScorerOutput]:
    if self._seed is not None:
      logger.info(f"seeding with {self._seed}")
      torch.manual_seed(self._seed)
    ratings = filter_ratings(ratings, self._minNumRatingsPerRater, self._minNumRatersPerNote)
    # Calculate initialization factors if necessary
    noteParamsInit = None
    raterParamsInit = None
    if self._modelingGroupToInitializeForStability:
      ratingsForStableInitialization = get_ratings_for_stable_init(
        ratings, userEnrollmentRaw, self._modelingGroupToInitializeForStability
      )
      mfRanker = MatrixFactorization()
      noteParamsInit, raterParamsInit, _ = mfRanker.run_mf(
        ratingsForStableInitialization, run_name=f"{self.get_name()}/stable_init"
      )

      # We only want to use factors to initialize, not intercepts
      noteParamsInit = noteParamsInit[[c.noteIdKey, c.internalNoteFactor1Key]]
      raterParamsInit = raterParamsInit[[c.raterParticipantIdKey, c.internalRaterFactor1Key]]

    # Fit multi-phase prescoring for reputation model
    (
      noteStats,
      raterStats,
      globalIntercept,
    ) = get_helpfulness_reputation_results_prescoring(
      ratings, noteInitState=noteParamsInit, raterInitState=raterParamsInit
    )
    # Fill in NaN values for any missing notes
    noteStats = noteStats.merge(noteStatusHistory[[c.noteIdKey]].drop_duplicates(), how="outer")
    assert len(noteStats) == len(noteStatusHistory)
    logger.info(
      f"""Reputation prescoring: returning these columns:
          noteStats: {noteStats.columns}
          raterStats: {raterStats.columns}
          """
    )

    metaScorerOutput = c.PrescoringMetaScorerOutput(
      globalIntercept=None,
      lowDiligenceGlobalIntercept=globalIntercept,
      tagFilteringThresholds=None,
      finalRoundNumRatings=None,
      finalRoundNumNotes=None,
      finalRoundNumUsers=None,
    )
    return noteStats, raterStats, metaScorerOutput

  def _score_notes_and_users(
    self,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    prescoringNoteModelOutput: pd.DataFrame,
    prescoringRaterModelOutput: pd.DataFrame,
    prescoringMetaScorerOutput: c.PrescoringMetaScorerOutput,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if self._seed is not None:
      logger.info(f"seeding with {self._seed}")
      torch.manual_seed(self._seed)
    ratings = filter_ratings(ratings, self._minNumRatingsPerRater, self._minNumRatersPerNote)
    if len(ratings) == 0:
      raise EmptyRatingException()

    # Apply model
    # Note: we use the low diligence global intercept here as a temporary hack, since the prod scorer's
    # globalIntercept field is a float and we need to store a c.ReputationGlobalIntercept.
    assert (
      prescoringMetaScorerOutput.lowDiligenceGlobalIntercept is not None
    ), "Missing prescoring global intercept"
    noteStats, raterStats = get_helpfulness_reputation_results_final(
      ratings,
      noteInitState=prescoringNoteModelOutput,
      raterInitState=prescoringRaterModelOutput,
      globalIntercept=prescoringMetaScorerOutput.lowDiligenceGlobalIntercept,
    )
    # Assign rating status
    noteStats[c.coverageRatingStatusKey] = c.needsMoreRatings
    noteStats.loc[
      noteStats[c.coverageNoteInterceptKey] > self._crhThreshold,
      c.coverageRatingStatusKey,
    ] = c.currentlyRatedHelpful
    # Fill in NaN values for any missing notes
    noteStats = noteStats.merge(noteStatusHistory[[c.noteIdKey]].drop_duplicates(), how="outer")
    assert len(noteStats) == len(noteStatusHistory)
    return noteStats, raterStats
