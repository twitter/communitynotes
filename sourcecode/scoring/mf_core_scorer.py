from typing import Dict, List, Optional, Tuple

from . import constants as c
from .mf_base_scorer import MFBaseScorer

import numpy as np
import pandas as pd


_CORE_BOOL = "coreBool"
_TOTAL = "total"
_RATIO = "ratio"


def filter_core_input(
  ratingsOrig: pd.DataFrame,
  noteStatusHistoryOrig: pd.DataFrame,
  userEnrollment: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Prune the contents of ratings and noteStatusHistory to scope model behavior.

  Filter ratings dataframe to only include ratings from CORE users.

  Args:
    ratings (pd.DataFrame): preprocessed ratings
    noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
    userEnrollment (pd.DataFrame): one row per user specifying enrollment properties

  Returns:
    Tuple[pd.DataFrame, pd.DataFrame]:
      ratings: ratings filtered to only contain rows of interest
      noteStatusHistory: noteStatusHistory filtered to only contain rows of interest
  """
  print("Identifying core notes and ratings")
  # Prune ratings that aren't defined as core
  ratings = ratingsOrig.merge(
    userEnrollment[[c.participantIdKey, c.modelingPopulationKey]].rename(
      columns={c.participantIdKey: c.raterParticipantIdKey}
    ),
    on=c.raterParticipantIdKey,
    how="left",
  )
  print(
    f"  Ratings from user without modelingPopulation: {pd.isna(ratings[c.modelingPopulationKey]).sum()}"
  )
  ratings = ratings.fillna({c.modelingPopulationKey: c.core})
  ratings = ratings[ratings[c.modelingPopulationKey] == c.core]
  print(f"  Core ratings: {len(ratings)}")
  return ratings.drop(columns=c.modelingPopulationKey), noteStatusHistoryOrig


def filter_core_output(
  ratingsOrig: pd.DataFrame,
  userEnrollment: pd.DataFrame,
  noteScores: pd.DataFrame,
  coreThreshold: float = 0.5,
) -> pd.DataFrame:
  # Drop ExpansionPlus ratings before determining ratios
  print("Filtering Core Output")
  print(f"  Original ratings length: {len(ratingsOrig)}")
  # Separate CORE and EXPANSION notes.
  userEnrollment[_CORE_BOOL] = userEnrollment[c.modelingPopulationKey] == c.core
  userGroups = userEnrollment[[c.participantIdKey, _CORE_BOOL]].copy()
  ratings = ratingsOrig.merge(
    userGroups.rename(columns={c.participantIdKey: c.raterParticipantIdKey}),
    on=c.raterParticipantIdKey,
    how="left",
    unsafeAllowed=_CORE_BOOL,
  )
  print(f"  Final ratings length: {len(ratings)}")
  ratings = ratings.fillna({_CORE_BOOL: True})
  ratings[_CORE_BOOL] = ratings[_CORE_BOOL].astype(np.bool8)
  ratios = ratings[[c.noteIdKey, _CORE_BOOL]].groupby(c.noteIdKey).mean().reset_index()
  # Identify CORE notes.  We define a CORE note to be any note which (1) has ratings, and
  # (2) half or more of the ratings are from CORE users.  This construction does mean that
  # notes without ratings can avoid locking, but as soon as they get enough ratings to be
  # captured and scored by CORE they will lock (if older than 2 weeks).
  print(f"  Original noteScores length: {len(noteScores)}")
  noteScores = noteScores.merge(ratios[ratios[_CORE_BOOL] >= coreThreshold][[c.noteIdKey]])
  print(f"  Final noteScores length: {len(noteScores)}")
  return noteScores


class MFCoreScorer(MFBaseScorer):
  def __init__(
    self,
    seed: Optional[int] = None,
    pseudoraters: Optional[bool] = False,
    useStableInitialization: bool = True,
    saveIntermediateState: bool = False,
    threads: int = c.defaultNumThreads,
  ) -> None:
    """Configure MFCoreScorer object.

    Args:
      seed: if not None, seed value to ensure deterministic execution
      pseudoraters: if True, compute optional pseudorater confidence intervals
      threads: number of threads to use for intra-op parallelism in pytorch
    """
    super().__init__(
      seed,
      pseudoraters,
      useStableInitialization=useStableInitialization,
      saveIntermediateState=saveIntermediateState,
      threads=threads,
    )

  def get_name(self):
    return "MFCoreScorer"

  def _get_note_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default note column names to custom names for a specific model."""
    return {
      c.internalNoteInterceptKey: c.coreNoteInterceptKey,
      c.internalNoteFactor1Key: c.coreNoteFactor1Key,
      c.internalRatingStatusKey: c.coreRatingStatusKey,
      c.internalActiveRulesKey: c.coreActiveRulesKey,
      c.noteInterceptMinKey: c.coreNoteInterceptMinKey,
      c.noteInterceptMaxKey: c.coreNoteInterceptMaxKey,
      c.numFinalRoundRatingsKey: c.coreNumFinalRoundRatingsKey,
      c.lowDiligenceNoteInterceptKey: c.lowDiligenceLegacyNoteInterceptKey,
    }

  def _get_user_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default user column names to custom names for a specific model."""
    return {
      c.internalRaterInterceptKey: c.coreRaterInterceptKey,
      c.internalRaterFactor1Key: c.coreRaterFactor1Key,
    }

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      c.coreNoteInterceptKey,
      c.coreNoteFactor1Key,
      c.coreRatingStatusKey,
      c.coreActiveRulesKey,
      c.activeFilterTagsKey,
      c.coreNoteInterceptMinKey,
      c.coreNoteInterceptMaxKey,
      c.coreNumFinalRoundRatingsKey,
    ]

  def get_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the helpfulnessScores output."""
    return [
      c.raterParticipantIdKey,
      c.coreRaterInterceptKey,
      c.coreRaterFactor1Key,
      c.crhCrnhRatioDifferenceKey,
      c.meanNoteScoreKey,
      c.raterAgreeRatioKey,
      c.aboveHelpfulnessThresholdKey,
    ]

  def _filter_input(
    self,
    noteTopics: pd.DataFrame,
    ratingsOrig: pd.DataFrame,
    noteStatusHistoryOrig: pd.DataFrame,
    userEnrollment: pd.DataFrame,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prune the contents of ratings and noteStatusHistory to scope model behavior."""
    return filter_core_input(ratingsOrig, noteStatusHistoryOrig, userEnrollment)

  def _postprocess_output(
    self,
    noteScores: pd.DataFrame,
    userScores: pd.DataFrame,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    userEnrollment: pd.DataFrame,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return filter_core_output(ratings, userEnrollment, noteScores), userScores
