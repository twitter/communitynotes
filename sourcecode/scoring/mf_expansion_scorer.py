from typing import Dict, List, Optional, Tuple

from . import constants as c
from .mf_base_scorer import MFBaseScorer

import numpy as np
import pandas as pd


_EXPANSION_BOOL = "expansionBool"


class MFExpansionScorer(MFBaseScorer):
  def __init__(
    self,
    seed: Optional[int] = None,
    useStableInitialization: bool = True,
    saveIntermediateState: bool = False,
    threads: int = c.defaultNumThreads,
  ) -> None:
    """Configure MFExpansionScorer object.

    Args:
      seed: if not None, seed value to ensure deterministic execution
      threads: number of threads to use for intra-op parallelism in pytorch
    """
    super().__init__(
      seed,
      pseudoraters=False,
      useStableInitialization=useStableInitialization,
      saveIntermediateState=saveIntermediateState,
      threads=threads,
    )
    self._expansionThreshold = 0.5

  def get_name(self):
    return "MFExpansionScorer"

  def _get_note_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default note column names to custom names for a specific model."""
    return {
      c.internalNoteInterceptKey: c.expansionNoteInterceptKey,
      c.internalNoteFactor1Key: c.expansionNoteFactor1Key,
      c.internalRatingStatusKey: c.expansionRatingStatusKey,
      c.noteInterceptMinKey: c.expansionNoteInterceptMinKey,
      c.noteInterceptMaxKey: c.expansionNoteInterceptMaxKey,
      c.internalActiveRulesKey: c.expansionInternalActiveRulesKey,
      c.numFinalRoundRatingsKey: c.expansionNumFinalRoundRatingsKey,
      c.lowDiligenceNoteInterceptKey: c.lowDiligenceLegacyNoteInterceptKey,
    }

  def _get_user_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default user column names to custom names for a specific model."""
    return {
      c.internalRaterInterceptKey: c.expansionRaterInterceptKey,
      c.internalRaterFactor1Key: c.expansionRaterFactor1Key,
    }

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      c.expansionNoteInterceptKey,
      c.expansionNoteFactor1Key,
      c.expansionRatingStatusKey,
      c.expansionNoteInterceptMinKey,
      c.expansionNoteInterceptMaxKey,
      c.expansionInternalActiveRulesKey,
      c.expansionNumFinalRoundRatingsKey,
    ]

  def get_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the helpfulnessScores output."""
    return [
      c.raterParticipantIdKey,
      c.expansionRaterInterceptKey,
      c.expansionRaterFactor1Key,
    ]

  def get_auxiliary_note_info_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the auxiliaryNoteInfo output."""
    return []

  def _get_dropped_note_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from scoredNotes and auxiliaryNoteInfo."""
    return super()._get_dropped_note_cols() + (
      [
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
    """Prune the contents of ratings to scope model behavior.

    The MFExpansionScorer input is filtered to exclude notes and ratings from EXPANSION_PLUS
    users.  All other ratings are included.

    Args:
      ratings (pd.DataFrame): preprocessed ratings
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
      userEnrollment (pd.DataFrame): one row per user specifying enrollment properties

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame]:
        ratingsOrig: ratings filtered to only contain rows of interest
        noteStatusHistoryOrig: noteStatusHistory filtered to only contain rows of interest
    """
    print("Identifying expansion notes and ratings")
    # Prune ratings to CORE and EXPANSION users.
    print(f"  Total ratings: {len(ratingsOrig)}")
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
    ratings = ratings.fillna({c.modelingPopulationKey: c.expansion})
    ratings = ratings[ratings[c.modelingPopulationKey] != c.expansionPlus]
    print(f"  Ratings after EXPANSION_PLUS filter: {len(ratings)}")

    return ratings.drop(columns=c.modelingPopulationKey), noteStatusHistoryOrig

  def _postprocess_output(
    self,
    noteScores: pd.DataFrame,
    userScores: pd.DataFrame,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    userEnrollment: pd.DataFrame,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Filtering Expansion Output")
    print(f"  Original ratings length: {len(ratings)}")
    # Separate CORE and EXPANSION notes from EXPANSION_PLUS.
    userEnrollment[_EXPANSION_BOOL] = userEnrollment[c.modelingPopulationKey].isin(
      {c.core, c.expansion}
    )
    userGroups = userEnrollment[[c.participantIdKey, _EXPANSION_BOOL]].copy()
    ratings = ratings.merge(
      userGroups.rename(columns={c.participantIdKey: c.raterParticipantIdKey}),
      on=c.raterParticipantIdKey,
      how="left",
      unsafeAllowed=_EXPANSION_BOOL,
    )
    print(f"  Final ratings length: {len(ratings)}")
    ratings = ratings.fillna({_EXPANSION_BOOL: True})
    ratings[_EXPANSION_BOOL] = ratings[_EXPANSION_BOOL].astype(np.bool8)
    ratios = ratings[[c.noteIdKey, _EXPANSION_BOOL]].groupby(c.noteIdKey).mean().reset_index()
    # Identify EXPANSION notes.  We define a EXPANSION note to be any note which (1) has ratings, and
    # (2) half or more of the ratings are from EXPANSION/CORE users.
    print(f"  Original noteScores length: {len(noteScores)}")
    noteScores = noteScores.merge(
      ratios[ratios[_EXPANSION_BOOL] >= self._expansionThreshold][[c.noteIdKey]]
    )
    print(f"  Final noteScores length: {len(noteScores)}")
    return noteScores, userScores
