from typing import Dict, List, Optional, Tuple

from . import constants as c
from .mf_base_scorer import MFBaseScorer

import numpy as np
import pandas as pd


_EXPANSION_PLUS_BOOL = "expansionPlusBool"


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

  def _filter_input(
    self,
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
    # Prepare userEnrollment for join with ratings.
    userEnrollment[_EXPANSION_PLUS_BOOL] = (
      userEnrollment[c.modelingPopulationKey] == c.expansionPlus
    )
    userEnrollment = userEnrollment[[c.participantIdKey, _EXPANSION_PLUS_BOOL]].copy()
    print("Identifying expansion notes and ratings")
    # Prune notes authored by EXPANSION_PLUS users.
    print(f"  Total notes: {len(noteStatusHistoryOrig)}")
    noteStatusHistory = noteStatusHistoryOrig.merge(
      userEnrollment.rename(columns={c.participantIdKey: c.noteAuthorParticipantIdKey}),
      on=c.noteAuthorParticipantIdKey,
      how="left",
    )
    print(
      f"  Notes from user without modelingPopulation: {pd.isna(noteStatusHistory[_EXPANSION_PLUS_BOOL]).sum()}"
    )
    noteStatusHistory = noteStatusHistory.fillna({_EXPANSION_PLUS_BOOL: False})
    noteStatusHistory[_EXPANSION_PLUS_BOOL] = noteStatusHistory[_EXPANSION_PLUS_BOOL].astype(
      np.bool8
    )
    noteStatusHistory = noteStatusHistory[~noteStatusHistory[_EXPANSION_PLUS_BOOL]]
    print(f"  Total CORE and EXPANSION notes: {len(noteStatusHistory)}")
    # Prune ratings from EXPANSION_PLUS users.
    print(f"  Total ratings: {len(ratingsOrig)}")
    ratings = ratingsOrig.merge(
      userEnrollment.rename(columns={c.participantIdKey: c.raterParticipantIdKey}),
      on=c.raterParticipantIdKey,
      how="left",
    )
    print(
      f"  Ratings from user without modelingPopulation: {pd.isna(ratings[_EXPANSION_PLUS_BOOL]).sum()}"
    )
    ratings = ratings.fillna({_EXPANSION_PLUS_BOOL: False})
    ratings[_EXPANSION_PLUS_BOOL] = ratings[_EXPANSION_PLUS_BOOL].astype(np.bool8)
    ratings = ratings[~ratings[_EXPANSION_PLUS_BOOL]]
    print(f"  Ratings after EXPANSION_PLUS filter: {len(ratings)}")
    # prune ratings on dropped notes
    ratings = ratings.merge(
      noteStatusHistory[[c.noteIdKey]].drop_duplicates(), on=c.noteIdKey, how="inner"
    )
    print(f"  Ratings after EXPANSION_PLUS notes filter: {len(ratings)}")

    return ratings.drop(columns=_EXPANSION_PLUS_BOOL), noteStatusHistory.drop(
      columns=_EXPANSION_PLUS_BOOL
    )
