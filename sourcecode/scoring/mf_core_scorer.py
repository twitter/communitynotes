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
  coreThreshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Prune the contents of ratings and noteStatusHistory to scope model behavior.

  This function identifies the subset of note and ratings to include in core model scoring.
  A note is included in the core model if >50% of the ratings on the note come from users
  in the CORE modelingPopulation.  A rating is included in the core model if the rating is
  on a CORE note *and* the rating is from a user in the CORE modeling population.

  Note that the criteria above implies that a note without any ratings can't be included in
  the CORE model, which is acceptable because notes without ratings will be assigned a default
  status of NEEDS_MORE_RATINGS by both the EXPANSION model and meta_score.

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
  # Identify EXPANSION_PLUS users and notes
  expansionPlusUsers = set(
    userEnrollment[userEnrollment[c.modelingPopulationKey] == c.expansionPlus][
      c.participantIdKey
    ].values
  )
  print(f"  EXPANSION_PLUS users: {len(expansionPlusUsers)}")
  expansionPlusNotes = set(
    noteStatusHistoryOrig[
      noteStatusHistoryOrig[c.noteAuthorParticipantIdKey].isin(expansionPlusUsers)
    ][c.noteIdKey].values
  )
  print(f"  EXPANSION_PLUS notes: {len(expansionPlusNotes)}")
  # Remove EXPANSION_PLUS users and notes
  print(f"  original note status history length: {len(noteStatusHistoryOrig)}")
  noteStatusHistory = noteStatusHistoryOrig[
    ~noteStatusHistoryOrig[c.noteAuthorParticipantIdKey].isin(expansionPlusUsers)
  ]
  print(f"  note status history length after EXPANSION_PLUS filter: {len(noteStatusHistory)}")
  print(f"  original ratings length: {len(ratingsOrig)}")
  ratings = ratingsOrig[~ratingsOrig[c.raterParticipantIdKey].isin(expansionPlusUsers)]
  ratings = ratings[~ratings[c.noteIdKey].isin(expansionPlusNotes)]
  print(f"  ratings length after EXPANSION_PLUS filter: {len(ratings)}")
  # Separate CORE and EXPANSION notes.
  userEnrollment[_CORE_BOOL] = userEnrollment[c.modelingPopulationKey] == c.core
  userGroups = userEnrollment[[c.participantIdKey, _CORE_BOOL]].copy()
  ratings = ratings.merge(
    userGroups.rename(columns={c.participantIdKey: c.raterParticipantIdKey}),
    on=c.raterParticipantIdKey,
    how="left",
  )
  print(f"  Ratings from user without modelingPopulation: {pd.isna(ratings[_CORE_BOOL]).sum()}")
  ratings = ratings.fillna({_CORE_BOOL: True})
  ratings[_CORE_BOOL] = ratings[_CORE_BOOL].astype(np.bool8)
  counts = ratings[[c.noteIdKey, _CORE_BOOL]].copy()
  counts[_TOTAL] = 1
  counts = counts.groupby(c.noteIdKey).sum(numeric_only=True).reset_index()
  counts[_RATIO] = counts[_CORE_BOOL] / counts[_TOTAL]
  # Identify CORE notes.  We define an EXPANSION note to be any note which (1) has ratings
  # and (2) less than half of the ratings are from CORE users.  Any other note is considered
  # a CORE note.  This construction means that we only count a note as EXPANSION when there
  # is reason to believe that the EXPANSION model could assign the note status.  In all other
  # case we leave the note as CORE so that the note will be eligble for locking.  In effect,
  # this approach biases us towards locking note status at 2 weeks and only avoiding locking
  # when a note is scored by the EXPANSION model.
  print(f"  Total notes: {len(noteStatusHistory)}")
  print(f"  Total notes with ratings: {len(counts)}")
  expansionNotes = set(counts[counts[_RATIO] <= coreThreshold][c.noteIdKey])
  coreNotes = set(noteStatusHistory[c.noteIdKey]) - expansionNotes
  print(f"  Total core notes: {len(coreNotes)}")
  print(f"  Total expansion notes: {len(expansionNotes)}")
  # Prune notes and ratings to ratings from CORE users on CORE notes.
  ratings = ratings[ratings[_CORE_BOOL]]
  ratings = ratings.drop(columns=_CORE_BOOL)
  ratings = ratings[ratings[c.noteIdKey].isin(coreNotes)]
  noteStatusHistory = noteStatusHistory[noteStatusHistory[c.noteIdKey].isin(coreNotes)]
  print(f"  Core ratings: {len(ratings)}")
  return ratings, noteStatusHistory


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
    ratingsOrig: pd.DataFrame,
    noteStatusHistoryOrig: pd.DataFrame,
    userEnrollment: pd.DataFrame,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prune the contents of ratings and noteStatusHistory to scope model behavior."""
    return filter_core_input(ratingsOrig, noteStatusHistoryOrig, userEnrollment)
