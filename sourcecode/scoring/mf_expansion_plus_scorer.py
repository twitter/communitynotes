from typing import Dict, List, Optional, Tuple

from . import constants as c
from .mf_base_scorer import MFBaseScorer

import pandas as pd


class MFExpansionPlusScorer(MFBaseScorer):
  def __init__(
    self,
    seed: Optional[int] = None,
    useStableInitialization: bool = True,
    saveIntermediateState: bool = False,
    threads: int = c.defaultNumThreads,
  ) -> None:
    """Configure MFExpansionPlusScorer object.

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
    return "MFExpansionPlusScorer"

  def _get_note_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default note column names to custom names for a specific model."""
    return {
      c.internalNoteInterceptKey: c.expansionPlusNoteInterceptKey,
      c.internalNoteFactor1Key: c.expansionPlusNoteFactor1Key,
      c.internalRatingStatusKey: c.expansionPlusRatingStatusKey,
    }

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      c.expansionPlusNoteInterceptKey,
      c.expansionPlusNoteFactor1Key,
      c.expansionPlusRatingStatusKey,
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
        c.noteInterceptMinKey,
        c.noteInterceptMaxKey,
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

  def _postprocess_output(
    self,
    noteScores: pd.DataFrame,
    userScores: pd.DataFrame,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    userEnrollment: pd.DataFrame,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter noteScores to only include notes authored by EXPANSION_PLUS users.

    Args:
      noteScores: note outputs from scoring
      userScores: user outputs from scoring
      ratings (pd.DataFrame): preprocessed ratings
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
      userEnrollment (pd.DataFrame): one row per user specifying enrollment properties

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame]:
        noteScores: filtered and updated note scoring output
        userScores: unaltered
    """
    # Identify EXPANSION_PLUS users.
    expansionPlusAuthors = userEnrollment[
      userEnrollment[c.modelingPopulationKey] == c.expansionPlus
    ][[c.participantIdKey]].rename(columns={c.participantIdKey: c.noteAuthorParticipantIdKey})
    # Identify note written by EXPANSION_PLUS users.
    expnasionPlusNotes = noteStatusHistory.merge(
      expansionPlusAuthors, on=c.noteAuthorParticipantIdKey
    )[[c.noteIdKey]]
    # Prune to EXPANSION_PLUS users and return.
    noteScores = noteScores.merge(expnasionPlusNotes, on=c.noteIdKey)
    return noteScores, userScores
