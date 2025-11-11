from typing import Dict, List, Optional

from . import constants as c
from .mf_base_scorer import MFBaseScorer


class MFExpansionPlusScorer(MFBaseScorer):
  def __init__(
    self,
    seed: Optional[int] = None,
    useStableInitialization: bool = True,
    saveIntermediateState: bool = False,
    threads: int = c.defaultNumThreads,
    minMinorityNetHelpfulRatings: Optional[int] = 4,
    minMinorityNetHelpfulRatio: Optional[float] = 0.05,
  ) -> None:
    """Configure MFExpansionPlusScorer object.

    Args:
      seed: if not None, seed value to ensure deterministic execution
      threads: number of threads to use for intra-op parallelism in pytorch
    """
    super().__init__(
      includedGroups=(c.coreGroups | c.expansionGroups | c.expansionPlusGroups),
      includeUnassigned=True,
      seed=seed,
      pseudoraters=False,
      useStableInitialization=useStableInitialization,
      saveIntermediateState=saveIntermediateState,
      threads=threads,
      minMinorityNetHelpfulRatings=minMinorityNetHelpfulRatings,
      minMinorityNetHelpfulRatio=minMinorityNetHelpfulRatio,
    )

  def get_name(self):
    return "MFExpansionPlusScorer"

  def _get_note_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default note column names to custom names for a specific model."""
    return {
      c.internalNoteInterceptKey: c.expansionPlusNoteInterceptKey,
      c.internalNoteFactor1Key: c.expansionPlusNoteFactor1Key,
      c.internalRatingStatusKey: c.expansionPlusRatingStatusKey,
      c.internalActiveRulesKey: c.expansionPlusInternalActiveRulesKey,
      c.numFinalRoundRatingsKey: c.expansionPlusNumFinalRoundRatingsKey,
      c.lowDiligenceNoteInterceptKey: c.lowDiligenceLegacyNoteInterceptKey,
      c.internalNoteInterceptNoHighVolKey: c.expansionPlusNoteInterceptNoHighVolKey,
      c.internalNoteInterceptNoCorrelatedKey: c.expansionPlusNoteInterceptNoCorrelatedKey,
    }

  def _get_user_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default user column names to custom names for a specific model."""
    return {
      c.internalRaterInterceptKey: c.expansionPlusRaterInterceptKey,
      c.internalRaterFactor1Key: c.expansionPlusRaterFactor1Key,
    }

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      c.expansionPlusNoteInterceptKey,
      c.expansionPlusNoteFactor1Key,
      c.expansionPlusRatingStatusKey,
      c.expansionPlusInternalActiveRulesKey,
      c.expansionPlusNumFinalRoundRatingsKey,
      c.expansionPlusNoteInterceptNoHighVolKey,
      c.expansionPlusNoteInterceptNoCorrelatedKey,
    ]

  def get_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the helpfulnessScores output."""
    return [
      c.raterParticipantIdKey,
      c.expansionPlusRaterInterceptKey,
      c.expansionPlusRaterFactor1Key,
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
      c.crhCrnhRatioDifferenceKey,
      c.meanNoteScoreKey,
      c.raterAgreeRatioKey,
      c.aboveHelpfulnessThresholdKey,
      c.internalFirstRoundRaterInterceptKey,
      c.internalFirstRoundRaterFactor1Key,
    ]
