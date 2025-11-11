from typing import Dict, List, Optional

from . import constants as c
from .mf_base_scorer import MFBaseScorer


_EXPANSION_BOOL = "expansionBool"


class MFExpansionScorer(MFBaseScorer):
  def __init__(
    self,
    seed: Optional[int] = None,
    useStableInitialization: bool = True,
    saveIntermediateState: bool = False,
    threads: int = c.defaultNumThreads,
    firmRejectThreshold: Optional[float] = 0.3,
    minMinorityNetHelpfulRatings: Optional[int] = 4,
    minMinorityNetHelpfulRatio: Optional[float] = 0.05,
  ) -> None:
    """Configure MFExpansionScorer object.

    Args:
      seed: if not None, seed value to ensure deterministic execution
      threads: number of threads to use for intra-op parallelism in pytorch
    """
    super().__init__(
      includedGroups=(c.coreGroups | c.expansionGroups),
      includeUnassigned=True,
      captureThreshold=0.5,
      seed=seed,
      pseudoraters=False,
      useStableInitialization=useStableInitialization,
      saveIntermediateState=saveIntermediateState,
      threads=threads,
      firmRejectThreshold=firmRejectThreshold,
      minMinorityNetHelpfulRatings=minMinorityNetHelpfulRatings,
      minMinorityNetHelpfulRatio=minMinorityNetHelpfulRatio,
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
      c.internalActiveRulesKey: c.expansionInternalActiveRulesKey,
      c.numFinalRoundRatingsKey: c.expansionNumFinalRoundRatingsKey,
      c.lowDiligenceNoteInterceptKey: c.lowDiligenceLegacyNoteInterceptKey,
      c.internalNoteInterceptNoHighVolKey: c.expansionNoteInterceptNoHighVolKey,
      c.internalNoteInterceptNoCorrelatedKey: c.expansionNoteInterceptNoCorrelatedKey,
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
      c.expansionNoteInterceptNoHighVolKey,
      c.expansionNoteInterceptNoCorrelatedKey,
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
      c.internalFirstRoundRaterInterceptKey,
      c.internalFirstRoundRaterFactor1Key,
    ]
