from typing import Dict, List, Optional

from . import constants as c
from .mf_base_scorer import MFBaseScorer


class MFCoreWithTopicsScorer(MFBaseScorer):
  def __init__(
    self,
    seed: Optional[int] = None,
    pseudoraters: Optional[bool] = False,
    useStableInitialization: bool = True,
    saveIntermediateState: bool = False,
    threads: int = c.defaultNumThreads,
    firmRejectThreshold: Optional[float] = None,
    minMinorityNetHelpfulRatings: Optional[int] = 4,
    minMinorityNetHelpfulRatio: Optional[float] = 0.05,
  ) -> None:
    """Configure MFCoreWithTopicsScorer object.

    Args:
      seed: if not None, seed value to ensure deterministic execution
      pseudoraters: if True, compute optional pseudorater confidence intervals
      threads: number of threads to use for intra-op parallelism in pytorch
    """
    super().__init__(
      includedGroups=c.coreGroups,
      includeUnassigned=True,
      captureThreshold=0.5,
      seed=seed,
      pseudoraters=pseudoraters,
      useStableInitialization=useStableInitialization,
      saveIntermediateState=saveIntermediateState,
      threads=threads,
      firmRejectThreshold=firmRejectThreshold,
      minMinorityNetHelpfulRatings=minMinorityNetHelpfulRatings,
      minMinorityNetHelpfulRatio=minMinorityNetHelpfulRatio,
    )

  def get_name(self):
    return "MFCoreWithTopicsScorer"

  def _get_note_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default note column names to custom names for a specific model."""
    return {
      c.internalNoteInterceptKey: c.coreWithTopicsNoteInterceptKey,
      c.internalNoteFactor1Key: c.coreWithTopicsNoteFactor1Key,
      c.internalRatingStatusKey: c.coreWithTopicsRatingStatusKey,
      c.internalActiveRulesKey: c.coreWithTopicsActiveRulesKey,
      c.noteInterceptMinKey: c.coreWithTopicsNoteInterceptMinKey,
      c.noteInterceptMaxKey: c.coreWithTopicsNoteInterceptMaxKey,
      c.numFinalRoundRatingsKey: c.coreWithTopicsNumFinalRoundRatingsKey,
      c.lowDiligenceNoteInterceptKey: c.lowDiligenceLegacyNoteInterceptKey,
      c.internalNoteInterceptNoHighVolKey: c.coreWithTopicsNoteInterceptNoHighVolKey,
      c.internalNoteInterceptNoCorrelatedKey: c.coreWithTopicsNoteInterceptNoCorrelatedKey,
    }

  def _get_user_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default user column names to custom names for a specific model."""
    return {
      c.internalRaterInterceptKey: c.coreWithTopicsRaterInterceptKey,
      c.internalRaterFactor1Key: c.coreWithTopicsRaterFactor1Key,
    }

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      c.coreWithTopicsNoteInterceptKey,
      c.coreWithTopicsNoteFactor1Key,
      c.coreWithTopicsRatingStatusKey,
      c.coreWithTopicsActiveRulesKey,
      c.coreWithTopicsNoteInterceptMinKey,
      c.coreWithTopicsNoteInterceptMaxKey,
      c.coreWithTopicsNumFinalRoundRatingsKey,
      c.coreWithTopicsNoteInterceptNoHighVolKey,
      c.coreWithTopicsNoteInterceptNoCorrelatedKey,
    ]

  def get_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the helpfulnessScores output."""
    return [
      c.raterParticipantIdKey,
      c.coreWithTopicsRaterInterceptKey,
      c.coreWithTopicsRaterFactor1Key,
    ]

  def _get_dropped_note_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from scoredNotes and auxiliaryNoteInfo."""
    return super()._get_dropped_note_cols() + (
      [
        c.ratingWeightKey,
        c.activeFilterTagsKey,
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

  def get_auxiliary_note_info_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the auxiliaryNoteInfo output."""
    return []
