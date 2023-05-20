from typing import Dict, List, Optional

from . import constants as c
from .mf_base_scorer import MFBaseScorer


class MFExpansionScorer(MFBaseScorer):
  def __init__(
    self,
    seed: Optional[int] = None,
  ) -> None:
    """Configure MFExpansionScorer object.

    Args:
      seed: if not None, seed value to ensure deterministic execution
    """
    super().__init__(seed, pseudoraters=False)

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
