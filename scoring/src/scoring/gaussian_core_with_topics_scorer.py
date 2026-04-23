from typing import Dict, List, Optional

from . import constants as c
from .gaussian_scorer import GaussianScorer


class GaussianCoreWithTopicsScorer(GaussianScorer):
  """Gaussian convolution scorer restricted to core groups (with topics variant).

  This scorer inherits all Gaussian scoring logic but filters ratings to only
  include raters from coreGroups and unassigned raters, mirroring the population
  used by MFCoreWithTopicsScorer.
  """

  def __init__(
    self,
    seed: Optional[int] = None,
    threads: int = c.defaultNumThreads,
    saveIntermediateState: bool = False,
  ) -> None:
    """Configure GaussianCoreWithTopicsScorer object.

    Args:
      seed: if not None, seed value to ensure deterministic execution
      threads: number of threads to use for intra-op parallelism in pytorch
      saveIntermediateState: if True, save intermediate state for debugging
    """
    super().__init__(
      includedGroups=c.coverageGroups,
      excludeTopics=False,
      includeUnassigned=True,
      captureThreshold=0.5,
      seed=seed,
      threads=threads,
      saveIntermediateState=saveIntermediateState,
    )

  def get_name(self):
    return "GaussianCoreWithTopicsScorer"

  def _get_note_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default note column names to custom names for a specific model."""
    return {
      c.internalNoteInterceptKey: c.gaussianCoreWithTopicsNoteInterceptKey,
      c.internalNoteFactor1Key: c.gaussianCoreWithTopicsNoteFactor1Key,
      c.internalActiveRulesKey: c.gaussianCoreWithTopicsActiveRulesKey,
      c.numFinalRoundRatingsKey: c.gaussianCoreWithTopicsNumFinalRoundRatingsKey,
      c.internalNoteInterceptNoHighVolKey: c.gaussianCoreWithTopicsNoteInterceptNoHighVolKey,
      c.internalNoteInterceptNoCorrelatedKey: c.gaussianCoreWithTopicsNoteInterceptNoCorrelatedKey,
      c.internalNoteInterceptPopulationSampledKey: c.gaussianCoreWithTopicsNoteInterceptPopulationSampledKey,
      c.lowDiligenceNoteInterceptKey: c.lowDiligenceLegacyNoteInterceptKey,
      c.internalRatingStatusKey: c.gaussianCoreWithTopicsRatingStatusKey,
    }

  def _get_user_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default user column names to custom names for a specific model."""
    return {}

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      c.gaussianCoreWithTopicsNoteInterceptKey,
      c.gaussianCoreWithTopicsNoteFactor1Key,
      c.gaussianCoreWithTopicsRatingStatusKey,
      c.gaussianCoreWithTopicsActiveRulesKey,
      c.gaussianCoreWithTopicsNumFinalRoundRatingsKey,
      c.gaussianCoreWithTopicsNoteInterceptNoHighVolKey,
      c.gaussianCoreWithTopicsNoteInterceptNoCorrelatedKey,
      c.gaussianCoreWithTopicsNoteInterceptPopulationSampledKey,
    ]

  def get_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the helpfulnessScores output."""
    return [
      c.raterParticipantIdKey,
    ]

  def get_auxiliary_note_info_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the auxiliaryNoteInfo output."""
    return [
      c.noteIdKey,
    ]
