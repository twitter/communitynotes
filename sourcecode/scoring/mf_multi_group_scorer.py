from . import constants as c
from .mf_base_scorer import coalesce_columns
from .mf_group_scorer import MFGroupScorer

import pandas as pd


def coalesce_multi_group_model_scored_notes(scoredNotes: pd.DataFrame) -> pd.DataFrame:
  """Coalesce all multi group modeling columns across note scoring.

  Since each Scorer must have distinct output columns, we use coalescing to run
  multiple instances of MFGroupScorer objects and then condense the results into
  a single set of columns.  This approach works because each note will be scored
  by at most one MFGroupScorer instance.
  """
  for col in [
    c.multiGroupNoteInterceptKey,
    c.multiGroupNoteFactor1Key,
    c.multiGroupRatingStatusKey,
    c.modelingMultiGroupKey,
    c.multiGroupInternalActiveRulesKey,
    c.multiGroupNumFinalRoundRatingsKey,
    c.multiGroupNoteInterceptNoHighVolKey,
  ]:
    scoredNotes = coalesce_columns(scoredNotes, col)

  return scoredNotes


def coalesce_multi_group_model_helpfulness_scores(helpfulnessScores: pd.DataFrame) -> pd.DataFrame:
  """Coalesce all group modeling columns across user scoring.

  Since each Scorer must have distinct output columns, we use coalescing to run
  multiple instances of MFGroupScorer objects and then condense the results into
  a single set of columns.  This approach works because each note will be scored
  by at most one MFGroupScorer instance.
  """
  for col in [c.multiGroupRaterInterceptKey, c.multiGroupRaterFactor1Key, c.modelingMultiGroupKey]:
    helpfulnessScores = coalesce_columns(helpfulnessScores, col)
  return helpfulnessScores


class MFMultiGroupScorer(MFGroupScorer):
  def _init_column_names(self):
    self._groupNoteInterceptKey = f"{c.multiGroupNoteInterceptKey}_{self._groupId}"
    self._groupNoteFactor1Key = f"{c.multiGroupNoteFactor1Key}_{self._groupId}"
    self._groupRatingStatusKey = f"{c.multiGroupRatingStatusKey}_{self._groupId}"
    self._groupInternalActiveRulesKey = f"{c.multiGroupInternalActiveRulesKey}_{self._groupId}"
    self._groupNumFinalRoundRatingsKey = f"{c.multiGroupNumFinalRoundRatingsKey}_{self._groupId}"
    self._groupRaterInterceptKey = f"{c.multiGroupRaterInterceptKey}_{self._groupId}"
    self._groupRaterFactor1Key = f"{c.multiGroupRaterFactor1Key}_{self._groupId}"
    self._modelingGroupKey = f"{c.modelingMultiGroupKey}_{self._groupId}"
    self._groupNoteInterceptNoHighVolKey = (
      f"{c.multiGroupNoteInterceptNoHighVolKey}_{self._groupId}"
    )

  def get_name(self):
    return f"MFMultiGroupScorer_{self._groupId}"
