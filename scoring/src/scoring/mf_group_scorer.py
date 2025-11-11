from typing import Dict, List, Optional, Set, Tuple

from . import constants as c
from .mf_base_scorer import MFBaseScorer, coalesce_columns

import pandas as pd


# Number of consecutive MFGroupScorer objects we expect to instantiate,
# this does not include the NMR group scorer
groupScorerCount = 14

# Group ID assigned to trial scoring algorithm
trialScoringGroup = 14

# Group ID assigned to NMR group scorer
nmrScoringGroup = 33


# Mapping of how many threads to assign to each group scorer
groupScorerParalleism = {
  # Group model 13 is larger and benefits from more threads.
  # Others can default to 4.
  13: 8
}


def coalesce_group_model_scored_notes(scoredNotes: pd.DataFrame) -> pd.DataFrame:
  """Coalesce all group modeling columns across note scoring.

  Since each Scorer must have distinct output columns, we use coalescing to run
  multiple instances of MFGroupScorer objects and then condense the results into
  a single set of columns.  This approach works because each note will be scored
  by at most one MFGroupScorer instance.
  """
  for col in [
    c.groupNoteInterceptKey,
    c.groupNoteFactor1Key,
    c.groupRatingStatusKey,
    c.modelingGroupKey,
    c.groupInternalActiveRulesKey,
    c.groupNumFinalRoundRatingsKey,
    c.groupNoteInterceptNoHighVolKey,
    c.groupNoteInterceptNoCorrelatedKey,
  ]:
    scoredNotes = coalesce_columns(scoredNotes, col)

  return scoredNotes


def coalesce_group_model_helpfulness_scores(helpfulnessScores: pd.DataFrame) -> pd.DataFrame:
  """Coalesce all group modeling columns across user scoring.

  Since each Scorer must have distinct output columns, we use coalescing to run
  multiple instances of MFGroupScorer objects and then condense the results into
  a single set of columns.  This approach works because each note will be scored
  by at most one MFGroupScorer instance.
  """
  for col in [c.groupRaterInterceptKey, c.groupRaterFactor1Key, c.modelingGroupKey]:
    helpfulnessScores = coalesce_columns(helpfulnessScores, col)
  return helpfulnessScores


class MFGroupScorer(MFBaseScorer):
  def __init__(
    self,
    includedGroups: Set[int],
    groupId: int,
    strictInclusion: bool = False,
    seed: Optional[int] = None,
    groupThreshold: float = 0.8,
    saveIntermediateState: bool = False,
    userFactorLambda=None,
    noteFactorLambda=None,
    userInterceptLambda=None,
    noteInterceptLambda=None,
    globalInterceptLambda=None,
    diamondLambda=None,
    normalizedLossHyperparameters=None,
    maxFirstMFTrainError: float = 0.16,
    maxFinalMFTrainError: float = 0.09,
    minMeanNoteScore: float = 0.05,
    crhThreshold: float = 0.40,
    crnhThresholdIntercept: float = -0.05,
    crnhThresholdNoteFactorMultiplier: float = -0.8,
    crnhThresholdNMIntercept: float = -0.15,
    crhSuperThreshold: Optional[float] = 0.5,
    crhThresholdNoHighVol: float = 0.37,
    crhThresholdNoCorrelated: float = 0.37,
    lowDiligenceThreshold: float = 0.263,
    factorThreshold: float = 0.5,
    multiplyPenaltyByHarassmentScore: bool = True,
    minimumHarassmentScoreToPenalize: float = 2.0,
    tagConsensusHarassmentHelpfulRatingPenalty: int = 10,
    tagFilterPercentile: int = 95,
    incorrectFilterThreshold: float = 2.5,
    threads: int = 4,
    minMinorityNetHelpfulRatings: Optional[int] = 4,
    minMinorityNetHelpfulRatio: Optional[float] = 0.05,
  ) -> None:
    """Configure MFGroupScorer object.

    Notice that each MFGroupScorer defines column names by appending the groupNumber to
    column prefixes which are constant.  Dynamically defining the column names allows the
    group scorer to be instantiated multiple times while maintaining the property that
    the columns attached by each scorer remain unique.  Once all scorers have ran, we
    (will) validate that each note was scored by at most one group scorer and then coalesce
    all of the group scoring columns and remove the groupNumber suffix.

    Args:
      groupNumber: int indicating which group this scorer instance should filter for.
      seed: if not None, seed value to ensure deterministic execution
      pseudoraters: if True, compute optional pseudorater confidence intervals
      groupThreshold: float indicating what fraction of ratings must be from within a group
        for the model to be active
    """
    super().__init__(
      includedGroups=includedGroups,
      strictInclusion=strictInclusion,
      includeUnassigned=False,
      captureThreshold=groupThreshold,
      seed=seed,
      pseudoraters=False,
      useStableInitialization=False,
      saveIntermediateState=saveIntermediateState,
      threads=threads,
      userFactorLambda=userFactorLambda,
      noteFactorLambda=noteFactorLambda,
      userInterceptLambda=userInterceptLambda,
      noteInterceptLambda=noteInterceptLambda,
      globalInterceptLambda=globalInterceptLambda,
      diamondLambda=diamondLambda,
      normalizedLossHyperparameters=normalizedLossHyperparameters,
      maxFirstMFTrainError=maxFirstMFTrainError,
      maxFinalMFTrainError=maxFinalMFTrainError,
      minMeanNoteScore=minMeanNoteScore,
      crhThreshold=crhThreshold,
      crnhThresholdIntercept=crnhThresholdIntercept,
      crnhThresholdNoteFactorMultiplier=crnhThresholdNoteFactorMultiplier,
      crnhThresholdNMIntercept=crnhThresholdNMIntercept,
      crhSuperThreshold=crhSuperThreshold,
      crhThresholdNoHighVol=crhThresholdNoHighVol,
      crhThresholdNoCorrelated=crhThresholdNoCorrelated,
      lowDiligenceThreshold=lowDiligenceThreshold,
      factorThreshold=factorThreshold,
      multiplyPenaltyByHarassmentScore=multiplyPenaltyByHarassmentScore,
      minimumHarassmentScoreToPenalize=minimumHarassmentScoreToPenalize,
      tagConsensusHarassmentHelpfulRatingPenalty=tagConsensusHarassmentHelpfulRatingPenalty,
      tagFilterPercentile=tagFilterPercentile,
      incorrectFilterThreshold=incorrectFilterThreshold,
      minMinorityNetHelpfulRatings=minMinorityNetHelpfulRatings,
      minMinorityNetHelpfulRatio=minMinorityNetHelpfulRatio,
    )
    assert groupId > 0, "groupNumber must be positive.  0 is reserved for unassigned."
    self._groupId = groupId
    self._init_column_names()

  def _init_column_names(self):
    """Initialize column names based on prefixes and groupId."""
    self._groupNoteInterceptKey = f"{c.groupNoteInterceptKey}_{self._groupId}"
    self._groupNoteFactor1Key = f"{c.groupNoteFactor1Key}_{self._groupId}"
    self._groupRatingStatusKey = f"{c.groupRatingStatusKey}_{self._groupId}"
    self._groupInternalActiveRulesKey = f"{c.groupInternalActiveRulesKey}_{self._groupId}"
    self._groupNumFinalRoundRatingsKey = f"{c.groupNumFinalRoundRatingsKey}_{self._groupId}"
    self._groupRaterInterceptKey = f"{c.groupRaterInterceptKey}_{self._groupId}"
    self._groupRaterFactor1Key = f"{c.groupRaterFactor1Key}_{self._groupId}"
    self._modelingGroupKey = f"{c.modelingGroupKey}_{self._groupId}"
    self._groupNoteInterceptNoHighVolKey = f"{c.groupNoteInterceptNoHighVolKey}_{self._groupId}"
    self._groupNoteInterceptNoCorrelatedKey = (
      f"{c.groupNoteInterceptNoCorrelatedKey}_{self._groupId}"
    )

  def get_name(self):
    return f"MFGroupScorer_{self._groupId}"

  def _get_note_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default note column names to custom names for a specific model."""
    return {
      c.internalNoteInterceptKey: self._groupNoteInterceptKey,
      c.internalNoteFactor1Key: self._groupNoteFactor1Key,
      c.internalRatingStatusKey: self._groupRatingStatusKey,
      c.internalActiveRulesKey: self._groupInternalActiveRulesKey,
      c.numFinalRoundRatingsKey: self._groupNumFinalRoundRatingsKey,
      c.lowDiligenceNoteInterceptKey: c.lowDiligenceLegacyNoteInterceptKey,
      c.internalNoteInterceptNoHighVolKey: self._groupNoteInterceptNoHighVolKey,
      c.internalNoteInterceptNoCorrelatedKey: self._groupNoteInterceptNoCorrelatedKey,
    }

  def _get_user_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default user column names to custom names for a specific model."""
    return {
      c.internalRaterInterceptKey: self._groupRaterInterceptKey,
      c.internalRaterFactor1Key: self._groupRaterFactor1Key,
    }

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      self._groupNoteInterceptKey,
      self._groupNoteFactor1Key,
      self._groupRatingStatusKey,
      self._groupInternalActiveRulesKey,
      self._modelingGroupKey,
      self._groupNumFinalRoundRatingsKey,
      self._groupNoteInterceptNoHighVolKey,
      self._groupNoteInterceptNoCorrelatedKey,
    ]

  def get_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the helpfulnessScores output."""
    return [
      c.raterParticipantIdKey,
      self._groupRaterInterceptKey,
      self._groupRaterFactor1Key,
      self._modelingGroupKey,
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

  def _postprocess_output(
    self,
    noteScores: pd.DataFrame,
    userScores: pd.DataFrame,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    userEnrollment: pd.DataFrame,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter noteScores and userScores and add column containing modeling group.

    Enforce the requirements that:
      - Notes scored by this group model have >=80% of ratings from the modeling group.
      - Notes scored by this group model were authored by a member of the modeling group.
      - Users assigned a factor / intercept are in the modeling group.

    Args:
      noteScores: note outputs from scoring
      userScores: user outputs from scoring
      ratings (pd.DataFrame): preprocessed ratings
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
      userEnrollment (pd.DataFrame): one row per user specifying enrollment properties

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame]:
        noteScores: filtered and updated note scoring output
        userScores: filtered and updated user scoring output
    """
    noteScores, userScores = super()._postprocess_output(
      noteScores, userScores, ratings, noteStatusHistory, userEnrollment
    )
    # Note that even though ratings were restricted to the modeling group, users outside of
    # the modeling group may still have authored a note which was rated and may consequently
    # appear in the userScores.  Accordingly, we drop any user which was outside of the
    # modeling group from the user scores.
    userScores = userScores.merge(
      userEnrollment[[c.participantIdKey, c.modelingGroupKey]].rename(
        columns={c.participantIdKey: c.raterParticipantIdKey}
      ),
      how="left",
    )
    userScores = userScores[userScores[c.modelingGroupKey].isin(self._includedGroups)]
    userScores = userScores.drop(columns=c.modelingGroupKey)
    # Set the modelingGroupKey column in each output
    noteScores[self._modelingGroupKey] = self._groupId
    userScores[self._modelingGroupKey] = self._groupId
    return noteScores, userScores
