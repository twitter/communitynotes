from typing import Dict, List, Optional, Tuple

from . import constants as c
from .mf_base_scorer import MFBaseScorer, coalesce_columns

import pandas as pd


def coalesce_topic_models(scoredNotes: pd.DataFrame) -> pd.DataFrame:
  """Coalesce all topic modeling columns across note and user scoring.

  Since each Scorer must have distinct output columns, we use coalescing to run
  multiple instances of MFTopicScorer objects and then condense the results into
  a single set of columns.  This approach works because each note will be scored
  by at most one MFTopicScorer instance.

  Args:
    scoredNotes: scoring output for notes.

  Returns:
    tuple containing coalesced scoring results for notes and users.
  """
  for col in [
    c.topicNoteInterceptKey,
    c.topicNoteFactor1Key,
    c.topicRatingStatusKey,
    c.topicNoteConfidentKey,
    c.noteTopicKey,
    c.topicInternalActiveRulesKey,
    c.topicNumFinalRoundRatingsKey,
    c.topicNoteInterceptNoHighVolKey,
    c.topicNoteInterceptNoCorrelatedKey,
  ]:
    scoredNotes = coalesce_columns(scoredNotes, col)

  return scoredNotes


class MFTopicScorer(MFBaseScorer):
  def __init__(
    self,
    topicName: str,
    seed: Optional[int] = None,
    pseudoraters: Optional[bool] = False,
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
    crhSuperThreshold: float = 0.5,
    crhThresholdNoHighVol: float = 0.37,
    crhThresholdNoCorrelated: float = 0.37,
    lowDiligenceThreshold: float = 0.263,
    factorThreshold: float = 0.5,
    multiplyPenaltyByHarassmentScore: bool = True,
    minimumHarassmentScoreToPenalize: float = 2.0,
    tagConsensusHarassmentHelpfulRatingPenalty: int = 10,
  ) -> None:
    """Configure MFTopicScorer object.

    Notice that each MFTopicScorer defines column names by appending the topicName to
    column prefixes which are constant.  Dynamically defining the column names allows the
    topic scorer to be instantiated multiple times while maintaining the property that
    the columns attached by each scorer remain unique.  Once all scorers have ran, we
    (will) validate that each note was scored by at most one topic scorer and then coalesce
    all of the topic scoring columns and remove the topicName suffix.

    Args:
      topicName: str indicating which topic this scorer instance should filter for.
      seed: if not None, seed value to ensure deterministic execution
      pseudoraters: if True, compute optional pseudorater confidence intervals
    """
    super().__init__(
      includedTopics={topicName},
      seed=seed,
      pseudoraters=pseudoraters,
      useStableInitialization=False,
      saveIntermediateState=saveIntermediateState,
      threads=4,
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
      useReputation=False,
    )
    self._topicName = topicName
    self._topicNoteInterceptKey = f"{c.topicNoteInterceptKey}_{self._topicName}"
    self._topicNoteFactor1Key = f"{c.topicNoteFactor1Key}_{self._topicName}"
    self._topicRatingStatusKey = f"{c.topicRatingStatusKey}_{self._topicName}"
    self._topicInternalActiveRulesKey = f"{c.topicInternalActiveRulesKey}_{self._topicName}"
    self._topicNumFinalRoundRatingsKey = f"{c.topicNumFinalRoundRatingsKey}_{self._topicName}"
    self._noteTopicKey = f"{c.noteTopicKey}_{self._topicName}"
    self._noteTopicConfidentKey = f"{c.topicNoteConfidentKey}_{self._topicName}"
    self._topicNoteInterceptNoHighVolKey = f"{c.topicNoteInterceptNoHighVolKey}_{self._topicName}"
    self._topicNoteInterceptNoCorrelatedKey = (
      f"{c.topicNoteInterceptNoCorrelatedKey}_{self._topicName}"
    )

  def get_name(self):
    return f"MFTopicScorer_{self._topicName}"

  def _get_note_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default note column names to custom names for a specific model."""
    return {
      c.internalNoteInterceptKey: self._topicNoteInterceptKey,
      c.internalNoteFactor1Key: self._topicNoteFactor1Key,
      c.internalRatingStatusKey: self._topicRatingStatusKey,
      c.internalActiveRulesKey: self._topicInternalActiveRulesKey,
      c.numFinalRoundRatingsKey: self._topicNumFinalRoundRatingsKey,
      c.lowDiligenceNoteInterceptKey: c.lowDiligenceLegacyNoteInterceptKey,
      c.internalNoteInterceptNoHighVolKey: self._topicNoteInterceptNoHighVolKey,
      c.internalNoteInterceptNoCorrelatedKey: self._topicNoteInterceptNoCorrelatedKey,
    }

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      self._topicNoteInterceptKey,
      self._topicNoteFactor1Key,
      self._topicRatingStatusKey,
      self._noteTopicKey,
      self._noteTopicConfidentKey,
      self._topicInternalActiveRulesKey,
      self._topicNumFinalRoundRatingsKey,
      self._topicNoteInterceptNoHighVolKey,
      self._topicNoteInterceptNoCorrelatedKey,
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
      c.internalRaterInterceptKey,
      c.internalRaterFactor1Key,
      c.raterParticipantIdKey,
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
    """Add noteTopicKey to notes output.

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
    # Set the modelingGroupKey column in each output
    noteScores[self._noteTopicKey] = self._topicName
    # Calculate total counts of positive and negative factor ratings
    scoredNotes = noteScores[~noteScores[c.internalNoteInterceptKey].isna()][[c.noteIdKey]]
    posFactorRaters = userScores[userScores[c.internalRaterFactor1Key] >= 0][
      [c.raterParticipantIdKey]
    ]
    posFactorRatings = (
      ratings[[c.noteIdKey, c.raterParticipantIdKey]].merge(scoredNotes).merge(posFactorRaters)
    )
    posFactorCounts = (
      posFactorRatings.groupby(c.noteIdKey)
      .count()
      .reset_index(drop=False)
      .rename(columns={c.raterParticipantIdKey: "posRatingTotal"})
    )
    negFactorRaters = userScores[userScores[c.internalRaterFactor1Key] < 0][
      [c.raterParticipantIdKey]
    ]
    negFactorRatings = (
      ratings[[c.noteIdKey, c.raterParticipantIdKey]].merge(scoredNotes).merge(negFactorRaters)
    )
    negFactorCounts = (
      negFactorRatings.groupby(c.noteIdKey)
      .count()
      .reset_index(drop=False)
      .rename(columns={c.raterParticipantIdKey: "negRatingTotal"})
    )
    # Set scoring confidence bit
    posFactorCounts = posFactorCounts[posFactorCounts["posRatingTotal"] > 4][[c.noteIdKey]]
    negFactorCounts = negFactorCounts[negFactorCounts["negRatingTotal"] > 4][[c.noteIdKey]]
    confidentNotes = posFactorCounts.merge(negFactorCounts)
    confidentNotes[self._noteTopicConfidentKey] = True
    noteScores = noteScores.merge(
      confidentNotes, how="left", unsafeAllowed=[self._noteTopicConfidentKey, c.defaultIndexKey]
    )
    noteScores = noteScores.fillna({self._noteTopicConfidentKey: False})
    return noteScores, userScores
