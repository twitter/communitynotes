from typing import Dict, List, Optional, Tuple

from . import constants as c
from .gaussian_scorer import GaussianScorer
from .mf_topic_scorer import MFTopicScorer

import pandas as pd


class GaussianTopicScorer(GaussianScorer):
  def __init__(
    self,
    topicName: str,
    seed: Optional[int] = None,
    saveIntermediateState: bool = False,
    minMeanNoteScore: float = 0.05,
    crhThreshold: float = 0.40,
    crnhThresholdIntercept: float = -0.05,
    crnhThresholdNoteFactorMultiplier: float = -0.8,
    crnhThresholdNMIntercept: float = -0.15,
    crnhThresholdUCBIntercept: float = -0.5,
    crhSuperThreshold: float = 0.5,
    crhThresholdNoHighVol: float = 0.37,
    crhThresholdNoCorrelated: float = 0.37,
    lowDiligenceThreshold: float = 0.263,
    factorThreshold: float = 0.5,
    tagFilterPercentile: int = 95,
    incorrectFilterThreshold: float = 2.5,
    numConfidenceRatings: int = 4,
    userFactorLambda=None,
    noteFactorLambda=None,
    userInterceptLambda=None,
    noteInterceptLambda=None,
    globalInterceptLambda=None,
    diamondLambda=None,
    normalizedLossHyperparameters=None,
    useGlobalIntercept: bool = True,
    crhParams: c.GaussianParams = c.gaussianCrhParams,
    crnhParams: c.GaussianParams = c.gaussianCrnhParams,
  ) -> None:
    """Configure GaussianTopicScorer object.

    Notice that each GaussianTopicScorer defines column names by appending the topicName to
    column prefixes which are constant.  Dynamically defining the column names allows the
    topic scorer to be instantiated multiple times while maintaining the property that
    the columns attached by each scorer remain unique.  Once all scorers have ran, we
    validate that each note was scored by at most one topic scorer and then coalesce
    all of the topic scoring columns and remove the topicName suffix.

    Args:
      topicName: str indicating which topic this scorer instance should filter for.
      seed: if not None, seed value to ensure deterministic execution
    """
    super().__init__(
      includedTopics={topicName},
      excludeTopics=False,
      includedGroups=set(),
      includeUnassigned=False,
      captureThreshold=None,
      seed=seed,
      saveIntermediateState=saveIntermediateState,
      threads=4,
      minMeanNoteScore=minMeanNoteScore,
      crhThreshold=crhThreshold,
      crnhThresholdIntercept=crnhThresholdIntercept,
      crnhThresholdNoteFactorMultiplier=crnhThresholdNoteFactorMultiplier,
      crnhThresholdNMIntercept=crnhThresholdNMIntercept,
      crnhThresholdUCBIntercept=crnhThresholdUCBIntercept,
      crhSuperThreshold=crhSuperThreshold,
      crhThresholdNoHighVol=crhThresholdNoHighVol,
      crhThresholdNoCorrelated=crhThresholdNoCorrelated,
      lowDiligenceThreshold=lowDiligenceThreshold,
      factorThreshold=factorThreshold,
      useReputation=False,
      tagFilterPercentile=tagFilterPercentile,
      incorrectFilterThreshold=incorrectFilterThreshold,
      crhParams=crhParams,
      crnhParams=crnhParams,
    )
    # Store MF parameters for constructing the MFTopicScorer used in prescoring.
    self._mfTopicScorerArgs = dict(
      topicName=topicName,
      seed=seed,
      userFactorLambda=userFactorLambda,
      noteFactorLambda=noteFactorLambda,
      userInterceptLambda=userInterceptLambda,
      noteInterceptLambda=noteInterceptLambda,
      globalInterceptLambda=globalInterceptLambda,
      diamondLambda=diamondLambda,
      normalizedLossHyperparameters=normalizedLossHyperparameters,
      useGlobalIntercept=useGlobalIntercept,
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
    self._numConfidenceRatings = numConfidenceRatings

  def get_prescoring_name(self):
    return f"MFTopicScorer_{self._topicName}"

  def get_name(self):
    return f"GaussianTopicScorer_{self._topicName}"

  def _prescore_notes_and_users(
    self,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    userEnrollmentRaw: pd.DataFrame,
  ) -> Tuple[pd.DataFrame, pd.DataFrame, c.PrescoringMetaScorerOutput]:
    mfScorer = MFTopicScorer(**self._mfTopicScorerArgs)
    return mfScorer._prescore_notes_and_users(ratings, noteStatusHistory, userEnrollmentRaw)

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
    return super()._get_dropped_note_cols()

  def _get_dropped_user_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from helpfulnessScores output.

    Note: GaussianScorer's helpfulnessScores only contains raterParticipantIdKey and
    internalRaterFactor1Key. The parent GaussianScorer._get_dropped_user_cols() already
    drops internalRaterFactor1Key, so we only need to additionally drop raterParticipantIdKey.
    """
    return super()._get_dropped_user_cols() + [
      c.raterParticipantIdKey,
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
    # Set the noteTopicKey column in each output
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
    posFactorCounts = posFactorCounts[
      posFactorCounts["posRatingTotal"] > self._numConfidenceRatings
    ][[c.noteIdKey]]
    negFactorCounts = negFactorCounts[
      negFactorCounts["negRatingTotal"] > self._numConfidenceRatings
    ][[c.noteIdKey]]
    confidentNotes = posFactorCounts.merge(negFactorCounts)
    confidentNotes[self._noteTopicConfidentKey] = True
    noteScores = noteScores.merge(
      confidentNotes, how="left", unsafeAllowed=[self._noteTopicConfidentKey, c.defaultIndexKey]
    )
    noteScores = noteScores.fillna({self._noteTopicConfidentKey: False})
    return noteScores, userScores
