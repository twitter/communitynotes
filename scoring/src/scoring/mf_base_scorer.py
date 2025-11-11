import gc
import logging
from typing import Dict, List, Optional, Set, Tuple

from . import (
  constants as c,
  helpfulness_scores,
  note_ratings,
  process_data,
  tag_consensus,
  tag_filter,
)
from .incorrect_filter import get_user_incorrect_ratio
from .matrix_factorization.matrix_factorization import MatrixFactorization
from .matrix_factorization.pseudo_raters import PseudoRatersRunner
from .pandas_utils import get_df_fingerprint, keep_columns
from .reputation_matrix_factorization.diligence_model import (
  fit_low_diligence_model_final,
  fit_low_diligence_model_prescoring,
)
from .scorer import EmptyRatingException, Scorer

import numpy as np
import pandas as pd
import torch


logger = logging.getLogger("birdwatch.mf_base_scorer")
logger.setLevel(logging.INFO)


def coalesce_columns(df: pd.DataFrame, columnPrefix: str) -> pd.DataFrame:
  """Condense all columns beginning with columnPrefix into a single column.

  With each row there must be at most one column with a non-NaN value in the set of
  columns beginning with columnPrefix.  If a non-NaN value is present that will
  become the value in the condensed column, otherwise the value will be NaN.  After
  column values are condensed the original (prefixed) columns will be dropped.

  Args:
    df: DataFrame containing columns to condense
    collumnPrefix: Prefix used to detect columns to coalesce, and the name for
      the output column.

  Returns:
    DataFrame with all columns prefixed by columnPrefix dropped and replaced by
    a single column named columnPrefix

  Raises:
    AssertionError if multiple columns prefixed by columnPrefix have non-NaN values
    for any row.
  """
  # Identify columns to coalesce
  columns = [col for col in df.columns if col.startswith(f"{columnPrefix}_")]
  if not columns:
    return df
  # Validate that at most one column is set, and store which rows have a column set
  rowResults = np.invert(df[columns].isna()).sum(axis=1)
  assert all(rowResults <= 1), "each row should only be in one modeling group"

  # Coalesce results
  def _get_value(row):
    idx = row.first_valid_index()
    return row[idx] if idx is not None else np.nan

  coalesced = df[columns].apply(_get_value, axis=1)
  # Drop old columns and replace with new
  df = df.drop(columns=columns)
  df[columnPrefix] = coalesced
  return df


def get_ratings_for_stable_init(
  ratingsForTraining: pd.DataFrame,
  userEnrollmentRaw: pd.DataFrame,
  modelingGroupToInitializeForStability: int,
  minPercentRatingsFromModelingGroup: float = 0.75,
  minNumRatingsToIncludeInStableInitialization: int = 5,
) -> pd.DataFrame:
  """Returns a subset of ratings to use for training an initial matrix factorization.
  Args:
      ratingsForTraining (pd.DataFrame)
      userEnrollmentRaw (pd.DataFrame)
      modelingGroupToInitializeForStability: modeling group for round 0 ratings
      minPercentRatingsFromModelingGroup: notes must have this fraction of ratings from the modeling group
      minNumRatingsToIncludeInStableInitialization: required from modeling group

  Returns:
      DF containing a subset of ratings
  """
  ratingsForTrainingWithModelingGroup = ratingsForTraining.merge(
    userEnrollmentRaw[[c.participantIdKey, c.modelingGroupKey]],
    left_on=c.raterParticipantIdKey,
    right_on=c.participantIdKey,
  )

  ratingsForTrainingWithModelingGroup[c.ratingFromInitialModelingGroupKey] = (
    ratingsForTrainingWithModelingGroup[c.modelingGroupKey] == modelingGroupToInitializeForStability
  )

  # Only include ratings from the modeling group
  ratingsForStableInitialization = ratingsForTrainingWithModelingGroup[
    ratingsForTrainingWithModelingGroup[c.ratingFromInitialModelingGroupKey]
  ]

  # Only include notes that have received at least 75% of their ratings from the modeling group (and 5 total)
  ratingsForTrainingWithModelingGroup[c.ratingCountKey] = 1
  noteStatsByRatedModelingGroup = (
    ratingsForTrainingWithModelingGroup[
      [c.noteIdKey, c.ratingFromInitialModelingGroupKey, c.ratingCountKey]
    ]
    .groupby(c.noteIdKey)
    .sum()
    .reset_index()
  )
  noteStatsByRatedModelingGroup[c.percentFromInitialModelingGroupKey] = (
    noteStatsByRatedModelingGroup[c.ratingFromInitialModelingGroupKey]
    / noteStatsByRatedModelingGroup[c.ratingCountKey]
  )
  noteStatsByRatedModelingGroup[
    c.percentFromInitialModelingGroupKey
  ] = noteStatsByRatedModelingGroup[c.percentFromInitialModelingGroupKey].fillna(0)
  notesRatedMostlyByInitialModelingGroup = noteStatsByRatedModelingGroup[
    (
      noteStatsByRatedModelingGroup[c.percentFromInitialModelingGroupKey]
      >= minPercentRatingsFromModelingGroup
    )
    & (
      noteStatsByRatedModelingGroup[c.ratingCountKey]
      >= minNumRatingsToIncludeInStableInitialization
    )
  ]
  ratingsForStableInitialization = ratingsForStableInitialization.merge(
    notesRatedMostlyByInitialModelingGroup[[c.noteIdKey]], on=c.noteIdKey
  )

  assert (
    len(ratingsForStableInitialization) > 0
  ), "No ratings from stable initialization modeling group."

  return ratingsForStableInitialization


# TODO: Consider merging compute_scored_notes, is_crh, is_crnh, filter_ratings,
# compute_general_helpfulness_scores and filter_ratings_by_helpfulness_scores into this class.
# These functions are only called by this class, and merging them in will allow accessing
# member state and simplify the callsites.
class MFBaseScorer(Scorer):
  """Runs MatrixFactorization to determine raw note scores and ultimately note status."""

  def __init__(
    self,
    includedTopics: Set[str] = set(),
    excludeTopics: bool = False,
    includedGroups: Set[int] = set(),
    includeUnassigned: bool = False,
    strictInclusion: bool = False,
    captureThreshold: Optional[float] = None,
    seed: Optional[int] = None,
    pseudoraters: Optional[bool] = True,
    minNumRatingsPerRater: int = 10,
    minNumRatersPerNote: int = 5,
    minRatingsNeeded: int = 5,
    minMeanNoteScore: float = 0.05,
    minCRHVsCRNHRatio: float = 0.00,
    minRaterAgreeRatio: float = 0.66,
    crhThreshold: float = 0.40,
    crnhThresholdIntercept: float = -0.05,
    crnhThresholdNoteFactorMultiplier: float = -0.8,
    crnhThresholdNMIntercept: float = -0.15,
    crnhThresholdUCBIntercept: float = -0.04,
    crhSuperThreshold: Optional[float] = 0.5,
    crhThresholdNoHighVol: float = 0.37,
    crhThresholdNoCorrelated: float = 0.37,
    lowDiligenceThreshold: float = 0.263,
    factorThreshold: float = 0.5,
    inertiaDelta: float = 0.01,
    useStableInitialization: bool = True,
    saveIntermediateState: bool = False,
    threads: int = c.defaultNumThreads,
    maxFirstMFTrainError: float = 0.16,
    maxFinalMFTrainError: float = 0.09,
    userFactorLambda=None,
    noteFactorLambda=None,
    userInterceptLambda=None,
    noteInterceptLambda=None,
    globalInterceptLambda=None,
    diamondLambda=None,
    normalizedLossHyperparameters=None,
    multiplyPenaltyByHarassmentScore: bool = True,
    minimumHarassmentScoreToPenalize: float = 2.0,
    tagConsensusHarassmentHelpfulRatingPenalty: int = 10,
    useReputation: bool = True,
    tagFilterPercentile: int = 95,
    incorrectFilterThreshold: float = 2.5,
    firmRejectThreshold: Optional[float] = None,
    minMinorityNetHelpfulRatings: Optional[int] = None,
    minMinorityNetHelpfulRatio: Optional[float] = None,
    populationSampledRatingPerNoteLossRatio: Optional[float] = 10.0,
  ):
    """Configure MatrixFactorizationScorer object.

    Args:
      includedGroups: if set, filter ratings and results based on includedGroups
      includedTopics: if set, filter ratings based on includedTopics
      excludedTopics: if set, filter ratings based on excludedTopics
      seed: if not None, seed value to ensure deterministic execution
      pseudoraters: if True, compute optional pseudorater confidence intervals
      minNumRatingsPerRater: Minimum number of ratings which a rater must produce to be
        included in scoring.  Raters with fewer ratings are removed.
      minNumRatersPerNote: Minimum number of ratings which a note must have to be included
        in scoring.  Notes with fewer ratings are removed.
      minRatingsNeeded: Minimum number of ratings for a note to achieve status.
      minMeanNoteScore: Raters included in the second MF round must achieve this minimum
        average intercept for any notes written.
      minCRHVsCRNHRatio: Minimum crhCrnhRatioDifference for raters included in the second
        MF round. crhCrnhRatioDifference is a weighted measure comparing how often an author
        produces CRH / CRNH notes.  See author_helpfulness for more info.
      minRaterAgreeRatio: Raters in the second MF round must exceed this minimum standard for how
        often a rater must predict the eventual outcome when rating before a note is assigned status.
      crhThreshold: Minimum intercept for most notes to achieve CRH status.
      crnhThresholdIntercept: Maximum intercept for most notes to achieve CRNH status.
      crnhThresholdNoteFactorMultiplier: Scaling factor making controlling the relationship between
        CRNH threshold and note intercept.  Note that this constant is set negative so that notes with
        larger (magnitude) factors must have proportionally lower intercepts to become CRNH.
      crnhThresholdNMIntercept: Maximum intercept for notes which do not claim a tweet is misleading
        to achieve CRNH status.
      crnhThresholdUCBIntercept: Maximum UCB of the intercept (determined with pseudoraters) for
        notes to achieve CRNH status.
      crhSuperThreshold: Minimum intercept for notes which have consistent and common patterns of
        repeated reason tags in not-helpful ratings to achieve CRH status.
      inertiaDelta: Minimum amount which a note that has achieve CRH status must drop below the
        applicable threshold to lose CRH status.
      useStableInitialization: whether to use a specific modeling group of users to stably initialize
      threads: number of threads to use for intra-op parallelism in pytorch
      maxFirstMFTrainError: maximum error allowed for the first MF training process
      maxFinalMFTrainError: maximum error allowed for the final MF training process
      populationSampledRatingPerNoteLossRatio: optional override for ratingPerNoteLossRatio when computing the population sampled intercept
    """
    super().__init__(
      includedTopics=includedTopics,
      excludeTopics=excludeTopics,
      includedGroups=includedGroups,
      strictInclusion=strictInclusion,
      includeUnassigned=includeUnassigned,
      captureThreshold=captureThreshold,
      seed=seed,
      threads=threads,
    )
    self._pseudoraters = pseudoraters
    self._minNumRatingsPerRater = minNumRatingsPerRater
    self._minNumRatersPerNote = minNumRatersPerNote
    self._minRatingsNeeded = minRatingsNeeded
    self._minMeanNoteScore = minMeanNoteScore
    self._minCRHVsCRNHRatio = minCRHVsCRNHRatio
    self._minRaterAgreeRatio = minRaterAgreeRatio
    self._crhThreshold = crhThreshold
    self._crnhThresholdIntercept = crnhThresholdIntercept
    self._crnhThresholdNoteFactorMultiplier = crnhThresholdNoteFactorMultiplier
    self._crnhThresholdNMIntercept = crnhThresholdNMIntercept
    self._crnhThresholdUCBIntercept = crnhThresholdUCBIntercept
    self._crhSuperThreshold = crhSuperThreshold
    self._crhThresholdNoHighVol = crhThresholdNoHighVol
    self._crhThresholdNoCorrelated = crhThresholdNoCorrelated
    self._inertiaDelta = inertiaDelta
    self._modelingGroupToInitializeForStability = 13 if useStableInitialization else None
    self._saveIntermediateState = saveIntermediateState
    self._maxFirstMFTrainError = maxFirstMFTrainError
    self._maxFinalMFTrainError = maxFinalMFTrainError
    self._lowDiligenceThreshold = lowDiligenceThreshold
    self._factorThreshold = factorThreshold
    self.multiplyPenaltyByHarassmentScore = multiplyPenaltyByHarassmentScore
    self.minimumHarassmentScoreToPenalize = minimumHarassmentScoreToPenalize
    self.tagConsensusHarassmentHelpfulRatingPenalty = tagConsensusHarassmentHelpfulRatingPenalty
    self._useReputation = useReputation
    self._tagFilterPercentile = tagFilterPercentile
    self._incorrectFilterThreshold = incorrectFilterThreshold
    self._firmRejectThreshold = firmRejectThreshold
    self._minMinorityNetHelpfulRatings = minMinorityNetHelpfulRatings
    self._minMinorityNetHelpfulRatio = minMinorityNetHelpfulRatio
    self._populationSampledRatingPerNoteLossRatio = populationSampledRatingPerNoteLossRatio
    mfArgs = dict(
      [
        pair
        for pair in [
          ("userFactorLambda", userFactorLambda) if userFactorLambda is not None else None,
          ("noteFactorLambda", noteFactorLambda) if noteFactorLambda is not None else None,
          ("userInterceptLambda", userInterceptLambda) if userInterceptLambda is not None else None,
          ("noteInterceptLambda", noteInterceptLambda) if noteInterceptLambda is not None else None,
          ("globalInterceptLambda", globalInterceptLambda)
          if globalInterceptLambda is not None
          else None,
          ("diamondLambda", diamondLambda) if diamondLambda is not None else None,
          ("normalizedLossHyperparameters", normalizedLossHyperparameters)
          if normalizedLossHyperparameters is not None
          else None,
          ("initLearningRate", 0.02 if normalizedLossHyperparameters is not None else 0.2),
          ("noInitLearningRate", 0.02 if normalizedLossHyperparameters is not None else 1.0),
          ("seed", seed) if seed is not None else None,
        ]
        if pair is not None
      ]
    )
    self._mfRanker = MatrixFactorization(**mfArgs)

  def assert_train_error_is_below_threshold(self, ratings, maxTrainError) -> None:
    """
    If we are running a non-test run (number of ratings is above threshold),
    assert that the final training error for the MF ranker is below the threshold.
    """
    testRun = (
      ratings[c.noteIdKey].nunique() < c.minNumNotesForProdData if ratings is not None else False
    )
    if not testRun:
      finalTrainError = self._mfRanker.get_final_train_error()
      if finalTrainError is None:
        raise ValueError("Final train error is None")
      else:
        if finalTrainError > maxTrainError:
          raise ValueError(f"Train error ({finalTrainError}) is above threshold ({maxTrainError})")

  def get_crh_threshold(self) -> float:
    """Return CRH threshold for general scoring logic."""
    return self._crhThreshold

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      c.internalNoteInterceptKey,
      c.internalNoteFactor1Key,
      c.internalRatingStatusKey,
      c.internalActiveRulesKey,
      c.activeFilterTagsKey,
      c.noteInterceptMaxKey,
      c.noteInterceptMinKey,
      c.numFinalRoundRatingsKey,
      c.internalNoteInterceptNoHighVolKey,
      c.internalNoteInterceptNoCorrelatedKey,
    ]

  def get_internal_scored_notes_cols(self) -> List[str]:
    """Returns a list of internal columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      c.internalNoteInterceptKey,
      c.internalNoteFactor1Key,
      c.internalRatingStatusKey,
      c.internalActiveRulesKey,
      c.activeFilterTagsKey,
      c.noteInterceptMaxKey,
      c.noteInterceptMinKey,
      c.numFinalRoundRatingsKey,
      c.lowDiligenceNoteInterceptKey,
      c.lowDiligenceNoteFactor1Key,
    ]

  def get_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the helpfulnessScores output."""
    return [
      c.raterParticipantIdKey,
      c.internalRaterInterceptKey,
      c.internalRaterFactor1Key,
      c.crhCrnhRatioDifferenceKey,
      c.meanNoteScoreKey,
      c.raterAgreeRatioKey,
      c.aboveHelpfulnessThresholdKey,
      c.internalFirstRoundRaterInterceptKey,
      c.internalFirstRoundRaterFactor1Key,
    ]

  def get_internal_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of internal columns which should be present in the helpfulnessScores output."""
    return [
      c.raterParticipantIdKey,
      c.internalRaterInterceptKey,
      c.internalRaterFactor1Key,
      c.crhCrnhRatioDifferenceKey,
      c.meanNoteScoreKey,
      c.raterAgreeRatioKey,
      c.aboveHelpfulnessThresholdKey,
      c.lowDiligenceRaterInterceptKey,
      c.lowDiligenceRaterFactor1Key,
      c.lowDiligenceRaterReputationKey,
      c.internalFirstRoundRaterInterceptKey,
      c.internalFirstRoundRaterFactor1Key,
    ]

  def get_auxiliary_note_info_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the auxiliaryNoteInfo output."""
    return [
      c.noteIdKey,
      c.ratingWeightKey,
    ] + (
      c.notHelpfulTagsAdjustedColumns
      + c.notHelpfulTagsAdjustedRatioColumns
      + c.incorrectFilterColumns
    )

  def _get_dropped_note_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from scoredNotes and auxiliaryNoteInfo."""
    dropped_cols = [
      c.currentlyRatedHelpfulBoolKey,
      c.currentlyRatedNotHelpfulBoolKey,
      c.awaitingMoreRatingsBoolKey,
      c.currentLabelKey,
      c.classificationKey,
      c.numRatingsKey,
      c.noteAuthorParticipantIdKey,
    ]

    # only drop population sampled column if it's not mapped to an output column
    note_col_mapping = self._get_note_col_mapping()
    if c.internalNoteInterceptPopulationSampledKey not in note_col_mapping:
      dropped_cols.extend(
        [
          c.internalNoteInterceptPopulationSampledKey,
          c.negFactorPopulationSampledRatingCountKey,
          c.posFactorPopulationSampledRatingCountKey,
        ]
      )

    return (
      dropped_cols
      + c.helpfulTagsTSVOrder
      + c.notHelpfulTagsTSVOrder
      + c.noteParameterUncertaintyTSVAuxColumns
    )

  def _get_dropped_user_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from helpfulnessScores output."""
    return []

  def _prepare_data_for_scoring(self, ratings: pd.DataFrame, final: bool = False) -> pd.DataFrame:
    """Prepare data for scoring. This includes filtering out notes and raters which do not meet
    minimum rating counts, and may be overridden by subclasses to add additional filtering.
    """
    if final:
      return process_data.filter_ratings(
        ratings, minNumRatingsPerRater=0, minNumRatersPerNote=self._minNumRatersPerNote
      )
    else:
      return process_data.filter_ratings(
        ratings,
        minNumRatingsPerRater=self._minNumRatingsPerRater,
        minNumRatersPerNote=self._minNumRatersPerNote,
      )

  def _run_regular_matrix_factorization(self, ratingsForTraining: pd.DataFrame):
    """Train a matrix factorization model on the ratingsForTraining data.

    Args:
        ratingsForTraining (pd.DataFrame)

    Returns:
        noteParams (pd.DataFrame)
        raterParams (pd.DataFrame)
        globalIntercept (float)
    """
    return self._mfRanker.run_mf(ratingsForTraining, run_name=f"{self.get_name()}/regular_mf")

  def _run_stable_matrix_factorization(
    self,
    ratingsForTraining: pd.DataFrame,
    userEnrollmentRaw: pd.DataFrame,
  ):
    """Train a matrix factorization model on the ratingsForTraining data.
    Due to stability issues when trained on the entire dataset with no initialization, this is done in
    two steps:
    1. Train a model on the subset of the data with modeling group 13 (stable initialization).
    2. Train a model on the entire dataset, initializing with the results from step 1.
    Without this initialization, the factors for some subsets of the data with low crossover between raters can
    flip relative to each other from run to run.

    Args:
        ratingsForTraining (pd.DataFrame)
        userEnrollmentRaw (pd.DataFrame)

    Returns:
        noteParams (pd.DataFrame)
        raterParams (pd.DataFrame)
        globalIntercept (float)
    """
    if self._modelingGroupToInitializeForStability is None:
      return self._run_regular_matrix_factorization(ratingsForTraining)

    with self.time_block("Prepare data for stable initialization"):
      ratingsForStableInitialization = get_ratings_for_stable_init(
        ratingsForTraining,
        userEnrollmentRaw,
        self._modelingGroupToInitializeForStability,
      )

    with self.time_block("MF on stable-initialization subset"):
      initializationMF = self._mfRanker.get_new_mf_with_same_args()
      noteParamsInit, raterParamsInit, globalInterceptInit = initializationMF.run_mf(
        ratingsForStableInitialization, run_name=f"{self.get_name()}/stable_init"
      )

    with self.time_block("First full MF (initializated with stable-initialization)"):
      modelResult = self._mfRanker.run_mf(
        ratingsForTraining,
        noteInit=noteParamsInit,
        userInit=raterParamsInit,
        globalInterceptInit=globalInterceptInit,
        run_name=f"{self.get_name()}/full_init",
      )
    return modelResult

  def compute_tag_thresholds_for_percentile(
    self, scoredNotes, raterParams, ratings
  ) -> Dict[str, float]:
    with c.time_block(f"{self.get_name()}: Compute tag thresholds for percentiles"):
      # Compute tag aggregates (in the same way as is done in final scoring in note_ratings.compute_scored_notes)
      tagAggregates = tag_filter.get_note_tag_aggregates(ratings, scoredNotes, raterParams)
      assert len(tagAggregates) == len(
        scoredNotes
      ), "There should be one aggregate per scored note."
      scoredNotes = tagAggregates.merge(scoredNotes, on=c.noteIdKey, how="outer")

      # Compute percentile thresholds for each tag
      crhNotes = scoredNotes[scoredNotes[c.currentlyRatedHelpfulBoolKey]][[c.noteIdKey]]
      crhStats = scoredNotes.merge(crhNotes, on=c.noteIdKey, how="inner")
      thresholds = tag_filter.get_tag_thresholds(crhStats, self._tagFilterPercentile)
    return thresholds

  def _prescore_notes_and_users(
    self,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    userEnrollmentRaw: pd.DataFrame,
  ) -> Tuple[pd.DataFrame, pd.DataFrame, c.PrescoringMetaScorerOutput]:
    """
    Fit initial matrix factorization model(s) on the ratings data in order to generate
    initial note and rater parameters (and rater helpfulness scores) that are passed to
    the final scoring step. The final scoring step will be able to run faster by
    using the rater helpfulness scores computed here, and also intializing its parameters
    with these parameters.


    Args:
        ratings (pd.DataFrame)
        noteStatusHistory (pd.DataFrame)
        userEnrollmentRaw (pd.DataFrame)

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
          noteParamsUnfiltered (pd.DataFrame)
          raterParamsUnfiltered (pd.DataFrame)
          helpfulnessScores (pd.DataFrame)
    """
    if self._seed is not None:
      logger.info(f"seeding with {self._seed}")
      torch.manual_seed(self._seed)

    # Removes ratings where either (1) the note did not receive enough ratings, or
    # (2) the rater did not rate enough notes.
    logger.info(
      f"ratings summary {self.get_name()}: {get_df_fingerprint(ratings, [c.noteIdKey, c.raterParticipantIdKey])}"
    )

    with self.time_block("Prepare ratings"):
      ratingsForTraining = self._prepare_data_for_scoring(
        ratings[
          [
            c.noteIdKey,
            c.raterParticipantIdKey,
            c.helpfulNumKey,
            c.createdAtMillisKey,
            c.helpfulnessLevelKey,
            c.notHelpfulIncorrectTagKey,
            c.notHelpfulIrrelevantSourcesTagKey,
            c.notHelpfulSourcesMissingOrUnreliableTagKey,
            c.notHelpfulSpamHarassmentOrAbuseTagKey,
            c.notHelpfulOtherTagKey,
          ]
        ]
      )
    if len(ratingsForTraining) == 0:
      # This is only expected to occur for MFTopicScorer_MessiRonaldo in --recent runs
      assert (
        self.get_name() == "MFTopicScorer_MessiRonaldo"
      ), f"Unexpected scorer: {self.get_name()}"
      raise EmptyRatingException
    logger.info(
      f"ratingsForTraining summary {self.get_name()}: {get_df_fingerprint(ratingsForTraining, [c.noteIdKey, c.raterParticipantIdKey])}"
    )
    logger.info(
      f"noteStatusHistory summary {self.get_name()}: {get_df_fingerprint(noteStatusHistory, [c.noteIdKey])}"
    )
    logger.info(
      f"userEnrollmentRaw summary {self.get_name()}: {get_df_fingerprint(userEnrollmentRaw, [c.participantIdKey])}"
    )
    if self._saveIntermediateState:
      self.ratingsForTraining = ratingsForTraining

    # TODO: Save parameters from this first run in note_model_output next time we add extra fields to model output TSV.
    with self.time_block("First MF/stable init"):
      (
        noteParamsUnfiltered,
        raterParamsUnfiltered,
        globalBias,
      ) = self._run_stable_matrix_factorization(
        ratingsForTraining[[c.noteIdKey, c.raterParticipantIdKey, c.helpfulNumKey]],
        userEnrollmentRaw[[c.participantIdKey, c.modelingGroupKey]],
      )
    if self._saveIntermediateState:
      self.noteParamsUnfiltered = noteParamsUnfiltered
      self.raterParamsUnfiltered = raterParamsUnfiltered
      self.globalBias = globalBias
    self.assert_train_error_is_below_threshold(
      ratingsForTraining[[c.noteIdKey]], self._maxFirstMFTrainError
    )

    # If reputation is disabled, generate final intercepts, factors and note status
    # based on the first round scoring results.  Disabling reputation can be desirable
    # in situations where the overall volume of ratings is lower (e.g. topic models).
    if not self._useReputation:
      assert "MFTopicScorer" in self.get_name(), f"Unexpected scorer: {self.get_name()}"
      logger.info(f"Skipping rep-filtering in prescoring for {self.get_name()}")
      helpfulnessScores = raterParamsUnfiltered[[c.raterParticipantIdKey]]
      helpfulnessScores[
        [
          c.crhCrnhRatioDifferenceKey,
          c.meanNoteScoreKey,
          c.raterAgreeRatioKey,
          c.aboveHelpfulnessThresholdKey,
          c.successfulRatingNotHelpfulCount,
          c.successfulRatingHelpfulCount,
          c.unsuccessfulRatingNotHelpfulCount,
          c.unsuccessfulRatingHelpfulCount,
          c.totalHelpfulHarassmentRatingsPenaltyKey,
          c.raterAgreeRatioWithHarassmentAbusePenaltyKey,
        ]
      ] = np.nan
      noteParams = noteParamsUnfiltered
      raterParams = raterParamsUnfiltered
      # TODO: delete after we run prescoring diligence properly
      # diligenceGlobalIntercept = None
      finalRoundRatings = ratingsForTraining
      harassmentAbuseNoteParams = noteParamsUnfiltered[[c.noteIdKey]]
      harassmentAbuseNoteParams[[c.harassmentNoteInterceptKey, c.harassmentNoteFactor1Key]] = np.nan
    else:
      assert "MFTopicScorer" not in self.get_name(), f"Unexpected scorer: {self.get_name()}"
      logger.info(f"Performing rep-filtering for {self.get_name()}")
      # Get a dataframe of scored notes based on the algorithm results above
      with self.time_block("Compute scored notes"):
        scoredNotes = note_ratings.compute_scored_notes(
          ratings[
            [
              c.noteIdKey,
              c.raterParticipantIdKey,
              c.helpfulnessLevelKey,
              c.createdAtMillisKey,
            ]
            + c.notHelpfulTagsTSVOrder
            + c.helpfulTagsTSVOrder
            + [c.ratingSourceBucketedKey]
          ],
          keep_columns(
            noteParamsUnfiltered,
            [
              c.noteIdKey,
              c.internalNoteInterceptKey,
              c.internalNoteFactor1Key,
              c.internalNoteInterceptPopulationSampledKey,
            ]
            + c.noteParameterUncertaintyTSVColumns,
          ),
          raterParamsUnfiltered[
            [
              c.raterParticipantIdKey,
              c.internalRaterFactor1Key,
            ]
          ],
          noteStatusHistory[
            [
              c.noteIdKey,
              c.createdAtMillisKey,
              c.noteAuthorParticipantIdKey,
              c.classificationKey,
              c.currentLabelKey,
              c.lockedStatusKey,
            ]
          ],
          minRatingsNeeded=self._minRatingsNeeded,
          crhThreshold=self._crhThreshold,
          crnhThresholdIntercept=self._crnhThresholdIntercept,
          crnhThresholdNoteFactorMultiplier=self._crnhThresholdNoteFactorMultiplier,
          crnhThresholdNMIntercept=self._crnhThresholdNMIntercept,
          crnhThresholdUCBIntercept=self._crnhThresholdUCBIntercept,
          crhSuperThreshold=self._crhSuperThreshold,
          inertiaDelta=self._inertiaDelta,
          incorrectFilterThreshold=self._incorrectFilterThreshold,
          tagFilterThresholds=None,
          finalRound=False,
          firmRejectThreshold=self._firmRejectThreshold,
        )
      if self._saveIntermediateState:
        self.prescoringScoredNotes = scoredNotes

      # Determine "valid" ratings
      with self.time_block("Compute valid ratings"):
        validRatings = note_ratings.get_valid_ratings(
          ratings[
            [
              c.noteIdKey,
              c.raterParticipantIdKey,
              c.helpfulNumKey,
              c.createdAtMillisKey,
            ]
          ],
          noteStatusHistory[
            [
              c.noteIdKey,
              c.createdAtMillisKey,
              c.timestampMillisOfNoteMostRecentNonNMRLabelKey,
            ]
          ],
          scoredNotes[
            [
              c.noteIdKey,
              c.currentlyRatedHelpfulBoolKey,
              c.currentlyRatedNotHelpfulBoolKey,
              c.awaitingMoreRatingsBoolKey,
            ]
          ],
        )
      if self._saveIntermediateState:
        self.validRatings = validRatings

      if len(validRatings) == 0:
        # This is only expected for MFGroupScorer_33 on --recent runs.
        assert self.get_name() == "MFGroupScorer_33", f"Unexpected scorer: {self.get_name()}"
        raise EmptyRatingException

      # Assigns contributor (author & rater) helpfulness bit based on (1) performance
      # authoring and reviewing previous and current notes.
      with self.time_block("Helpfulness scores pre-harassment "):
        helpfulnessScoresPreHarassmentFilter = (
          helpfulness_scores.compute_general_helpfulness_scores(
            scoredNotes[
              [
                c.noteAuthorParticipantIdKey,
                c.currentlyRatedHelpfulBoolKey,
                c.currentlyRatedNotHelpfulBoolKey,
                c.internalNoteInterceptKey,
              ]
            ],
            validRatings[
              [
                c.raterParticipantIdKey,
                c.ratingAgreesWithNoteStatusKey,
                c.ratingCountKey,
                c.successfulRatingNotHelpfulCount,
                c.successfulRatingHelpfulCount,
                c.unsuccessfulRatingNotHelpfulCount,
                c.unsuccessfulRatingHelpfulCount,
              ]
            ],
            self._minMeanNoteScore,
            self._minCRHVsCRNHRatio,
            self._minRaterAgreeRatio,
            ratingsForTraining[[c.noteIdKey, c.raterParticipantIdKey, c.helpfulNumKey]],
          )
        )
      if self._saveIntermediateState:
        self.prescoringHelpfulnessScores = helpfulnessScoresPreHarassmentFilter

      # Filters ratings matrix to include only rows (ratings) where the rater was
      # considered helpful.
      with self.time_block("Filtering by helpfulness score"):
        ratingsHelpfulnessScoreFilteredPreHarassmentFilter = (
          helpfulness_scores.filter_ratings_by_helpfulness_scores(
            ratingsForTraining[
              [
                c.noteIdKey,
                c.raterParticipantIdKey,
                c.notHelpfulSpamHarassmentOrAbuseTagKey,
                c.createdAtMillisKey,
                c.helpfulnessLevelKey,
                c.notHelpfulOtherTagKey,
              ]
            ],
            helpfulnessScoresPreHarassmentFilter,
          )
        )

      if self._saveIntermediateState:
        self.ratingsHelpfulnessScoreFilteredPreHarassmentFilter = (
          ratingsHelpfulnessScoreFilteredPreHarassmentFilter
        )

      with self.time_block("Harassment tag consensus"):
        harassmentAbuseNoteParams, _, _ = tag_consensus.train_tag_model(
          ratings=ratingsHelpfulnessScoreFilteredPreHarassmentFilter,
          tag=c.notHelpfulSpamHarassmentOrAbuseTagKey,
          helpfulModelNoteParams=noteParamsUnfiltered[
            [c.noteIdKey, c.internalNoteInterceptKey, c.internalNoteFactor1Key]
          ],
          helpfulModelRaterParams=raterParamsUnfiltered[
            [
              c.raterParticipantIdKey,
              c.internalRaterInterceptKey,
              c.internalRaterFactor1Key,
            ]
          ],
          name="harassment",
        )
      if not self._saveIntermediateState:
        del ratingsHelpfulnessScoreFilteredPreHarassmentFilter
        gc.collect()

      # Assigns contributor (author & rater) helpfulness bit based on (1) performance
      # authoring and reviewing previous and current notes, and (2) including an extra
      # penalty for rating a harassment/abuse note as helpful.
      with self.time_block("Helpfulness scores post-harassment"):
        helpfulnessScores = helpfulness_scores.compute_general_helpfulness_scores(
          scoredNotes[
            [
              c.noteAuthorParticipantIdKey,
              c.currentlyRatedHelpfulBoolKey,
              c.currentlyRatedNotHelpfulBoolKey,
              c.internalNoteInterceptKey,
            ]
          ],
          validRatings[
            [
              c.raterParticipantIdKey,
              c.ratingAgreesWithNoteStatusKey,
              c.ratingCountKey,
              c.successfulRatingNotHelpfulCount,
              c.successfulRatingHelpfulCount,
              c.unsuccessfulRatingNotHelpfulCount,
              c.unsuccessfulRatingHelpfulCount,
            ]
          ],
          self._minMeanNoteScore,
          self._minCRHVsCRNHRatio,
          self._minRaterAgreeRatio,
          ratings=ratingsForTraining[[c.noteIdKey, c.raterParticipantIdKey, c.helpfulNumKey]],
          tagConsensusHarassmentAbuseNotes=harassmentAbuseNoteParams,
          tagConsensusHarassmentHelpfulRatingPenalty=self.tagConsensusHarassmentHelpfulRatingPenalty,
          multiplyPenaltyByHarassmentScore=self.multiplyPenaltyByHarassmentScore,
          minimumHarassmentScoreToPenalize=self.minimumHarassmentScoreToPenalize,
        )
      if not self._saveIntermediateState:
        del validRatings
        gc.collect()
      if self._saveIntermediateState:
        self.helpfulnessScores = helpfulnessScores

      ## One extra final round!
      # Filter ratings based on prev helpfulness scores
      with c.time_block("Final round MF"):
        finalRoundRatings = helpfulness_scores.filter_ratings_by_helpfulness_scores(
          ratingsForTraining[
            [
              c.noteIdKey,
              c.raterParticipantIdKey,
              c.helpfulNumKey,
              c.notHelpfulIncorrectTagKey,
              c.notHelpfulSourcesMissingOrUnreliableTagKey,
              c.notHelpfulIrrelevantSourcesTagKey,
            ]
          ],
          helpfulnessScores[[c.raterParticipantIdKey, c.aboveHelpfulnessThresholdKey]],
        )
        noteParams, raterParams, globalBias = self._mfRanker.run_mf(
          ratings=finalRoundRatings[[c.noteIdKey, c.raterParticipantIdKey, c.helpfulNumKey]],
          noteInit=noteParamsUnfiltered[
            [c.noteIdKey, c.internalNoteInterceptKey, c.internalNoteFactor1Key]
          ],
          userInit=raterParamsUnfiltered[
            [
              c.raterParticipantIdKey,
              c.internalRaterInterceptKey,
              c.internalRaterFactor1Key,
            ]
          ],
          run_name=f"{self.get_name()}/final_round_mf",
        )

    # Run Diligence MF Prescoring, based on the final MF
    with self.time_block("Low Diligence MF"):
      # Initialize diligence rater factors with final round helpful MF rater factor
      raterParamsDiligenceInit = raterParams[
        [c.raterParticipantIdKey, c.internalRaterFactor1Key]
      ].rename({c.internalRaterFactor1Key: c.lowDiligenceRaterFactor1Key}, axis=1)
      logger.info(
        f"In {self.get_name()} prescoring, about to call diligence with {len(finalRoundRatings)} final round ratings."
      )
      (
        diligenceNoteParams,
        diligenceRaterParams,
        diligenceGlobalIntercept,
      ) = fit_low_diligence_model_prescoring(
        finalRoundRatings[
          [
            c.noteIdKey,
            c.raterParticipantIdKey,
            c.notHelpfulIncorrectTagKey,
            c.notHelpfulSourcesMissingOrUnreliableTagKey,
            c.notHelpfulIrrelevantSourcesTagKey,
          ]
        ],
        raterInitStateDiligence=raterParamsDiligenceInit,
      )
      noteParams = noteParams.merge(diligenceNoteParams, on=c.noteIdKey)

      noteParams = noteParams.merge(
        harassmentAbuseNoteParams[
          [c.noteIdKey, c.harassmentNoteInterceptKey, c.harassmentNoteFactor1Key]
        ],
        on=c.noteIdKey,
        how="left",
      )
      raterParams = raterParams.merge(diligenceRaterParams, on=c.raterParticipantIdKey)

    # Compute scored notes -- currently not returned; only used for downstream computation.
    scoredNotes = note_ratings.compute_scored_notes(
      ratings[
        [
          c.noteIdKey,
          c.raterParticipantIdKey,
          c.helpfulnessLevelKey,
          c.createdAtMillisKey,
        ]
        + c.notHelpfulTagsTSVOrder
        + c.helpfulTagsTSVOrder
        + [c.ratingSourceBucketedKey]
      ],
      keep_columns(
        noteParamsUnfiltered,
        [
          c.noteIdKey,
          c.internalNoteInterceptKey,
          c.internalNoteFactor1Key,
          c.internalNoteInterceptPopulationSampledKey,
        ]
        + c.noteParameterUncertaintyTSVColumns,
      ),
      raterParamsUnfiltered[
        [
          c.raterParticipantIdKey,
          c.internalRaterFactor1Key,
        ]
      ],
      noteStatusHistory[
        [
          c.noteIdKey,
          c.createdAtMillisKey,
          c.noteAuthorParticipantIdKey,
          c.classificationKey,
          c.currentLabelKey,
          c.lockedStatusKey,
        ]
      ],
      minRatingsNeeded=self._minRatingsNeeded,
      crhThreshold=self._crhThreshold,
      crnhThresholdIntercept=self._crnhThresholdIntercept,
      crnhThresholdNoteFactorMultiplier=self._crnhThresholdNoteFactorMultiplier,
      crnhThresholdNMIntercept=self._crnhThresholdNMIntercept,
      crnhThresholdUCBIntercept=self._crnhThresholdUCBIntercept,
      crhSuperThreshold=self._crhSuperThreshold,
      inertiaDelta=self._inertiaDelta,
      tagFilterThresholds=None,
      incorrectFilterThreshold=self._incorrectFilterThreshold,
      finalRound=False,
      factorThreshold=self._factorThreshold,
      firmRejectThreshold=self._firmRejectThreshold,
    )

    # Compute meta output
    metaOutput = c.PrescoringMetaScorerOutput(
      globalIntercept=globalBias,
      lowDiligenceGlobalIntercept=diligenceGlobalIntercept,
      tagFilteringThresholds=self.compute_tag_thresholds_for_percentile(
        scoredNotes=noteParams[[c.noteIdKey, c.internalNoteFactor1Key]].merge(
          scoredNotes[[c.noteIdKey, c.currentlyRatedHelpfulBoolKey]],
          on=c.noteIdKey,
          suffixes=("", "_dup"),
        ),
        raterParams=raterParams[[c.raterParticipantIdKey, c.internalRaterFactor1Key]],
        ratings=ratings[
          [
            c.noteIdKey,
            c.raterParticipantIdKey,
          ]
          + c.notHelpfulTagsTSVOrder
        ],
      ),
      finalRoundNumRatings=len(finalRoundRatings),
      finalRoundNumNotes=finalRoundRatings[c.noteIdKey].nunique(),
      finalRoundNumUsers=finalRoundRatings[c.raterParticipantIdKey].nunique(),
    )

    # Compute user incorrect tag aggregates
    userIncorrectTagUsageDf = get_user_incorrect_ratio(
      ratings[
        [
          c.noteIdKey,
          c.raterParticipantIdKey,
        ]
        + c.notHelpfulTagsTSVOrder
      ]
    )

    raterModelOutput = (
      raterParams.merge(
        helpfulnessScores[
          [
            c.raterParticipantIdKey,
            c.crhCrnhRatioDifferenceKey,
            c.meanNoteScoreKey,
            c.raterAgreeRatioKey,
            c.aboveHelpfulnessThresholdKey,
            c.successfulRatingHelpfulCount,
            c.successfulRatingNotHelpfulCount,
            c.unsuccessfulRatingHelpfulCount,
            c.unsuccessfulRatingNotHelpfulCount,
            c.totalHelpfulHarassmentRatingsPenaltyKey,
            c.raterAgreeRatioWithHarassmentAbusePenaltyKey,
          ]
        ],
        on=c.raterParticipantIdKey,
        how="outer",
      )
      .merge(
        userIncorrectTagUsageDf,
        on=c.raterParticipantIdKey,
        how="left",
        unsafeAllowed={c.totalRatingsMadeByRaterKey, c.incorrectTagRatingsMadeByRaterKey},
      )
      .merge(
        raterParamsUnfiltered[
          [
            c.raterParticipantIdKey,
            c.internalRaterInterceptKey,
            c.internalRaterFactor1Key,
          ]
        ].rename(
          columns={
            c.internalRaterInterceptKey: c.internalFirstRoundRaterInterceptKey,
            c.internalRaterFactor1Key: c.internalFirstRoundRaterFactor1Key,
          }
        ),
        on=c.raterParticipantIdKey,
        how="left",
      )
    )

    noteModelOutput = noteParams
    # Returning should remove references to these, but manually trigger GC just to reclaim
    # resources as soon as possible.
    del ratings
    del ratingsForTraining
    del finalRoundRatings
    gc.collect()
    return noteModelOutput, raterModelOutput, metaOutput

  def _score_notes_and_users(
    self,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    prescoringNoteModelOutput: pd.DataFrame,
    prescoringRaterModelOutput: pd.DataFrame,
    prescoringMetaScorerOutput: c.PrescoringMetaScorerOutput,
    flipFactorsForIdentification: bool = False,
    noteScoresNoHighVol: Optional[pd.DataFrame] = None,
    noteScoresNoCorrelated: Optional[pd.DataFrame] = None,
    noteScoresPopulationSampled: Optional[pd.DataFrame] = None,
    ratingPerNoteLossRatio: Optional[float] = None,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the "final" matrix factorization scoring algorithm.
    Accepts prescoring's output as its input, as well as the new ratings and note status history.

    See links below for more info:
      https://twitter.github.io/communitynotes/ranking-notes/
      https://twitter.github.io/communitynotes/contributor-scores/.

    Args:
      ratings (pd.DataFrame): preprocessed ratings
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
      prescoringNoteModelOutput (pd.DataFrame): note parameters.
      prescoringRaterModelOutput (pd.DataFrame): contains both rater parameters and helpfulnessScores.
      ratingPerNoteLossRatio (Optional[float]): optional override for ratingPerNoteLossRatio for MF run
    Returns:
      Tuple[pd.DataFrame, pd.DataFrame]:
        noteScores pd.DataFrame: one row per note contained note scores and parameters.
        userScores pd.DataFrame: one row per user containing a column for each helpfulness score.
    """
    if self._seed is not None:
      logger.info(f"seeding with {self._seed}")
      torch.manual_seed(self._seed)

    # Removes ratings where either the note did not receive enough ratings
    with self.time_block("Prepare ratings"):
      ratingsForTraining = self._prepare_data_for_scoring(ratings, final=True)
    if self._saveIntermediateState:
      self.ratingsForTraining = ratingsForTraining

    # Filter raters with no rater parameters in this scorer
    ratersWithParams = prescoringRaterModelOutput.loc[
      (
        (~pd.isna(prescoringRaterModelOutput[c.internalRaterInterceptKey]))
        & (~pd.isna(prescoringRaterModelOutput[c.internalRaterInterceptKey]))
      ),
      [c.raterParticipantIdKey],
    ]
    ratingsForTraining = ratingsForTraining.merge(
      ratersWithParams, how="inner", on=c.raterParticipantIdKey
    )

    # Filters ratings matrix to include only rows (ratings) where the rater was
    # considered helpful.
    if not self._useReputation:
      assert (
        "Topic" in self.get_name()
      ), f"Unexpected scorer has reputation filtering disabled: {self.get_name()}"
      logger.info(f"Skipping rep-filtering in 2nd phase for {self.get_name()}")
      finalRoundRatings = ratingsForTraining
    else:
      finalRoundRatings = helpfulness_scores.filter_ratings_by_helpfulness_scores(
        ratingsForTraining, prescoringRaterModelOutput
      )
      if self._saveIntermediateState:
        self.finalRoundRatings = finalRoundRatings

    assert (
      prescoringMetaScorerOutput.finalRoundNumNotes is not None
    ), "Missing final round num notes"
    assert (
      prescoringMetaScorerOutput.finalRoundNumRatings is not None
    ), "Missing final round num ratings"
    assert (
      prescoringMetaScorerOutput.finalRoundNumUsers is not None
    ), "Missing final round num users"

    if len(finalRoundRatings) == 0:
      return pd.DataFrame(), pd.DataFrame()

    # Re-runs matrix factorization using only ratings given by helpful raters.
    with self.time_block("Final helpfulness-filtered MF"):
      noteParams, raterParams, globalBias = self._mfRanker.run_mf(
        ratings=finalRoundRatings,
        noteInit=prescoringNoteModelOutput,
        userInit=prescoringRaterModelOutput,
        globalInterceptInit=prescoringMetaScorerOutput.globalIntercept,
        freezeRaterParameters=True,
        freezeGlobalParameters=True,
        ratingPerNoteLossRatio=(
          ratingPerNoteLossRatio
          if ratingPerNoteLossRatio is not None
          else (
            prescoringMetaScorerOutput.finalRoundNumRatings
            / prescoringMetaScorerOutput.finalRoundNumNotes
          )
        ),
        flipFactorsForIdentification=flipFactorsForIdentification,
        run_name=f"{self.get_name()}/final_helpfulness_filtered_mf",
      )

    if self._saveIntermediateState:
      self.noteParams = noteParams
      self.raterParams = raterParams
      self.globalBias = globalBias
      self.finalRoundRatings = finalRoundRatings
    # self.assert_train_error_is_below_threshold(finalRoundRatings, self._maxFinalMFTrainError)

    # Add pseudo-raters with the most extreme parameters and re-score notes, to estimate
    #  upper and lower confidence bounds on note parameters.
    if self._pseudoraters:
      with self.time_block("Pseudoraters"):
        noteParams = PseudoRatersRunner(
          finalRoundRatings, noteParams, raterParams, globalBias, self._mfRanker
        ).compute_note_parameter_confidence_bounds_with_pseudo_raters()
        if self._saveIntermediateState:
          self.prePseudoratersNoteParams = self.noteParams
          self.noteParams = noteParams
    else:
      for col in c.noteParameterUncertaintyTSVColumns:
        noteParams[col] = np.nan

    # Add low diligence intercepts.
    with self.time_block("Low Diligence Reputation Model"):
      logger.info(
        f"In {self.get_name()} final scoring, about to call diligence with {len(finalRoundRatings)} final round ratings."
      )
      assert (
        prescoringMetaScorerOutput.lowDiligenceGlobalIntercept is not None
      ), "Missing low diligence global intercept"
      diligenceNoteParams, diligenceRaterParams = fit_low_diligence_model_final(
        finalRoundRatings,
        noteInitStateDiligence=prescoringNoteModelOutput,
        raterInitStateDiligence=prescoringRaterModelOutput,
        globalInterceptDiligence=prescoringMetaScorerOutput.lowDiligenceGlobalIntercept,
        ratingsPerNoteLossRatio=prescoringMetaScorerOutput.finalRoundNumRatings
        / prescoringMetaScorerOutput.finalRoundNumNotes,
        ratingsPerUserLossRatio=prescoringMetaScorerOutput.finalRoundNumRatings
        / prescoringMetaScorerOutput.finalRoundNumUsers,
      )
      logger.info(f"diligenceNP cols: {diligenceNoteParams.columns}")
      noteParams = noteParams.merge(diligenceNoteParams, on=c.noteIdKey)
      logger.info(f"np cols: {noteParams.columns}")

    if self._saveIntermediateState:
      self.noteParams = noteParams
      self.raterParams = raterParams
      self.globalBias = globalBias

    raterParamsWithRatingCounts = raterParams.merge(
      prescoringRaterModelOutput[
        [
          c.raterParticipantIdKey,
          c.incorrectTagRatingsMadeByRaterKey,
          c.totalRatingsMadeByRaterKey,
        ]
      ],
      on=c.raterParticipantIdKey,
    )

    # Merge in intercept and status without high volume and correlated raters
    if noteScoresNoHighVol is not None:
      noteParams = noteParams.merge(noteScoresNoHighVol, on=c.noteIdKey, how="left")
    if noteScoresNoCorrelated is not None:
      noteParams = noteParams.merge(noteScoresNoCorrelated, on=c.noteIdKey, how="left")
    if noteScoresPopulationSampled is not None:
      noteParams = noteParams.merge(noteScoresPopulationSampled, on=c.noteIdKey, how="left")
    else:
      # Ensure population sampled column exists even when not computed, filled with NaN
      noteParams[c.internalNoteInterceptPopulationSampledKey] = np.nan

    # Assigns updated CRH / CRNH bits to notes based on volume of prior ratings
    # and ML output.
    with self.time_block("Final compute scored notes"):
      logger.info(f"About to call compute_scored_notes with {self.get_name()}")
      if noteScoresNoHighVol is not None:
        crhThresholdNoHighVol = self._crhThresholdNoHighVol
      else:
        crhThresholdNoHighVol = None
      if noteScoresNoCorrelated is not None:
        crhThresholdNoCorrelated = self._crhThresholdNoCorrelated
      else:
        crhThresholdNoCorrelated = None
      scoredNotes = note_ratings.compute_scored_notes(
        ratings,
        noteParams,
        raterParamsWithRatingCounts,
        noteStatusHistory,
        minRatingsNeeded=self._minRatingsNeeded,
        crhThreshold=self._crhThreshold,
        crnhThresholdIntercept=self._crnhThresholdIntercept,
        crnhThresholdNoteFactorMultiplier=self._crnhThresholdNoteFactorMultiplier,
        crnhThresholdNMIntercept=self._crnhThresholdNMIntercept,
        crnhThresholdUCBIntercept=self._crnhThresholdUCBIntercept,
        crhSuperThreshold=self._crhSuperThreshold,
        inertiaDelta=self._inertiaDelta,
        tagFilterThresholds=prescoringMetaScorerOutput.tagFilteringThresholds,
        incorrectFilterThreshold=self._incorrectFilterThreshold,
        lowDiligenceThreshold=self._lowDiligenceThreshold,
        finalRound=True,
        factorThreshold=self._factorThreshold,
        firmRejectThreshold=self._firmRejectThreshold,
        minMinorityNetHelpfulRatings=self._minMinorityNetHelpfulRatings,
        minMinorityNetHelpfulRatio=self._minMinorityNetHelpfulRatio,
        crhThresholdNoHighVol=crhThresholdNoHighVol,
        crhThresholdNoCorrelated=crhThresholdNoCorrelated,
      )
      logger.info(f"sn cols: {scoredNotes.columns}")

      # Takes raterParams from the MF run, but use the pre-computed
      # helpfulness scores from prescoringRaterModelOutput.
      helpfulnessScores = raterParams.merge(
        prescoringRaterModelOutput[
          [
            c.raterParticipantIdKey,
            c.crhCrnhRatioDifferenceKey,
            c.meanNoteScoreKey,
            c.raterAgreeRatioKey,
            c.aboveHelpfulnessThresholdKey,
            c.internalFirstRoundRaterInterceptKey,
            c.internalFirstRoundRaterFactor1Key,
          ]
        ],
        on=c.raterParticipantIdKey,
        how="outer",
      )

    if self._saveIntermediateState:
      self.scoredNotes = scoredNotes
      self.helpfulnessScores = helpfulnessScores

    return scoredNotes, helpfulnessScores

  def score_final(self, scoringArgs: c.FinalScoringArgs) -> c.ModelResult:
    """
    Process ratings to assign status to notes and optionally compute rater properties.

    Accepts prescoringNoteModelOutput and prescoringRaterModelOutput as args (fields on scoringArgs)
    which are the outputs of the prescore() function.  These are used to initialize the final scoring.
    It filters the prescoring output to only include the rows relevant to this scorer, based on the
    c.scorerNameKey field of those dataframes.
    """
    torch.set_num_threads(self._threads)
    logger.info(
      f"score_final: Torch intra-op parallelism for {self.get_name()} set to: {torch.get_num_threads()}"
    )

    # Filter unfiltered params to just params for this scorer (with copy).
    # Avoid editing the dataframe in FinalScoringArgs, which is shared across scorers.
    prescoringNoteModelOutput = scoringArgs.prescoringNoteModelOutput[
      scoringArgs.prescoringNoteModelOutput[c.scorerNameKey] == self.get_name()
    ].drop(columns=c.scorerNameKey, inplace=False)

    if scoringArgs.prescoringRaterModelOutput is None:
      return self._return_empty_final_scores()
    prescoringRaterModelOutput = scoringArgs.prescoringRaterModelOutput[
      scoringArgs.prescoringRaterModelOutput[c.scorerNameKey] == self.get_name()
    ].drop(columns=c.scorerNameKey, inplace=False)

    if self.get_name() not in scoringArgs.prescoringMetaOutput.metaScorerOutput:
      logger.info(
        f"Scorer {self.get_name()} not found in prescoringMetaOutput; returning empty scores from final scoring."
      )
      return self._return_empty_final_scores()
    prescoringMetaScorerOutput = scoringArgs.prescoringMetaOutput.metaScorerOutput[self.get_name()]

    # Filter raw input
    with self.time_block("Filter input"):
      ratings, noteStatusHistory = self._filter_input(
        scoringArgs.noteTopics,
        scoringArgs.ratings,
        scoringArgs.noteStatusHistory,
        scoringArgs.userEnrollment,
      )
      # If there are no ratings left after filtering, then return empty dataframes.
      if len(ratings) == 0:
        return self._return_empty_final_scores()

    # Separate low and high volume ratings.  Note that we rely on the ratings dataframe being sorted and
    # partition the sorted dataframe to avoid creating a copy of ratings (instead we use a view that spans
    # the first N rows).
    highVolCount = ratings[c.highVolumeRaterKey].sum()
    lowVolCount = len(ratings) - highVolCount
    assert ratings.iloc[:lowVolCount][c.highVolumeRaterKey].sum() == 0
    assert ratings.iloc[lowVolCount:][c.highVolumeRaterKey].sum() == highVolCount
    logger.info(
      f"Total Ratings vs Low Vol Ratings ({self.get_name()}): {len(ratings)} vs {lowVolCount}"
    )
    noteScoresNoHighVol, _ = self._score_notes_and_users(
      ratings=ratings.iloc[:lowVolCount],
      noteStatusHistory=noteStatusHistory,
      prescoringNoteModelOutput=prescoringNoteModelOutput,
      prescoringRaterModelOutput=prescoringRaterModelOutput,
      prescoringMetaScorerOutput=prescoringMetaScorerOutput,
    )
    logger.info(
      f"noteScoresNoHighVol Summary {self.get_name()}: length ({len(noteScoresNoHighVol)}), cols ({', '.join(noteScoresNoHighVol.columns)})"
    )
    if len(noteScoresNoHighVol) > 0:
      noteScoresNoHighVol = noteScoresNoHighVol[[c.noteIdKey, c.internalNoteInterceptKey]].rename(
        columns={
          c.internalNoteInterceptKey: c.internalNoteInterceptNoHighVolKey,
        }
      )
    else:
      # Incase of low volumes, ensure that the dataframe contains the expected columns downstream
      logger.info(f"Imputing expected columns ({self.get_name()})")
      noteScoresNoHighVol[c.noteIdKey] = []
      noteScoresNoHighVol[c.internalNoteInterceptNoHighVolKey] = []

    # Separate correlated ratings. Note that we rely on the ratings dataframe being sorted and
    # partition the sorted dataframe to avoid creating a copy of ratings.
    lowVolAndUncorrelated = lowVolCount - ratings[c.correlatedRaterKey].iloc[:lowVolCount].sum()
    highVolAndUncorrelated = highVolCount - ratings[c.correlatedRaterKey].iloc[lowVolCount:].sum()
    totalUncorrelatedRatings = len(ratings) - ratings[c.correlatedRaterKey].sum()
    uncorrelatedRatings = pd.concat(
      [
        ratings.iloc[:lowVolCount].iloc[:lowVolAndUncorrelated],
        ratings.iloc[lowVolCount:].iloc[:highVolAndUncorrelated],
      ],
      copy=False,
    )
    assert (
      len(uncorrelatedRatings) == totalUncorrelatedRatings
    ), f"Unexpected mismatch ({len(uncorrelatedRatings)}, {totalUncorrelatedRatings})"
    assert uncorrelatedRatings[c.correlatedRaterKey].sum() == 0
    gc.collect()
    logger.info(
      f"Total Ratings vs Non-Correlated Ratings ({self.get_name()}): {len(ratings)} vs {totalUncorrelatedRatings}"
    )
    noteScoresNoCorrelated, _ = self._score_notes_and_users(
      ratings=uncorrelatedRatings,
      noteStatusHistory=noteStatusHistory,
      prescoringNoteModelOutput=prescoringNoteModelOutput,
      prescoringRaterModelOutput=prescoringRaterModelOutput,
      prescoringMetaScorerOutput=prescoringMetaScorerOutput,
    )
    logger.info(
      f"noteScoresNoCorrelated Summary {self.get_name()}: length ({len(noteScoresNoCorrelated)}), cols ({', '.join(noteScoresNoCorrelated.columns)})"
    )
    if len(noteScoresNoCorrelated) > 0:
      noteScoresNoCorrelated = noteScoresNoCorrelated[
        [c.noteIdKey, c.internalNoteInterceptKey]
      ].rename(
        columns={
          c.internalNoteInterceptKey: c.internalNoteInterceptNoCorrelatedKey,
        }
      )
    else:
      # Incase of low volumes, ensure that the dataframe contains the expected columns downstream
      logger.info(f"Imputing expected columns ({self.get_name()})")
      noteScoresNoCorrelated[c.noteIdKey] = []
      noteScoresNoCorrelated[c.internalNoteInterceptNoCorrelatedKey] = []

    # Separate population sampled ratings
    if (
      c.ratingSourceBucketedKey in ratings.columns
      and (ratings[c.ratingSourceBucketedKey] == c.ratingSourcePopulationSampledValueTsv).sum() > 0
    ):
      populationSampledRatings = ratings[
        ratings[c.ratingSourceBucketedKey] == c.ratingSourcePopulationSampledValueTsv
      ]
      logger.info(
        f"Total Ratings vs Population Sampled Ratings ({self.get_name()}): {len(ratings)} vs {len(populationSampledRatings)}"
      )

      noteScoresPopulationSampled, _ = self._score_notes_and_users(
        ratings=populationSampledRatings,
        noteStatusHistory=noteStatusHistory,
        prescoringNoteModelOutput=prescoringNoteModelOutput,
        prescoringRaterModelOutput=prescoringRaterModelOutput,
        prescoringMetaScorerOutput=prescoringMetaScorerOutput,
        ratingPerNoteLossRatio=self._populationSampledRatingPerNoteLossRatio,
      )
      logger.info(
        f"noteScoresPopulationSampled Summary {self.get_name()}: length ({len(noteScoresPopulationSampled)}), cols ({', '.join(noteScoresPopulationSampled.columns)})"
      )
      if len(noteScoresPopulationSampled) > 0:
        noteScoresPopulationSampled = noteScoresPopulationSampled[
          [c.noteIdKey, c.internalNoteInterceptKey]
        ].rename(
          columns={
            c.internalNoteInterceptKey: c.internalNoteInterceptPopulationSampledKey,
          }
        )
      else:
        # In case of low volumes, ensure that the dataframe contains the expected columns downstream
        logger.info(f"Imputing expected columns for population sampled ({self.get_name()})")
        noteScoresPopulationSampled = pd.DataFrame(
          {
            c.noteIdKey: pd.array([], dtype=np.int64),
            c.internalNoteInterceptPopulationSampledKey: pd.array([], dtype=np.float64),
          }
        )
    else:
      logger.info(
        f"No population sampled ratings found for {self.get_name()}, skipping population sampled computation"
      )
      noteScoresPopulationSampled = None

    noteScores, userScores = self._score_notes_and_users(
      ratings=ratings,
      noteStatusHistory=noteStatusHistory,
      prescoringNoteModelOutput=prescoringNoteModelOutput,
      prescoringRaterModelOutput=prescoringRaterModelOutput,
      prescoringMetaScorerOutput=prescoringMetaScorerOutput,
      flipFactorsForIdentification=False,
      noteScoresNoHighVol=noteScoresNoHighVol,
      noteScoresNoCorrelated=noteScoresNoCorrelated,
      noteScoresPopulationSampled=noteScoresPopulationSampled,
    )
    logger.info(
      f"noteScores Summary {self.get_name()}: length ({len(noteScores)}), cols ({', '.join(noteScores.columns)})"
    )

    if len(noteScores) == 0 and len(userScores) == 0:
      logger.info(
        "No ratings left after filtering that happens in _score_notes_and_users, returning empty "
        "dataframes"
      )
      return self._return_empty_final_scores()

    with self.time_block("Postprocess output"):
      # Only some subclasses do any postprocessing.
      # E.g. topic models add confidence bit, group models prune according to authorship filter.
      noteScores, userScores = self._postprocess_output(
        noteScores,
        userScores,
        scoringArgs.ratings,
        scoringArgs.noteStatusHistory,
        scoringArgs.userEnrollment,
      )

      ## TODO: refactor this logic to compute 2nd round ratings out so score_final doesn't need to be overridden and duplicated.
      scoredNoteFinalRoundRatings = (
        ratings[[c.raterParticipantIdKey, c.noteIdKey]]
        .merge(userScores[[c.raterParticipantIdKey]], on=c.raterParticipantIdKey)
        .groupby(c.noteIdKey)
        .agg("count")
        .reset_index()
        .rename(columns={c.raterParticipantIdKey: c.numFinalRoundRatingsKey})
      )

      noteScores = noteScores.merge(
        scoredNoteFinalRoundRatings,
        on=c.noteIdKey,
        how="left",
        unsafeAllowed=[c.defaultIndexKey, c.numFinalRoundRatingsKey],
      )

      noteScores = noteScores.rename(columns=self._get_note_col_mapping())
      userScores = userScores.rename(columns=self._get_user_col_mapping())

      # Process noteScores
      noteScores = noteScores.drop(columns=self._get_dropped_note_cols())
      assert set(noteScores.columns) == set(
        self.get_scored_notes_cols() + self.get_auxiliary_note_info_cols()
      ), f"""all columns must be either dropped or explicitly defined in an output. 
      Extra columns that were in noteScores: {set(noteScores.columns) - set(self.get_scored_notes_cols() + self.get_auxiliary_note_info_cols())}
      Missing expected columns that should've been in noteScores: {set(self.get_scored_notes_cols() + self.get_auxiliary_note_info_cols()) - set(noteScores.columns)}"""

      # Process userScores
      userScores = userScores.drop(columns=self._get_dropped_user_cols())
      assert set(userScores.columns) == set(self.get_helpfulness_scores_cols()), f"""all columns must be either dropped or explicitly defined in an output. 
      Extra columns that were in userScores: {set(userScores.columns) - set(self.get_helpfulness_scores_cols())}
      Missing expected columns that should've been in userScores: {set(self.get_helpfulness_scores_cols()) - set(userScores.columns)}"""

    # Return dataframes with specified columns in specified order
    return c.ModelResult(
      scoredNotes=noteScores[self.get_scored_notes_cols()],
      helpfulnessScores=userScores[self.get_helpfulness_scores_cols()]
      if self.get_helpfulness_scores_cols()
      else None,
      auxiliaryNoteInfo=noteScores[self.get_auxiliary_note_info_cols()]
      if self.get_auxiliary_note_info_cols()
      else None,
      scorerName=self.get_name(),
      metaScores=None,
    )
