from typing import List, Optional, Tuple

from . import constants as c, helpfulness_scores, note_ratings, process_data, tag_consensus
from .matrix_factorization.matrix_factorization import MatrixFactorization
from .matrix_factorization.pseudo_raters import PseudoRatersRunner
from .reputation_matrix_factorization.diligence_model import get_low_diligence_intercepts
from .scorer import Scorer

import numpy as np
import pandas as pd
import torch


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
    ratingsForTrainingWithModelingGroup.groupby(c.noteIdKey)
    .sum()[[c.ratingFromInitialModelingGroupKey, c.ratingCountKey]]
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
    crhSuperThreshold: float = 0.5,
    lowDiligenceThreshold: float = 0.217,
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
  ):
    """Configure MatrixFactorizationScorer object.

    Args:
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
    """
    super().__init__(seed, threads)
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
    self._inertiaDelta = inertiaDelta
    self._modelingGroupToInitializeForStability = 13 if useStableInitialization else None
    self._saveIntermediateState = saveIntermediateState
    self._maxFirstMFTrainError = maxFirstMFTrainError
    self._maxFinalMFTrainError = maxFinalMFTrainError
    self._lowDiligenceThreshold = lowDiligenceThreshold
    self._factorThreshold = factorThreshold
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
    return (
      [
        c.currentlyRatedHelpfulBoolKey,
        c.currentlyRatedNotHelpfulBoolKey,
        c.awaitingMoreRatingsBoolKey,
        c.currentLabelKey,
        c.classificationKey,
        c.numRatingsKey,
        c.noteAuthorParticipantIdKey,
      ]
      + c.helpfulTagsTSVOrder
      + c.notHelpfulTagsTSVOrder
      + c.noteParameterUncertaintyTSVAuxColumns
    )

  def _get_dropped_user_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from helpfulnessScores output."""
    return []

  def _prepare_data_for_scoring(self, ratings: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for scoring. This includes filtering out notes and raters which do not meet
    minimum rating counts, and may be overridden by subclasses to add additional filtering.
    """
    return process_data.filter_ratings(
      ratings, self._minNumRatingsPerRater, self._minNumRatersPerNote
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
    return self._mfRanker.run_mf(ratingsForTraining)

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
        ratingsForTraining, userEnrollmentRaw, self._modelingGroupToInitializeForStability
      )

    with self.time_block("MF on stable-initialization subset"):
      initializationMF = self._mfRanker.get_new_mf_with_same_args()
      noteParamsInit, raterParamsInit, globalInterceptInit = initializationMF.run_mf(
        ratingsForStableInitialization
      )

    with self.time_block("First full MF (initializated with stable-initialization)"):
      modelResult = self._mfRanker.run_mf(
        ratingsForTraining,
        noteInit=noteParamsInit,
        userInit=raterParamsInit,
        globalInterceptInit=globalInterceptInit,
      )
    return modelResult

  def _score_notes_and_users(
    self, ratings: pd.DataFrame, noteStatusHistory: pd.DataFrame, userEnrollmentRaw: pd.DataFrame
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the matrix factorization scoring algorithm.

    See links below for more info:
      https://twitter.github.io/communitynotes/ranking-notes/
      https://twitter.github.io/communitynotes/contributor-scores/.

    Args:
      ratings (pd.DataFrame): preprocessed ratings
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
      userEnrollmentRaw (pd.DataFrame): one row per user; enrollment status for each user

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame]:
        noteScores pd.DataFrame: one row per note contained note scores and parameters.
        userScores pd.DataFrame: one row per user containing a column for each helpfulness score.
    """
    if self._seed is not None:
      print(f"seeding with {self._seed}")
      torch.manual_seed(self._seed)

    # Removes ratings where either (1) the note did not receive enough ratings, or
    # (2) the rater did not rate enough notes.
    with self.time_block("Prepare ratings"):
      ratingsForTraining = self._prepare_data_for_scoring(ratings)
    if self._saveIntermediateState:
      self.ratingsForTraining = ratingsForTraining

    # TODO: Save parameters from this first run in note_model_output next time we add extra fields to model output TSV.
    with self.time_block("First MF/stable init"):
      (
        noteParamsUnfiltered,
        raterParamsUnfiltered,
        globalBias,
      ) = self._run_stable_matrix_factorization(ratingsForTraining, userEnrollmentRaw)
    if self._saveIntermediateState:
      self.noteParamsUnfiltered = noteParamsUnfiltered
      self.raterParamsUnfiltered = raterParamsUnfiltered
      self.globalBias = globalBias
    self.assert_train_error_is_below_threshold(ratingsForTraining, self._maxFirstMFTrainError)

    # Get a dataframe of scored notes based on the algorithm results above
    with self.time_block("Compute scored notes"):
      scoredNotes = note_ratings.compute_scored_notes(
        ratings,
        noteParamsUnfiltered,
        raterParamsUnfiltered,
        noteStatusHistory,
        minRatingsNeeded=self._minRatingsNeeded,
        crhThreshold=self._crhThreshold,
        crnhThresholdIntercept=self._crnhThresholdIntercept,
        crnhThresholdNoteFactorMultiplier=self._crnhThresholdNoteFactorMultiplier,
        crnhThresholdNMIntercept=self._crnhThresholdNMIntercept,
        crnhThresholdUCBIntercept=self._crnhThresholdUCBIntercept,
        crhSuperThreshold=self._crhSuperThreshold,
        inertiaDelta=self._inertiaDelta,
        lowDiligenceThreshold=self._lowDiligenceThreshold,
      )
    if self._saveIntermediateState:
      self.firstRoundScoredNotes = scoredNotes

    # Determine "valid" ratings
    with self.time_block("Compute valid ratings"):
      validRatings = note_ratings.get_valid_ratings(
        ratings,
        noteStatusHistory,
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

    # Assigns contributor (author & rater) helpfulness bit based on (1) performance
    # authoring and reviewing previous and current notes.
    with self.time_block("Helpfulness scores pre-harassment "):
      helpfulnessScoresPreHarassmentFilter = helpfulness_scores.compute_general_helpfulness_scores(
        scoredNotes[
          [
            c.noteAuthorParticipantIdKey,
            c.currentlyRatedHelpfulBoolKey,
            c.currentlyRatedNotHelpfulBoolKey,
            c.internalNoteInterceptKey,
          ]
        ],
        validRatings,
        self._minMeanNoteScore,
        self._minCRHVsCRNHRatio,
        self._minRaterAgreeRatio,
        ratingsForTraining,
      )
      if self._saveIntermediateState:
        self.firstRoundHelpfulnessScores = helpfulnessScoresPreHarassmentFilter

      # Filters ratings matrix to include only rows (ratings) where the rater was
      # considered helpful.
      ratingsHelpfulnessScoreFilteredPreHarassmentFilter = (
        helpfulness_scores.filter_ratings_by_helpfulness_scores(
          ratingsForTraining, helpfulnessScoresPreHarassmentFilter
        )
      )

    if self._saveIntermediateState:
      self.ratingsHelpfulnessScoreFilteredPreHarassmentFilter = (
        ratingsHelpfulnessScoreFilteredPreHarassmentFilter
      )

    with self.time_block("Harassment tag consensus"):
      harassmentAbuseNoteParams, _, _ = tag_consensus.train_tag_model(
        ratingsHelpfulnessScoreFilteredPreHarassmentFilter,
        c.notHelpfulSpamHarassmentOrAbuseTagKey,
        noteParamsUnfiltered,
        raterParamsUnfiltered,
      )

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
        validRatings,
        self._minMeanNoteScore,
        self._minCRHVsCRNHRatio,
        self._minRaterAgreeRatio,
        ratings=ratingsForTraining,
        tagConsensusHarassmentAbuseNotes=harassmentAbuseNoteParams,
      )

      # Filters ratings matrix to include only rows (ratings) where the rater was
      # considered helpful.
      ratingsHelpfulnessScoreFiltered = helpfulness_scores.filter_ratings_by_helpfulness_scores(
        ratingsForTraining, helpfulnessScores
      )
    if self._saveIntermediateState:
      self.helpfulnessScores = helpfulnessScores
      self.ratingsHelpfulnessScoreFiltered = ratingsHelpfulnessScoreFiltered

    # Re-runs matrix factorization using only ratings given by helpful raters.
    with self.time_block("Final helpfulness-filtered MF"):
      noteParams, raterParams, globalBias = self._mfRanker.run_mf(
        ratingsHelpfulnessScoreFiltered,
        noteInit=noteParamsUnfiltered,
        userInit=raterParamsUnfiltered,
      )
    if self._saveIntermediateState:
      self.noteParams = noteParams
      self.raterParams = raterParams
      self.globalBias = globalBias
    self.assert_train_error_is_below_threshold(
      ratingsHelpfulnessScoreFiltered, self._maxFinalMFTrainError
    )

    # Add pseudo-raters with the most extreme parameters and re-score notes, to estimate
    #  upper and lower confidence bounds on note parameters.
    if self._pseudoraters:
      with self.time_block("Pseudoraters"):
        noteParams = PseudoRatersRunner(
          ratingsHelpfulnessScoreFiltered, noteParams, raterParams, globalBias, self._mfRanker
        ).compute_note_parameter_confidence_bounds_with_pseudo_raters()
        if self._saveIntermediateState:
          self.prePseudoratersNoteParams = self.noteParams
          self.noteParams = noteParams
    else:
      for col in c.noteParameterUncertaintyTSVColumns:
        noteParams[col] = np.nan

    # Add low diligence intercepts
    with self.time_block("Low Diligence Reputation Model"):
      diligenceParams = get_low_diligence_intercepts(
        ratingsHelpfulnessScoreFiltered, raterInitState=raterParams
      )
      noteParams = noteParams.merge(diligenceParams, on=c.noteIdKey)

    if self._saveIntermediateState:
      self.noteParams = noteParams
      self.raterParams = raterParams
      self.globalBias = globalBias

    # Assigns updated CRH / CRNH bits to notes based on volume of prior ratings
    # and ML output.
    with self.time_block("Final compute scored notes"):
      scoredNotes = note_ratings.compute_scored_notes(
        ratings,
        noteParams,
        raterParams,
        noteStatusHistory,
        minRatingsNeeded=self._minRatingsNeeded,
        crhThreshold=self._crhThreshold,
        crnhThresholdIntercept=self._crnhThresholdIntercept,
        crnhThresholdNoteFactorMultiplier=self._crnhThresholdNoteFactorMultiplier,
        crnhThresholdNMIntercept=self._crnhThresholdNMIntercept,
        crnhThresholdUCBIntercept=self._crnhThresholdUCBIntercept,
        crhSuperThreshold=self._crhSuperThreshold,
        inertiaDelta=self._inertiaDelta,
        lowDiligenceThreshold=self._lowDiligenceThreshold,
        finalRound=True,
        factorThreshold=self._factorThreshold,
      )

      # Takes raterParams from most recent MF run, but use the pre-computed
      # helpfulness scores.
      helpfulnessScores = raterParams.merge(
        helpfulnessScores[
          [
            c.raterParticipantIdKey,
            c.crhCrnhRatioDifferenceKey,
            c.meanNoteScoreKey,
            c.raterAgreeRatioKey,
            c.aboveHelpfulnessThresholdKey,
          ]
        ],
        on=c.raterParticipantIdKey,
        how="outer",
      )

    if self._saveIntermediateState:
      self.scoredNotes = scoredNotes
      self.helpfulnessScores = helpfulnessScores

    return scoredNotes, helpfulnessScores
