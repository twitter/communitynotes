import gc
import logging
import random
from typing import Dict, List, Optional, Set, Tuple

from . import constants as c, helpfulness_scores, note_ratings, process_data, tag_filter
from .matrix_factorization.matrix_factorization import MatrixFactorization
from .mf_core_scorer import MFCoreScorer
from .reputation_matrix_factorization.diligence_model import fit_low_diligence_model_final
from .scorer import Scorer

import numpy as np
import pandas as pd
import torch


logger = logging.getLogger("birdwatch.mf_base_scorer")
logger.setLevel(logging.INFO)


def compute_empirical_prior_df(
  ratingsWithFactors: pd.DataFrame,
  binWidth: float = 0.2,
  minNeg: int = 10,
  minPos: int = 10,
  maxNeg: int = 50,
  maxPos: int = 50,
) -> pd.DataFrame:
  ratingsWithFactors = ratingsWithFactors.loc[
    ratingsWithFactors[c.helpfulNumKey] != 0.5,
    [c.noteIdKey, c.internalRaterFactor1Key, c.helpfulNumKey],
  ]

  # bins x values
  ratingsWithFactors["bin"] = np.floor(
    (ratingsWithFactors[c.internalRaterFactor1Key] + 1.0) / binWidth
  ).astype(int)
  max_bin = int(np.floor(2.0 / binWidth))
  ratingsWithFactors["bin"] = ratingsWithFactors["bin"].clip(0, max_bin)

  # compute sign counts for bins, limit to notes within a certain range
  signCounts = (
    ratingsWithFactors.assign(
      neg=ratingsWithFactors[c.internalRaterFactor1Key] < 0,
      pos=ratingsWithFactors[c.internalRaterFactor1Key] > 0,
    )
    .groupby(c.noteIdKey)[["neg", "pos"]]
    .sum()
    .astype(int)
  )

  candidateMask = (
    (signCounts["neg"] >= minNeg)
    & (signCounts["pos"] >= minPos)
    & (signCounts["neg"] <= maxNeg)
    & (signCounts["pos"] <= maxPos)
  )
  candidateIds = signCounts[candidateMask].index

  candidates = ratingsWithFactors[ratingsWithFactors[c.noteIdKey].isin(candidateIds)]

  # build count matrix only for candidates
  countDf = (
    candidates.groupby([c.noteIdKey, "bin", c.helpfulNumKey]).size().reset_index(name="count")
  )
  pivot = countDf.pivot_table(
    index=c.noteIdKey, columns=["bin", c.helpfulNumKey], values="count", fill_value=0
  ).astype(np.int32)

  return pivot


def lookup_empirical_priors(
  empiricalTotals: pd.DataFrame,
  ratingsWithFactors: pd.DataFrame,
  allCurves: pd.DataFrame,
  strictMin: int = 7,
):
  candidateMatrix = empiricalTotals.values
  candidateIndex = empiricalTotals.index.values

  queryMatrix = compute_empirical_prior_df(
    ratingsWithFactors, minNeg=3, maxNeg=np.inf, minPos=3, maxPos=np.inf
  )

  # Reindex to match candidate columns exactly (adds missing as 0)
  queryMatrix = queryMatrix.reindex(columns=empiricalTotals.columns, fill_value=0).astype(np.int32)

  queryIds = queryMatrix.index.values
  queryMatrix = queryMatrix.values  # (Q, F)

  # require either that the prior candidates have strictly more than the query, or more than strictMin
  requiredMin = np.where(queryMatrix <= strictMin, queryMatrix, strictMin)

  requiredMin = np.where(queryMatrix > 0, requiredMin, 0)

  Q, C = len(queryIds), len(candidateIndex)
  isSupersetMatrix = np.ones((Q, C), dtype=bool)

  for f in range(candidateMatrix.shape[1]):
    colGe = candidateMatrix[:, f][np.newaxis, :] >= requiredMin[:, f][:, np.newaxis]
    isSupersetMatrix &= colGe

  # Remove self-matches
  if np.isin(queryIds, candidateIndex).any():
    # Map candidate_index to column positions
    candPos = {id_: i for i, id_ in enumerate(candidateIndex)}
    # Find which queries are also candidates
    selfMatchQueries = np.isin(queryIds, candidateIndex)
    queryRowsSelf = np.where(selfMatchQueries)[0]
    selfCols = np.array([candPos[qid] for qid in queryIds[selfMatchQueries]])
    isSupersetMatrix[queryRowsSelf, selfCols] = False

  # Collect 30 samples for each, fill with 0 where < 30 available
  supersetLists = np.array(
    [
      random.choices(candidateIndex[row].tolist(), k=30)
      if len(candidateIndex[row].tolist()) >= 30
      else candidateIndex[row].tolist() + [0] * (30 - len(candidateIndex[row].tolist()))
      for row in isSupersetMatrix
    ]
  )

  # Map back to original query_ids order
  resultSupersets = pd.DataFrame(
    np.empty(shape=(len(queryIds), 30), dtype=np.int64), index=queryIds
  )
  resultSupersets.loc[queryIds] = supersetLists

  noteIds = ratingsWithFactors[c.noteIdKey].values
  uniqueNotes, noteIndices = np.unique(noteIds, return_inverse=True)

  # calculate average of gaussian curves for prior candidates
  curveDf = pd.DataFrame(allCurves, index=uniqueNotes)
  nanDf = pd.DataFrame(np.nan, columns=curveDf.columns, index=[0], dtype=np.float64)
  curveDfWNan = pd.concat([curveDf, nanDf])
  testDf = curveDfWNan.loc[resultSupersets.values.flatten()]
  dfAvg = testDf.groupby(np.arange(len(testDf)) // 30).mean()
  dfAvg.index = queryIds
  return dfAvg


class GaussianScorer(Scorer):
  """Runs Gaussian convolution rating aggregation to determine raw note scores and ultimately note status."""

  def __init__(
    self,
    includedTopics: Set[str] = set(),
    excludeTopics: bool = True,
    includedGroups: Set[int] = c.coverageGroups,  # by default this runs only for core
    includeUnassigned: bool = True,
    strictInclusion: bool = False,
    captureThreshold: Optional[float] = 0.5,
    seed: Optional[int] = None,
    minNumRatingsPerRater: int = 10,
    minNumRatersPerNote: int = 5,
    minRatingsNeeded: int = 5,
    minMeanNoteScore: float = 0.05,
    minCRHVsCRNHRatio: float = 0.00,
    minRaterAgreeRatio: float = 0.66,
    crhThreshold: float = 0.51,
    crnhThresholdIntercept: float = -0.5,  # set this to a number that it should not be in effect
    crnhThresholdNoteFactorMultiplier: float = 0,
    crnhThresholdNMIntercept: float = -0.6,
    crnhThresholdUCBIntercept: float = -0.5,
    crhSuperThreshold: Optional[float] = 0.65,
    crhThresholdNoHighVol: float = 0.45,
    crhThresholdNoCorrelated: float = 0.45,
    lowDiligenceThreshold: float = 0.263,
    factorThreshold: float = 1.0,
    inertiaDelta: float = 0.01,
    saveIntermediateState: bool = False,
    threads: int = c.defaultNumThreads,
    useReputation: bool = True,
    tagFilterPercentile: int = 95,
    incorrectFilterThreshold: float = 2.5,
    firmRejectThreshold: Optional[float] = None,
    minMinorityNetHelpfulRatings: Optional[int] = None,
    minMinorityNetHelpfulRatio: Optional[float] = None,
    populationSampledRatingPerNoteLossRatio: Optional[float] = 10.0,
    nPoints=50,
    nBinsEachSide=25,
    calculateBins=False,
    crhParams: c.GaussianParams = c.gaussianCrhParams,
    crnhParams: c.GaussianParams = c.gaussianCrnhParams,
    useMfNoteParams=True,
  ):
    """Configure GaussianScorer object.

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
    self._saveIntermediateState = saveIntermediateState
    self._lowDiligenceThreshold = lowDiligenceThreshold
    self._factorThreshold = factorThreshold
    self._useReputation = useReputation
    self._tagFilterPercentile = tagFilterPercentile
    self._incorrectFilterThreshold = incorrectFilterThreshold
    self._firmRejectThreshold = firmRejectThreshold
    self._minMinorityNetHelpfulRatings = minMinorityNetHelpfulRatings
    self._minMinorityNetHelpfulRatio = minMinorityNetHelpfulRatio
    self._populationSampledRatingPerNoteLossRatio = populationSampledRatingPerNoteLossRatio
    self._nPoints = nPoints
    self._nBinsEachSide = nBinsEachSide
    self._calculateBins = calculateBins
    self._crhParams = crhParams
    self._crnhParams = crnhParams
    self._useMfNoteParams = useMfNoteParams

  def get_prescoring_name(self):
    return "MFCoreScorer"

  def get_name(self):
    return "GaussianScorer"

  def _get_note_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default note column names to custom names for a specific model."""
    return {
      c.internalNoteInterceptKey: c.gaussianNoteInterceptKey,
      c.internalNoteFactor1Key: c.gaussianNoteFactor1Key,
      c.internalActiveRulesKey: c.gaussianActiveRulesKey,
      c.numFinalRoundRatingsKey: c.gaussianNumFinalRoundRatingsKey,
      c.internalNoteInterceptNoHighVolKey: c.gaussianNoteInterceptNoHighVolKey,
      c.internalNoteInterceptNoCorrelatedKey: c.gaussianNoteInterceptNoCorrelatedKey,
      c.internalNoteInterceptPopulationSampledKey: c.gaussianNoteInterceptPopulationSampledKey,
      c.lowDiligenceNoteInterceptKey: c.lowDiligenceLegacyNoteInterceptKey,
      c.internalRatingStatusKey: c.gaussianRatingStatusKey,
    }

  def get_crh_threshold(self) -> float:
    """Return CRH threshold for general scoring logic."""
    return self._crhThreshold

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      c.gaussianNoteInterceptKey,
      c.gaussianNoteFactor1Key,
      c.gaussianRatingStatusKey,
      c.gaussianActiveRulesKey,
      c.gaussianNumFinalRoundRatingsKey,
      c.gaussianNoteInterceptNoHighVolKey,
      c.gaussianNoteInterceptNoCorrelatedKey,
      c.gaussianNoteInterceptPopulationSampledKey,
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
    ]

  def get_internal_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of internal columns which should be present in the helpfulnessScores output."""
    return [
      c.raterParticipantIdKey,
    ]

  def get_auxiliary_note_info_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the auxiliaryNoteInfo output."""
    return [
      c.noteIdKey,
    ]

  def _get_dropped_note_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from scoredNotes and auxiliaryNoteInfo."""
    droppedCols = (
      [
        c.currentlyRatedHelpfulBoolKey,
        c.currentlyRatedNotHelpfulBoolKey,
        c.awaitingMoreRatingsBoolKey,
        c.currentLabelKey,
        c.classificationKey,
        c.numRatingsKey,
        c.noteAuthorParticipantIdKey,
        c.activeFilterTagsKey,
        c.noteInterceptMaxKey,
        c.noteInterceptMinKey,
        c.ratingWeightKey,
      ]
      + (
        c.notHelpfulTagsAdjustedColumns
        + c.notHelpfulTagsAdjustedRatioColumns
        + c.incorrectFilterColumns
      )
      + [c.posFactorPopulationSampledRatingCountKey, c.negFactorPopulationSampledRatingCountKey]
    )

    # only drop population sampled column if it's not mapped to an output column
    note_col_mapping = self._get_note_col_mapping()
    if c.internalNoteInterceptPopulationSampledKey not in note_col_mapping:
      droppedCols.extend(
        [
          c.internalNoteInterceptPopulationSampledKey,
          c.negFactorPopulationSampledRatingCountKey,
          c.posFactorPopulationSampledRatingCountKey,
        ]
      )

    return (
      droppedCols
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
    if final == False:
      raise Exception("Gaussian scorer should not be used for prescoring")
    else:
      return process_data.filter_ratings(
        ratings, minNumRatingsPerRater=0, minNumRatersPerNote=self._minNumRatersPerNote
      )

  # Optimize kernel computation
  def _gaussian_kernel(self, distances, isCrh=True):
    """Vectorized kernel computation with pre-computed constants."""
    bandwidth = self._crhParams.bandwidth if isCrh else self._crnhParams.bandwidth
    normConst = 1.0 / (bandwidth * np.sqrt(2 * np.pi))
    return normConst * np.exp(-0.5 * (distances / bandwidth) ** 2)

  def _return_all_pts(
    self,
    ratingsForTrainingWithFactors: pd.DataFrame,
    quantileRange: np.array,
    isCrh=True,
    empiricalPriors=None,
  ) -> pd.DataFrame:
    params = self._crhParams if isCrh else self._crnhParams

    numQuantiles = len(quantileRange)
    quantileCols = [f"{x:5.2f}" for x in quantileRange]
    quantileArray = np.array(quantileRange, dtype=np.float32)

    assert (
      ratingsForTrainingWithFactors[c.internalRaterFactor1Key].isna().sum() == 0
    ), "all rater factors must be non nan"
    noteIds = ratingsForTrainingWithFactors[c.noteIdKey].values
    helpfulScores = ratingsForTrainingWithFactors[c.helpfulNumKey].values.astype(np.float32)
    # Adjust helpful for somewhat helpful (original helpful used below for priors)
    somewhatMask = helpfulScores == 0.5
    helpfulScores[somewhatMask] = params.somewhatHelpfulValue

    if params.flip:
      helpfulScores = 1 - helpfulScores

    raterFactors = ratingsForTrainingWithFactors[c.internalRaterFactor1Key].values.astype(
      np.float32
    )
    notHelpfulVariances = (
      ratingsForTrainingWithFactors.loc[
        ratingsForTrainingWithFactors[c.helpfulNumKey] == (0 if not params.flip else 1)
      ]
      .groupby(c.noteIdKey)[c.internalRaterFactor1Key]
      .agg("mean")
      .reset_index()
    )
    notHelpfulVariances.rename(columns={c.internalRaterFactor1Key: "mean"}, inplace=True)
    allRatings = (
      ratingsForTrainingWithFactors[[c.noteIdKey, c.internalRaterFactor1Key]]
      .groupby(c.noteIdKey)
      .agg("count")
    )
    allRatings.rename(columns={c.internalRaterFactor1Key: "count"}, inplace=True)
    notHelpfulVariances = notHelpfulVariances.merge(allRatings, on=c.noteIdKey)

    stdWindow = 0.28
    counts = ratingsForTrainingWithFactors.loc[
      ratingsForTrainingWithFactors["helpfulNum"] == (0 if not params.flip else 1)
    ][[c.noteIdKey, c.internalRaterFactor1Key]].merge(notHelpfulVariances, on=c.noteIdKey)
    counts["dist"] = np.abs(counts[c.internalRaterFactor1Key] - counts["mean"])

    counts[c.internalRaterFactor1Key] = 1 / (1 + np.exp(-200 * (counts["dist"] - stdWindow / 2)))
    unexpected = (
      counts[[c.internalRaterFactor1Key, c.noteIdKey]].groupby(c.noteIdKey).agg("sum").reset_index()
    )
    unexpected[c.internalRaterFactor1Key] = unexpected[c.internalRaterFactor1Key].astype(np.float64)
    varsWUnexpected = notHelpfulVariances.merge(unexpected, how="left", on=c.noteIdKey).fillna(0)
    varsWUnexpected["pct"] = varsWUnexpected[c.internalRaterFactor1Key] / (varsWUnexpected["count"])
    varsWUnexpected.set_index(c.noteIdKey, inplace=True)
    notHelpfulVariances = (varsWUnexpected.loc[:, "pct"] * (4 if params.flip else 3) + 1).clip(
      upper=(4.0 if params.flip else 2.0)
    )

    uniqueNotes, noteIndices = np.unique(noteIds, return_inverse=True)
    numNotes = len(uniqueNotes)
    ratingCounts = np.bincount(noteIndices, minlength=numNotes).astype(np.float32)

    # Compute kernel weights
    raterExpanded = raterFactors[:, None]
    centers = quantileArray[None, :]
    distances = np.abs(raterExpanded - centers)
    kernelWeights = self._gaussian_kernel(distances, isCrh=isCrh).astype(np.float32)

    # Normalize rows
    if params.normalize:
      rowSums = kernelWeights.sum(axis=1, keepdims=True)
      weights = kernelWeights / rowSums
    else:
      weights = kernelWeights

    # Base total weights for insufficient check
    totalBaseWeights = np.empty((numNotes, numQuantiles), dtype=np.float32)
    for quantileIndex in range(numQuantiles):
      totalBaseWeights[:, quantileIndex] = np.bincount(
        noteIndices, weights=weights[:, quantileIndex], minlength=numNotes
      )
    insufficientData = (totalBaseWeights < params.weightLim).any(axis=1)

    # Adjust weights for negative ratings
    negMask = helpfulScores == 0
    if negMask.any():
      notHelpfulVariances = notHelpfulVariances.reindex(uniqueNotes, fill_value=1.0)
      varianceFactors = np.where(
        np.isnan(notHelpfulVariances[noteIds]), 1.0, notHelpfulVariances[noteIds]
      )
      weights[negMask] *= varianceFactors[negMask][:, np.newaxis]

    weights[negMask] *= params.negWeight

    # Rater total weights (post-neg adjustment)
    totalRaterWeights = np.empty((numNotes, numQuantiles), dtype=np.float32)
    for quantileIndex in range(numQuantiles):
      totalRaterWeights[:, quantileIndex] = np.bincount(
        noteIndices, weights=weights[:, quantileIndex], minlength=numNotes
      )

    # Rater values and totals
    weightedRaterValues = weights * helpfulScores[:, None]
    totalRaterValues = np.empty((numNotes, numQuantiles), dtype=np.float32)
    for quantileIndex in range(numQuantiles):
      totalRaterValues[:, quantileIndex] = np.bincount(
        noteIndices, weights=weightedRaterValues[:, quantileIndex], minlength=numNotes
      )

    # Compute note priors using average of helpful rater factors - NH rater factors
    originalHelpful = ratingsForTrainingWithFactors[c.helpfulNumKey].values.astype(np.float32)
    if params.flip:
      originalHelpful = 1 - originalHelpful
    adjustedHelpful = (originalHelpful - 0.5) * 2 * raterFactors
    sumFactors = np.bincount(noteIndices, weights=adjustedHelpful, minlength=numNotes)
    noteFactors = sumFactors / ratingCounts

    # Smoothing values
    if params.priorFactor:
      priorValues = (noteFactors[:, None] * centers) / 2 + params.smoothingValue
      smoothingValues = np.minimum(priorValues, params.smoothingValue).astype(np.float32)
      if params.minPrior is not None:
        smoothingValues = np.maximum(smoothingValues, params.minPrior).astype(np.float32)
    else:
      smoothingValues = np.full((numNotes, numQuantiles), params.smoothingValue, dtype=np.float32)

    if empiricalPriors is not None:
      smoothingValues[np.where(np.isin(uniqueNotes, empiricalPriors.index))[0]] = empiricalPriors[
        quantileCols
      ].values

    if not isCrh:
      # Ensure notes with fewer than 3 ratings on each side get 0.1 smoothing
      signCounts = (
        ratingsForTrainingWithFactors.assign(
          neg=ratingsForTrainingWithFactors[c.internalRaterFactor1Key] < 0,
          pos=ratingsForTrainingWithFactors[c.internalRaterFactor1Key] > 0,
        )
        .groupby(c.noteIdKey)[["neg", "pos"]]
        .sum()
        .astype(int)
      )
      insufficientMask = (signCounts["neg"] < 3) | (signCounts["pos"] < 3)
      insufficientNoteIds = signCounts[insufficientMask].index
      isInsufficient = np.isin(uniqueNotes, insufficientNoteIds)
      smoothingValues[isInsufficient] = 0.1

    # Smoothing weights
    if params.adaptiveWeightBase is not None:
      smoothingWeights = (
        np.log(np.maximum(ratingCounts, params.adaptiveWeightBase))
        / np.log(params.adaptiveWeightBase)
        * params.smoothingWeight
      )
    else:
      smoothingWeights = np.full(numNotes, params.smoothingWeight, dtype=np.float32)
    smoothingWeights = smoothingWeights.astype(np.float32)
    # Totals with prior
    totalWeightsWithPrior = totalRaterWeights + smoothingWeights[:, None]
    totalValuesWithPrior = totalRaterValues + (smoothingValues * smoothingWeights[:, None])
    # Weighted means with fillna(0.05)
    outArray = np.full_like(totalValuesWithPrior, params.clipLower, dtype=np.float32)
    weightedMeanMatrix = np.divide(
      totalValuesWithPrior, totalWeightsWithPrior, out=outArray, where=totalWeightsWithPrior != 0
    )

    # Clip
    clippedValues = np.clip(weightedMeanMatrix, params.clipLower, params.clipUpper)

    # Set insufficient to -1
    clippedValues[insufficientData, :] = -1

    clippedDf = pd.DataFrame(clippedValues, columns=quantileCols, index=uniqueNotes)
    clippedDf[c.internalNoteFactor1Key] = noteFactors

    return clippedDf

  def _gaussian_kernel_extrapolator_vectorized(
    self,
    ratingsForTrainingWithFactors: pd.DataFrame,
    quantileRange: np.array,
    isCrh: bool = True,
    empiricalTotals: Optional[pd.DataFrame] = None,
  ):
    clippedValues = self._return_all_pts(
      ratingsForTrainingWithFactors,
      quantileRange,
      isCrh=isCrh,
    )
    if empiricalTotals is not None:
      empiricalPriors = lookup_empirical_priors(
        empiricalTotals, ratingsForTrainingWithFactors, clippedValues
      )
      clippedValues = self._return_all_pts(
        ratingsForTrainingWithFactors, quantileRange, isCrh=isCrh, empiricalPriors=empiricalPriors
      )

    quantileCols = [f"{x:5.2f}" for x in quantileRange]

    # Compute intercept
    logValues = np.log(clippedValues[quantileCols].values)
    meanLog = np.mean(logValues, axis=1)
    meanLog = np.where(np.isnan(meanLog), -np.inf, meanLog)
    intercepts = np.exp(meanLog)

    # Build final DataFrame
    dataDict = {c.noteIdKey: clippedValues.index.values}
    for col in quantileCols:
      dataDict[col] = clippedValues[col].values
    dataDict[c.internalNoteInterceptKey] = intercepts
    dataDict[c.internalNoteFactor1Key] = clippedValues[c.internalNoteFactor1Key].values
    weightedMean = pd.DataFrame(dataDict)

    return weightedMean

  def _calculate_gaussian_scores(
    self,
    ratingsForTraining: pd.DataFrame,
    crhQuantileRange: np.array,
    crnhQuantileRange: np.array,
    prescoringRaterModelOutput: pd.DataFrame,
    empiricalTotals: Optional[pd.DataFrame] = None,
  ):
    """Calculate Gaussian convolution scores  on the ratingsForTraining data.

    Args:
        ratingsForTraining (pd.DataFrame)
        quantileRange: (np.array)

    Returns:
        noteParams (pd.DataFrame)
    """
    ratingsForTrainingWithFactors = ratingsForTraining.merge(
      prescoringRaterModelOutput[[c.raterParticipantIdKey, c.internalRaterFactor1Key]],
      on=c.raterParticipantIdKey,
    )
    assert (
      ratingsForTrainingWithFactors.shape[0] == ratingsForTraining.shape[0]
    ), "number of ratings changed"
    noteParams = self._gaussian_kernel_extrapolator_vectorized(
      ratingsForTrainingWithFactors, crhQuantileRange, isCrh=True, empiricalTotals=None
    )
    crnhNoteParams = self._gaussian_kernel_extrapolator_vectorized(
      ratingsForTrainingWithFactors, crnhQuantileRange, isCrh=False, empiricalTotals=empiricalTotals
    )
    crnhNoteParams[c.internalNoteInterceptKey] = -crnhNoteParams[c.internalNoteInterceptKey]
    crnhNoteParams.rename(columns={c.internalNoteInterceptKey: "crnhNoteIntercept"}, inplace=True)
    noteParams = noteParams.merge(
      crnhNoteParams[[c.noteIdKey, "crnhNoteIntercept"]], on=c.noteIdKey, how="outer"
    )
    noteParams[c.internalNoteInterceptKey] = np.where(
      np.abs(noteParams[c.internalNoteInterceptKey]) < np.abs(noteParams["crnhNoteIntercept"]),
      noteParams["crnhNoteIntercept"],
      noteParams[c.internalNoteInterceptKey],
    )
    return noteParams[[c.noteIdKey, c.internalNoteInterceptKey, c.internalNoteFactor1Key]]

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
    mfScorer = MFCoreScorer()
    return mfScorer._prescore_notes_and_users(ratings, noteStatusHistory, userEnrollmentRaw)

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
        & (~pd.isna(prescoringRaterModelOutput[c.internalRaterFactor1Key]))
      ),
      [c.raterParticipantIdKey, c.internalRaterFactor1Key],
    ]
    ratingsForTraining = ratingsForTraining.merge(
      ratersWithParams[[c.raterParticipantIdKey]], how="inner", on=c.raterParticipantIdKey
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

      aboveThresholdRaters = prescoringRaterModelOutput.loc[
        prescoringRaterModelOutput[c.aboveHelpfulnessThresholdKey].fillna(False),
        c.raterParticipantIdKey,
      ].values

      ratersWithParams = ratersWithParams.loc[
        ratersWithParams[c.raterParticipantIdKey].isin(aboveThresholdRaters)
      ]
    if self._saveIntermediateState:
      self.finalRoundRatings = finalRoundRatings
    assert ratersWithParams.shape[0] == len(np.unique(ratersWithParams[c.raterParticipantIdKey]))
    if self._calculateBins:
      if (
        ratersWithParams.loc[ratersWithParams[c.internalRaterFactor1Key] < 0][
          c.internalRaterFactor1Key
        ].nunique()
        > self._nBinsEachSide
      ) & (
        ratersWithParams.loc[ratersWithParams[c.internalRaterFactor1Key] > 0][
          c.internalRaterFactor1Key
        ].nunique()
        > self._nBinsEachSide
      ):
        _, l_range = pd.qcut(
          ratersWithParams.loc[ratersWithParams[c.internalRaterFactor1Key] < 0][
            c.internalRaterFactor1Key
          ],
          self._nBinsEachSide,
          retbins=True,
        )
        _, r_range = pd.qcut(
          ratersWithParams.loc[ratersWithParams[c.internalRaterFactor1Key] > 0][
            c.internalRaterFactor1Key
          ],
          self._nBinsEachSide,
          retbins=True,
        )
        lMids = (l_range[:-1] + l_range[1:]) / 2
        rMids = (r_range[:-1] + r_range[1:]) / 2
        mids = (np.array(sorted(abs(lMids))) + np.array(sorted(abs(rMids)))) / 2
        crhQuantileRange = np.concatenate([sorted(-mids), mids])
        crnhQuantileRange = np.concatenate([sorted(-mids), mids])
        logger.info(f"crh quantile range: {crhQuantileRange}")
        logger.info(f"crnh quantile range: {crnhQuantileRange}")
      # if there are not enough unique raters to even calculate bins, do not predict
      else:
        scoredNotes = pd.DataFrame(columns=self.get_internal_scored_notes_cols())
        helpfulnessScores = pd.DataFrame(columns=self.get_internal_helpfulness_scores_cols())

    else:
      crhQuantileRange = c.quantileRange
      crnhQuantileRange = c.smoothedRange
    logger.info(f"crh quantiles {crhQuantileRange}")
    logger.info(f"crnh quantiles {crnhQuantileRange}")
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

    raterParams = prescoringRaterModelOutput[
      [c.raterParticipantIdKey, c.internalRaterInterceptKey, c.internalRaterFactor1Key]
    ]
    assert raterParams.shape[0] == len(
      np.unique(raterParams[c.raterParticipantIdKey])
    ), "duplicate rater ids in prescoring rmo"
    noteParams = self._calculate_gaussian_scores(
      finalRoundRatings, crhQuantileRange, crnhQuantileRange, prescoringRaterModelOutput
    )
    if self._useMfNoteParams:
      with self.time_block("Get Note Factor from MF"):
        mfNoteParams, _, _ = MatrixFactorization(seed=self._seed).run_mf(
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
          run_name="gaussianNoteFactorMF",
        )
      noteParams[c.internalNoteFactor1Key] = noteParams[c.internalNoteFactor1Key].astype(np.float32)
      noteParams.rename(
        columns={c.internalNoteFactor1Key: c.internalNoteFactor1Key + "_gaus"}, inplace=True
      )
      mfNoteParams.rename(
        columns={c.internalNoteFactor1Key: c.internalNoteFactor1Key + "_mf"}, inplace=True
      )
      noteParams = noteParams.merge(
        mfNoteParams[[c.noteIdKey, c.internalNoteFactor1Key + "_mf"]],
        on=c.noteIdKey,
        how="left",
      )
      noteParams[c.internalNoteFactor1Key] = noteParams[
        c.internalNoteFactor1Key + "_mf"
      ].combine_first(noteParams[c.internalNoteFactor1Key + "_gaus"])
      noteParams = noteParams.drop(
        columns=[c.internalNoteFactor1Key + "_mf", c.internalNoteFactor1Key + "_gaus"]
      )

    logger.info(f"noteParams shape {noteParams.shape[0]}")
    if self._saveIntermediateState:
      self.noteParams = noteParams
      self.raterParams = raterParams
      self.finalRoundRatings = finalRoundRatings

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
      helpfulnessScores = prescoringRaterModelOutput[
        [
          c.raterParticipantIdKey,
        ]
      ]

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
      scoringArgs.prescoringNoteModelOutput[c.scorerNameKey] == self.get_prescoring_name()
    ].drop(columns=c.scorerNameKey, inplace=False)

    if scoringArgs.prescoringRaterModelOutput is None:
      return self._return_empty_final_scores()
    prescoringRaterModelOutput = scoringArgs.prescoringRaterModelOutput[
      scoringArgs.prescoringRaterModelOutput[c.scorerNameKey] == self.get_prescoring_name()
    ].drop(columns=c.scorerNameKey, inplace=False)

    if self.get_prescoring_name() not in scoringArgs.prescoringMetaOutput.metaScorerOutput:
      logger.info(
        f"Scorer {self.get_prescoring_name()} not found in prescoringMetaOutput; returning empty scores from final scoring."
      )
      return self._return_empty_final_scores()
    prescoringMetaScorerOutput = scoringArgs.prescoringMetaOutput.metaScorerOutput[
      self.get_prescoring_name()
    ]

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
