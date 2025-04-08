"""Train and apply a model to predict which notes will lose CRH status.

This module implements a supervised model designed to be applied throughout the period
when a enters or be in status stabilization. The model predicts whether the note will ultimately
lock to CRH status or flip back to NMR.  The features used by the model include:
* Note author and timing of the note in relation to the associated post.
* Individual user Helpfulness and tag ratings.
* Ratios of how common each tag is in ratings associated with the note.
* Aggregate statistics about rater factors (e.g. standard deviation of rater factors for
  Helpful ratings, number of {Helpful, Somewhat Helpful, Not Helpful} ratings from users
  with {positive, center, negative} factors, etc.).
* Burstiness of ratings (e.g. how many ratings occur soon after note creation, over short
  periods of time or within the recent past).
* Creation, ratings and scoring outcomes of peer notes associated with the same post.

The training data includes notes that were CRH at some point and have either locked to
CRH or NMR, except for any note that either drifted to NMR status after locking.  Notes
included in training must be associated with a post created after August 19, 2024, which
marks the point when usage of the timestampMillisOfFirstNmrDueToMinStableCrhTime changed
such that the value always reflects the timestamp when a note first entered stabilization
or NaN if the note never entered stabilization.

Since the model is designed to be applied throughout the entire stabilization period we include
each note twice, with features reflecting both when the note enters stabilization and when the
note achieves CRH status.  If the note skips stabilization or exits stabilization without achieving
CRH status, then the note is included twice with identical features to maintain consistent
weighting across notes.

The feature extraction and model training are implemented entirely in scikit-learn.
"""

# Standard libraries
from io import BytesIO
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Project libraries
from . import constants as c
from .enums import Scorers
from .pandas_utils import get_df_fingerprint

# 3rd party libraries
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc as area_under_curve, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
  FunctionTransformer,
  KBinsDiscretizer,
  OneHotEncoder,
  PolynomialFeatures,
)


# Configure logger
logger = logging.getLogger("birdwatch.pflip_plus_model")
logger.setLevel(logging.INFO)

# Exposed constants
LABEL = "LABEL"
CRH = "CRH"
FLIP = "FLIP"

# Internal constants
_MAX = "MAX"
_MIN = "MIN"
_MIN_TWEET_ID = 1825679568688054351

# Internal column names
_RATER_FACTOR = "RATER_FACTOR"
_SCORING_CUTOFF_MTS = "SCORING_CUTOFF_MTS"
_SCORING_CUTOFF_MODE = "SCORING_CUTOFF_MODE"
_USER_HELPFULNESS_RATINGS = "USER_HELPFULNESS_RATINGS"
_USER_HELPFUL_TAGS = "USER_HELPFUL_TAGS"
_USER_NOT_HELPFUL_TAGS = "USER_NOT_HELPFUL_TAGS"
_NEGATIVE = "NEGATIVE"
_NEUTRAL = "NEUTRAL"
_POSITIVE = "POSITIVE"
_BUCKET_COUNT_COLS = [
  f"{viewpoint}_{rating}"
  for viewpoint in [_NEGATIVE, _NEUTRAL, _POSITIVE]
  for rating in [c.helpfulValueTsv, c.somewhatHelpfulValueTsv, c.notHelpfulValueTsv]
]
_RATING_TIME_BUCKETS = [1, 5, 15, 60]
_QUICK_RATING_COLS = [f"FIRST_{cutoff}_TOTAL" for cutoff in _RATING_TIME_BUCKETS] + [
  f"FIRST_{cutoff}_RATIO" for cutoff in _RATING_TIME_BUCKETS
]
_BURST_RATING_COLS = [f"BURST_{cutoff}_TOTAL" for cutoff in _RATING_TIME_BUCKETS] + [
  f"BURST_{cutoff}_RATIO" for cutoff in _RATING_TIME_BUCKETS
]
_RECENT_RATING_COLS = [f"RECENT_{cutoff}_TOTAL" for cutoff in _RATING_TIME_BUCKETS] + [
  f"RECENT_{cutoff}_RATIO" for cutoff in _RATING_TIME_BUCKETS
]
_NOTE_WRITING_LATENCY = "NOTE_WRITING_LATENCY"
_MAX_POS_HELPFUL = "MAX_POS_HELPFUL"
_MAX_NEG_HELPFUL = "MAX_NEG_HELPFUL"
_MEAN_POS_HELPFUL = "MEAN_POS_HELPFUL"
_MEAN_NEG_HELPFUL = "MEAN_NEG_HELPFUL"
_STD_HELPFUL = "STD_HELPFUL"
_MAX_DIFF = "MAX_DIFF"
_MEAN_DIFF = "MEAN_DIFF"
_STATS_COLS = [
  _MAX_POS_HELPFUL,
  _MAX_NEG_HELPFUL,
  _MEAN_POS_HELPFUL,
  _MEAN_NEG_HELPFUL,
  _STD_HELPFUL,
  _MAX_DIFF,
  _MEAN_DIFF,
]
_NOTE_CREATION_MILLIS = "noteCreationMillis"
_TWEET_CREATION_MILLIS = "tweetCreationMillis"
_RATED_ON_NOTE_ID = "ratedOnNoteId"
_LOCAL = "LOCAL"
_PEER_MISLEADING = "PEER_MISLEADING"
_PEER_NON_MISLEADING = "PEER_NON_MISLEADING"
_TOTAL_PEER_NOTES = "TOTAL_PEER_NOTES"
_TOTAL_PEER_MISLEADING_NOTES = "TOTAL_PEER_MISLEADING_NOTES"
_TOTAL_PEER_NON_MISLEADING_NOTES = "TOTAL_PEER_NON_MISLEADING_NOTES"
_TOTAL_PEER_STABILIZATION_NOTES = "TOTAL_PEER_STABILIZATION_NOTES"
_TOTAL_PEER_CRH_NOTES = "TOTAL_PEER_CRH_NOTES"


# Define helper functions at module level so feature extraction pipeline doesn't require
# any lambda functions (and consequently can be pickled.)
def _identity(x: Any) -> Any:
  """Used to create modified preprocessing and tokenization for CountVectorizer."""
  return x


def _feature_log(features: pd.Series) -> pd.Series:
  """Helper to log-scale features while allowing for zero valued features."""
  return np.log(1 + features) / np.log(2)


def _get_timestamp_from_snowflake(snowflake: int) -> int:
  """Helper function to recover a timestamp from a snowflake ID."""
  return (snowflake >> 22) + 1288834974657


def _set_to_list(series: pd.Series) -> List:
  """Helper function to convert a Series containing sets to lists."""
  assert isinstance(series, pd.Series)
  return [list(x) for x in series]


def _fill_na(series: pd.Series) -> np.ndarray:
  return series.fillna(-1).values


def _reshape(series: pd.Series) -> np.ndarray:
  assert isinstance(series, pd.Series)
  return series.values.reshape(-1, 1)


class PFlipPlusModel(object):
  def __init__(
    self,
    ratingRecencyCutoffMinutes: int = 15,
    helpfulnessRaterMin: int = 1,
    tagRaterMin: int = 5,
    helpfulTagPercentile: int = 10,
    notHelpfulTagPercentile: int = 50,
    penalty: str = "l1",
    C: float = 0.04,
    maxIter: int = 2500,
    verbose: int = 0,
    classWeight: Dict[int, int] = {0: 1, 1: 1},
    trainSize: float = 0.9,
    solver: str = "liblinear",
    tol: float = 1e-4,
    seed: Optional[int] = None,
    crhFpRate: float = 0.6,
    tag_ratio_bins: int = 10,
    rating_count_bins: int = 7,
    factor_stats_bins: int = 10,
    burst_bins: int = 5,
    latency_bins: int = 10,
    peer_note_count_bins: int = 4,
  ):
    """Configure PFlipModel.

    Args:
      ratingRecencyCutoffMinutes: Avoiding skew in training requires identifying the time at which
        the note would have been scored in production and which notes / ratings would have been
        available at that time.  ratingRecencyCutoffMinutes specifies the assumed latency from
        rating creation through to when a rating becomes available in final scoring (e.g. if we know
        a note was set CRH at a particular point in time then we require ratings be created at least
        ratingRecencyCutoffMinutes minutes earlier).
    """
    self._pipeline: Optional[Pipeline] = None
    self._predictionThreshold: Optional[float] = None
    self._ratingRecencyCutoff = 1000 * 60 * ratingRecencyCutoffMinutes
    self._helpfulnessRaterMin = helpfulnessRaterMin
    self._tagRaterMin = tagRaterMin
    self._helpfulTagPercentile = helpfulTagPercentile
    self._notHelpfulTagPercentile = notHelpfulTagPercentile
    self._penalty = penalty
    self._C = C
    self._maxIter = maxIter
    self._verbose = verbose
    self._classWeight = classWeight
    self._solver = solver
    self._tol = tol
    self._trainSize = trainSize
    self._seed = seed
    self._crhFpRate = crhFpRate
    self._tag_ratio_bins = tag_ratio_bins
    self._rating_count_bins = rating_count_bins
    self._factor_stats_bins = factor_stats_bins
    self._burst_bins = burst_bins
    self._latency_bins = latency_bins
    self._peer_note_count_bins = peer_note_count_bins
    self._column_thresholds: Dict[str, float] = dict()

  def _get_notes(
    self,
    notes: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
  ):
    """Determine notes to include in scoring and associated metadata.

    To be included, notes must have an associated tweet and classification.  Returned columns
    include: noteId, tweetId, classification, noteCreationMillis, tweetCreationMillis

    Args:
      notes: Input DataFrame containing creation times.
      noteStatusHistory: Input DataFrame containing creation times.
    """
    # Validate that "notes" contains a subset of noteStatusHistory.
    assert notes[c.noteIdKey].nunique() == len(notes), "notes contains duplicate noteIds"
    assert noteStatusHistory[c.noteIdKey].nunique() == len(
      noteStatusHistory
    ), "noteStatusHistory contains duplicate noteIds"
    assert len(notes) == len(
      noteStatusHistory[[c.noteIdKey, c.createdAtMillisKey]].merge(
        notes[[c.noteIdKey, c.createdAtMillisKey]],
        on=[c.noteIdKey, c.createdAtMillisKey],
        how="inner",
      )
    ), "notes is not a subset of noteStatusHistory"
    # Develop set of candidate notes
    candidateNotes = noteStatusHistory[[c.noteIdKey, c.createdAtMillisKey]].rename(
      columns={c.createdAtMillisKey: _NOTE_CREATION_MILLIS}
    )
    candidateNotes = candidateNotes.merge(notes[[c.noteIdKey, c.tweetIdKey, c.classificationKey]])
    candidateNotes[_TWEET_CREATION_MILLIS] = [
      _get_timestamp_from_snowflake(tweetId) for tweetId in candidateNotes[c.tweetIdKey]
    ]
    # Prune candidates to require that associated tweet and classification are available, and possibly
    # that associated tweet is recent.
    candidateNotes = candidateNotes[
      (candidateNotes[c.tweetIdKey] > 0) & (candidateNotes[c.classificationKey].notna())
    ]
    return candidateNotes

  def _compute_scoring_cutoff(
    self, noteStatusHistory: pd.DataFrame, ratings: pd.DataFrame, mode: str, minRatings: int = 5
  ) -> pd.DataFrame:
    """Compute time limits on which ratings to include for each note.

    Recall that some notes may enter stabilization and then later exit to CRH, some may enter stabilization
    and exit to NMR, and some may skip stabilizaiton and go directly to CRH.  We support taking the min
    and max of the two timestamps so that we have a datapoint for notes both at the beginning and end of
    any stabilization period.  For notes that skip stabilization or never go CRH, this means the note is
    included twice with the same features, which is OK since it ensures weight parity with notes that
    are included twice with different features.  We do want to capture notes at both the beginning and
    end of the stabilization period so that the model performs well throughout the entire stabilization
    period.

    Notes that never enter stabilization or go CRH will be dropped because they should not be included
    in training.

    Args:
      noteStatusHistory: pd.DataFrame used to determine time of the first CRH status,
        if applicable.
      mode: whether to return the max or min of timestampMillisOfFirstNmrDueToMinStableCrhTime
        and timestampMillisOfNoteFirstNonNMRLabel

    Returns:
      pd.DataFrame with noteId and STATUS_MTS columns
    """
    # Select columns and prune to notes that either entered stabilization or went CRH.  Note that
    # the goal here is to not set a rating cutoff for notes that went CRNH since the model won't
    # be applied to these notes in prod.
    scoringCutoff = noteStatusHistory[
      [
        c.noteIdKey,
        c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey,
        c.timestampMillisOfNoteFirstNonNMRLabelKey,
        c.firstNonNMRLabelKey,
      ]
    ].copy()
    scoringCutoff = scoringCutoff[
      (scoringCutoff[c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey].notna())
      | (scoringCutoff[c.firstNonNMRLabelKey] == c.currentlyRatedHelpful)
    ].drop(columns=c.firstNonNMRLabelKey)
    # Set cutoff timestamps
    if mode == _MAX:
      cutoffs = scoringCutoff[
        [
          c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey,
          c.timestampMillisOfNoteFirstNonNMRLabelKey,
        ]
      ].max(axis=1)
    else:
      assert mode == _MIN, f"unexpected mode: {mode}"
      cutoffs = scoringCutoff[
        [
          c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey,
          c.timestampMillisOfNoteFirstNonNMRLabelKey,
        ]
      ].min(axis=1)
    # cutoffs doesn't contain any NaN values and has float type, but Pandas generates an
    # error when casting directly to Int64Dtype.  There is no error if cast to int64 first.
    assert cutoffs.dtype == np.float64
    assert cutoffs.isna().sum() == 0
    scoringCutoff[_SCORING_CUTOFF_MTS] = cutoffs.astype(np.int64).astype(pd.Int64Dtype())
    scoringCutoff[_SCORING_CUTOFF_MTS] = (
      scoringCutoff[_SCORING_CUTOFF_MTS] - self._ratingRecencyCutoff
    )
    # Compute cutoff that is guaranteed to include at least 5 ratings
    ratings = ratings[[c.noteIdKey, c.createdAtMillisKey]]
    cutoffByRatings = (
      ratings.groupby(c.noteIdKey)
      .max()
      .reset_index(drop=False)
      .rename(columns={c.createdAtMillisKey: "maxRatingMts"})
      .astype(pd.Int64Dtype())
    )
    cutoffByRatings = cutoffByRatings.merge(
      ratings.sort_values(c.createdAtMillisKey, ascending=True)
      .groupby(c.noteIdKey)
      .nth(minRatings - 1)
      .rename(columns={c.createdAtMillisKey: "nthRatingMts"})
      .astype(pd.Int64Dtype()),
      how="left",
    )
    cutoffByRatings["ratingMin"] = cutoffByRatings[["maxRatingMts", "nthRatingMts"]].min(axis=1)
    # Merge cutoffs by time and by ratings
    beforeMerge = len(scoringCutoff)
    scoringCutoff = scoringCutoff.merge(cutoffByRatings[[c.noteIdKey, "ratingMin"]])
    assert len(scoringCutoff) == beforeMerge
    scoringCutoff[_SCORING_CUTOFF_MTS] = scoringCutoff[[_SCORING_CUTOFF_MTS, "ratingMin"]].max(
      axis=1
    )
    scoringCutoff[_SCORING_CUTOFF_MODE] = mode
    return scoringCutoff[[c.noteIdKey, _SCORING_CUTOFF_MTS, _SCORING_CUTOFF_MODE]]

  def _label_notes(
    self,
    noteStatusHistory: pd.DataFrame,
  ) -> pd.DataFrame:
    """Generate a DataFrame mapping noteIds to labels.

    We define a CRH note as any note that is locked to CRH, and a FLIP note as any note
    that was scored as CRH at some point but has since locked to NMR.  Note that we exclude
    notes that are locked to CRH but decided by ScoringDriftGuard, since that indicates the
    model wanted to score the note as NMR (and therefore it is unclear whether the note is
    best labeled CRH or FLIP).

    Args:
      noteStatusHistory: pd.DataFrame used to determine locked status and whether there was
        a prior CRH status.

    Returns:
      pd.DataFrame with noteId and LABEL columns
    """
    # Assemble relevant data for labeling
    labels = noteStatusHistory[
      [
        c.noteIdKey,
        # If set, implies note was on track to be CRH at some point
        c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey,
        # If set to CRH, implies note was actually CRH at some point
        c.firstNonNMRLabelKey,
        # Use to determine final status and whether note is locked
        c.lockedStatusKey,
        # If set to ScoringDriftGuard, indicates note may be prone to flipping
        c.currentDecidedByKey,
      ]
    ].copy()
    labels[LABEL] = pd.NA
    labels.loc[(labels[c.lockedStatusKey] == c.currentlyRatedHelpful), LABEL] = CRH
    labels.loc[
      (labels[c.firstNonNMRLabelKey] == c.currentlyRatedHelpful)
      & (labels[c.lockedStatusKey].isin({c.needsMoreRatings, c.currentlyRatedNotHelpful})),
      LABEL,
    ] = FLIP
    labels.loc[
      (~labels[c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey].isna())
      & (labels[c.firstNonNMRLabelKey].isna())
      & (labels[c.lockedStatusKey].isin({c.needsMoreRatings, c.currentlyRatedNotHelpful})),
      LABEL,
    ] = FLIP
    labels = labels.dropna(subset=LABEL)
    logger.info(f"labels before ScoringDriftGuard:\n{labels[LABEL].value_counts(dropna=False)}")
    # Note that we don't exclude notes decided by ScoringDriftGuard when a note locks to NMR
    # after being CRH and is now decided by ScoringDriftGuard (implying the note was once again
    # scored as CRH) because in that case the note is labeled as FLIP and the involvement of
    # ScoringDriftGuard only provides further evidence that the note flips status.
    dropRows = (labels[LABEL] == CRH) & (
      np.array(
        [decider.startswith("ScoringDriftGuard") for decider in labels[c.currentDecidedByKey]]
      )
    )
    labels = labels[~dropRows][[c.noteIdKey, LABEL]]
    logger.info(f"labels after ScoringDriftGuard:\n{labels[LABEL].value_counts(dropna=False)}")
    return labels

  def _compute_rater_factors(self, prescoringRaterModelOutput: pd.DataFrame) -> pd.DataFrame:
    """Generate a DataFrame mapping raterParticipantIds to factors.

    Each rater is assigned their factor from either Core, Expansion or ExpansionPlus,
    prioritizing Core, then Expansion, then ExpansionPlus.

    Args:
      prescoringRaterModelOutput: pd.DataFrame used to determine rater factors.

    Returns:
      pd.DataFrame with raterParticipantId and RATER_FACTOR columns
    """
    # Obtain prescoring rater factors
    coreFactors = prescoringRaterModelOutput[
      prescoringRaterModelOutput[c.scorerNameKey] == Scorers.MFCoreScorer.name
    ][[c.raterParticipantIdKey, c.internalRaterFactor1Key]].rename(
      columns={c.internalRaterFactor1Key: c.coreRaterFactor1Key}
    )
    expansionFactors = prescoringRaterModelOutput[
      prescoringRaterModelOutput[c.scorerNameKey] == Scorers.MFExpansionScorer.name
    ][[c.raterParticipantIdKey, c.internalRaterFactor1Key]].rename(
      columns={c.internalRaterFactor1Key: c.expansionRaterFactor1Key}
    )
    expansionPlusFactors = prescoringRaterModelOutput[
      prescoringRaterModelOutput[c.scorerNameKey] == Scorers.MFExpansionPlusScorer.name
    ][[c.raterParticipantIdKey, c.internalRaterFactor1Key]].rename(
      columns={c.internalRaterFactor1Key: c.expansionPlusRaterFactor1Key}
    )
    # Combine and prioritize factors
    raterFactors = coreFactors.merge(expansionFactors, how="outer").merge(
      expansionPlusFactors, how="outer"
    )
    raterFactors[_RATER_FACTOR] = raterFactors[c.expansionPlusRaterFactor1Key]
    raterFactors.loc[
      ~raterFactors[c.expansionRaterFactor1Key].isna(), _RATER_FACTOR
    ] = raterFactors[c.expansionRaterFactor1Key]
    raterFactors.loc[~raterFactors[c.coreRaterFactor1Key].isna(), _RATER_FACTOR] = raterFactors[
      c.coreRaterFactor1Key
    ]
    return raterFactors[[c.raterParticipantIdKey, _RATER_FACTOR]]

  def _prepare_local_ratings(
    self,
    ratings: pd.DataFrame,
    scoredNotes: pd.DataFrame,
  ) -> pd.DataFrame:
    """Filter ratings DF to only include ratings on the notes being scored.

    Args:
      ratings: pd.DataFrame containing all ratings used in scoring
      scoredNotes: pd.DataFrame specifying which notes are actually being scored
    """
    return ratings.merge(scoredNotes[[c.noteIdKey]])

  def _prepare_peer_ratings(
    self, ratings: pd.DataFrame, notes: pd.DataFrame, scoredNotes: pd.DataFrame
  ) -> pd.DataFrame:
    """Construct a ratings DF that captures ratings on peer notes.

    Peer notes are defined as different notes that are on the same post.  We consider the
    ratings for the peer note to provide signal for the note being scored (e.g. if another
    note received many "note not needed" ratings, then it is more likely that the post
    itself does not need a note).

    Since each unique {note, peer} pair is a row in the output dataframe, the column name
    for the note being scored remains "noteId" and the note which originally received the
    rating is the "ratedOnNoteId".

    Args:
      ratings: pd.DataFrame containing all ratings used in scoring
      notes: pd.DataFrame relating notes to posts
      scoredNotes: pd.DataFrame specifying which notes are actually being scored
    """
    # Augment ratings with tweetId and classification
    beforeMerge = len(ratings)
    ratings = ratings.merge(notes[[c.noteIdKey, c.tweetIdKey, c.classificationKey]])
    assert len(ratings) == beforeMerge
    ratings = ratings.rename(columns={c.noteIdKey: _RATED_ON_NOTE_ID})
    # Create one copy of each rating for each scored note on the post, excluding
    # ratings that occur on that post itself
    assert len(scoredNotes) == len(scoredNotes[[c.noteIdKey, c.tweetIdKey]].drop_duplicates())
    ratings = ratings.merge(scoredNotes[[c.noteIdKey, c.tweetIdKey]], on=c.tweetIdKey)
    ratings = ratings[
      ratings[c.noteIdKey] != ratings[_RATED_ON_NOTE_ID]
    ]  # throw out ratings that occur on the note itself
    ratings = ratings.drop(columns={c.tweetIdKey})
    return ratings

  def _apply_cutoff(self, ratings: pd.DataFrame, scoredNotes: pd.DataFrame) -> pd.DataFrame:
    """Filter a ratings DF to only contain ratings before an applicable cutoff.

    Args:
      ratings: pd.DataFrame containing peer and/or local ratings.
      scoredNotes: pd.DataFrame specifying scoring cutoff timestamps
    """
    assert scoredNotes[_SCORING_CUTOFF_MTS].isna().sum() == 0
    beforeMerge = len(ratings)
    ratings = ratings.merge(scoredNotes[[c.noteIdKey, _SCORING_CUTOFF_MTS]])
    assert len(ratings) == beforeMerge
    return ratings[ratings[c.createdAtMillisKey] <= ratings[_SCORING_CUTOFF_MTS]].drop(
      columns=_SCORING_CUTOFF_MTS
    )

  def _get_note_writing_latency(self, notes: pd.DataFrame) -> pd.DataFrame:
    """Identify the time in milliseconds from post to note creation."""
    notes = notes[[c.noteIdKey, _NOTE_CREATION_MILLIS, _TWEET_CREATION_MILLIS]].copy()
    notes[_NOTE_WRITING_LATENCY] = notes[_NOTE_CREATION_MILLIS] - notes[_TWEET_CREATION_MILLIS]
    return notes[[c.noteIdKey, _NOTE_WRITING_LATENCY]]

  def _get_quick_rating_stats(self, notes: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Return counts and ratios of how many ratings occurred in the first 1/5/15/60 minutes.

    Args:
      notes: DF specifying note creation timestamps.
      ratings: DF specifying local rating timestamps.
    """
    ratingTotals = (
      ratings[[c.noteIdKey]]
      .value_counts()
      .to_frame()
      .reset_index(drop=False)
      .rename(columns={"count": "total"})
    )
    ratingTotals = notes[[c.noteIdKey]].merge(ratingTotals, how="left")
    ratingTotals = ratingTotals.fillna({"total": 0}).astype(pd.Int64Dtype())
    for cutoff in _RATING_TIME_BUCKETS:
      beforeCutoff = ratings[[c.noteIdKey, c.createdAtMillisKey]].rename(
        columns={c.createdAtMillisKey: "ratingCreationMts"}
      )
      beforeCutoff = beforeCutoff.merge(notes[[c.noteIdKey, _NOTE_CREATION_MILLIS]])
      beforeCutoff = beforeCutoff[
        beforeCutoff["ratingCreationMts"]
        < (beforeCutoff[_NOTE_CREATION_MILLIS] + (1000 * 60 * cutoff))
      ]
      cutoffCount = (
        beforeCutoff[[c.noteIdKey]]
        .value_counts()
        .to_frame()
        .reset_index(drop=False)
        .rename(columns={"count": f"FIRST_{cutoff}_TOTAL"})
      )
      ratingTotals = ratingTotals.merge(cutoffCount, how="left").fillna(0)
    ratingTotals = ratingTotals.astype(pd.Int64Dtype())
    for cutoff in _RATING_TIME_BUCKETS:
      ratingTotals[f"FIRST_{cutoff}_RATIO"] = ratingTotals[f"FIRST_{cutoff}_TOTAL"] / (
        ratingTotals["total"].clip(lower=1)
      )
    return ratingTotals[[c.noteIdKey] + _QUICK_RATING_COLS]

  def _get_burst_rating_stats(self, notes: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Return counts and ratios of the max ratings in 1/5/15/60 minute windows.

    Args:
      ratings: DF specifying local rating timestamps.
    """
    ratingTotals = (
      ratings[[c.noteIdKey]]
      .value_counts()
      .to_frame()
      .reset_index(drop=False)
      .rename(columns={"count": "total"})
    )
    initialNotes = len(notes)
    ratingTotals = notes[[c.noteIdKey]].merge(ratingTotals, how="left")
    ratingTotals = ratingTotals.fillna({"total": 0}).astype(pd.Int64Dtype())
    for cutoff in _RATING_TIME_BUCKETS:
      ratingCounts = []
      for offset in range(cutoff):
        offsetRatings = ratings[[c.noteIdKey, c.createdAtMillisKey]].copy()
        offsetRatings[c.createdAtMillisKey] = (
          offsetRatings[c.createdAtMillisKey] + (1000 * 60 * offset)
        ) // (1000 * 60 * cutoff)
        offsetRatings = (
          offsetRatings.value_counts()
          .to_frame()
          .reset_index(drop=False)[[c.noteIdKey, "count"]]
          .groupby(c.noteIdKey)
          .max()
          .reset_index(drop=False)
        )
        ratingCounts.append(offsetRatings)
      ratingCounts = (
        pd.concat(ratingCounts)
        .groupby(c.noteIdKey)
        .max()
        .reset_index(drop=False)
        .rename(columns={"count": f"BURST_{cutoff}_TOTAL"})
      ).astype(pd.Int64Dtype())
      ratingTotals = ratingTotals.merge(ratingCounts, how="left").fillna(
        {f"BURST_{cutoff}_TOTAL": 0}
      )
      ratingTotals[f"BURST_{cutoff}_RATIO"] = ratingTotals[f"BURST_{cutoff}_TOTAL"] / (
        ratingTotals["total"].clip(lower=1)
      )
    assert (
      len(ratingTotals) == initialNotes
    ), f"unexpected length mismatch: {len(ratingTotals)} vs. {initialNotes}"
    return ratingTotals[[c.noteIdKey] + _BURST_RATING_COLS]

  def _get_recent_rating_stats(
    self, scoredNotes: pd.DataFrame, ratings: pd.DataFrame, prepareForTraining: bool
  ):
    """Generate counts of ratings within the last 1/5/15/20 minutes.

    Note that this process must work differently during training and in production, since
    during training future ratings will be available.  We solve this by treating the
    _SCORING_CUTOFF_MTS as the effective time at which scoring occurred, and include ratings
    based on that timestamp.

    Args:
      scoredNotes: pd.DataFrame specifying scoring cutoff timestamps
      ratings: pd.DataFrame containing local ratings.
      prepareForTraining: bool specifying whether to prune ratings.
    """
    # Define notion of effective present for each rating
    initialRatings = len(ratings)
    ratings = ratings[[c.noteIdKey, c.createdAtMillisKey]].copy()
    if prepareForTraining:
      ratings = ratings.merge(
        scoredNotes[[c.noteIdKey, _SCORING_CUTOFF_MTS]].rename(
          columns={_SCORING_CUTOFF_MTS: "effectivePresent"}
        )
      )
    else:
      ratings["effectivePresent"] = ratings[c.createdAtMillisKey].max()
    assert len(ratings) == initialRatings
    assert (ratings[c.createdAtMillisKey] > ratings["effectivePresent"]).sum() == 0
    assert ratings[c.createdAtMillisKey].isna().sum() == 0
    assert ratings["effectivePresent"].isna().sum() == 0
    # Develop counts and ratios of recent ratings in specific time ranges
    ratingTotals = (
      ratings[[c.noteIdKey]]
      .value_counts()
      .to_frame()
      .reset_index(drop=False)
      .rename(columns={"count": "total"})
    )
    ratingTotals = scoredNotes[[c.noteIdKey]].merge(ratingTotals, how="left")
    ratingTotals = ratingTotals.fillna({"total": 0}).astype(pd.Int64Dtype())
    for cutoff in _RATING_TIME_BUCKETS:
      afterCutoff = ratings[
        ratings[c.createdAtMillisKey] > (ratings["effectivePresent"] - (1000 * 60 * cutoff))
      ]
      cutoffCount = (
        afterCutoff[[c.noteIdKey]]
        .value_counts()
        .to_frame()
        .reset_index(drop=False)
        .rename(columns={"count": f"RECENT_{cutoff}_TOTAL"})
      )
      ratingTotals = ratingTotals.merge(cutoffCount, how="left").fillna(0)
    ratingTotals = ratingTotals.astype(pd.Int64Dtype())
    for cutoff in _RATING_TIME_BUCKETS:
      ratingTotals[f"RECENT_{cutoff}_RATIO"] = ratingTotals[f"RECENT_{cutoff}_TOTAL"] / (
        ratingTotals["total"].clip(lower=1)
      )
    return ratingTotals[[c.noteIdKey] + _RECENT_RATING_COLS]

  def _get_helpfulness_ratings(self, notes: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one row per note and a column with a nested list of helpfulness ratings.

    Args:
      notes: pd.DataFrame used to determine full set of notes
      ratings: pd.DataFrame containing ratings that should contribute to model features.

    Returns:
      pd.DataFrame with two columns: noteIds and all user helpfulness ratings on the note.
    """
    raters = ratings[c.raterParticipantIdKey].astype(str)
    helpfulnessRatings = ratings[[c.noteIdKey]].copy()
    helpfulnessRatings.loc[:, _USER_HELPFULNESS_RATINGS] = (
      raters + ":" + ratings[c.helpfulnessLevelKey].astype(str)
    )
    helpfulnessRatings = (
      helpfulnessRatings[[c.noteIdKey, _USER_HELPFULNESS_RATINGS]]
      .groupby(c.noteIdKey)
      .agg(set)
      .reset_index(drop=False)
    )
    helpfulnessRatings = notes.merge(helpfulnessRatings, how="left")
    helpfulnessRatings[_USER_HELPFULNESS_RATINGS] = helpfulnessRatings[
      _USER_HELPFULNESS_RATINGS
    ].apply(lambda d: d if isinstance(d, set) else {})
    return helpfulnessRatings

  def _get_tag_ratios(self, notes: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Produce a DataFrame specifying the ratio of ratings for each note that included each tag.

    Args:
      notes: pd.DataFrame used to specify the universe of all notes to include.
      ratings: pd.DataFrame containing all ratings for feature extraction.

    Returns:
      pd.DataFrame containing one row per note and one column per rating tag.
    """
    tags = ratings[[c.noteIdKey] + c.helpfulTagsTSVOrder + c.notHelpfulTagsTSVOrder].copy()
    total_ratings = "total_ratings"
    tags[total_ratings] = 1
    tags = tags.groupby(c.noteIdKey).sum().reset_index(drop=False)
    tags[c.helpfulTagsTSVOrder + c.notHelpfulTagsTSVOrder] = tags[
      c.helpfulTagsTSVOrder + c.notHelpfulTagsTSVOrder
    ].divide(tags[total_ratings], axis=0)
    tags = notes[[c.noteIdKey]].merge(tags.drop(columns=total_ratings), how="left")
    return tags[[c.noteIdKey] + c.helpfulTagsTSVOrder + c.notHelpfulTagsTSVOrder]

  def _get_user_tag_ratings(
    self,
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
    outCol: str,
    tagCols: List[str],
  ) -> pd.DataFrame:
    """Return a DataFrame with one row per note and a column with a nested list of user rating tags.

    Args:
      notes: pd.DataFrame used to specify the universe of all notes to include.
      ratings: pd.DataFrame containing all ratings for feature extraction.
      outCol: str identifying output column to contain list of user tag ratings
      tagCols: List of tag columns to include in outCol

    Returns:
      pd.DataFrame containing one row per note and one column containing all user rating tags.
    """
    ratingTags = ratings[[c.noteIdKey, c.raterParticipantIdKey] + tagCols].copy()
    tagStrs = np.array(tagCols)
    ratingTags[outCol] = [
      {f"{rater}:{tag}" for tag in tagStrs[np.where(row)[0]]}
      for rater, row in zip(ratingTags[c.raterParticipantIdKey], ratingTags[tagCols].values)
    ]
    ratingTags = (
      ratingTags[[c.noteIdKey, outCol]]
      .groupby(c.noteIdKey)
      .agg(lambda x: set().union(*x))
      .reset_index(drop=False)
    )
    ratingTags = notes[[c.noteIdKey]].merge(ratingTags, how="left")
    ratingTags[outCol] = ratingTags[outCol].apply(
      lambda userTags: userTags if isinstance(userTags, set) else {}
    )
    return ratingTags

  def _get_bucket_count_totals(self, notes: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame with one row per note and 9 columns containing buckets of rating counts.

    Args:
      notes: pd.DataFrame used to specify the universe of all notes to include.
      ratings: pd.DataFrame containing all ratings for feature extraction.

    Returns:
      pd.DataFrame containing one row per note and one column containing all user rating tags.
    """
    summary = ratings[[c.noteIdKey, _RATER_FACTOR, c.helpfulnessLevelKey]].copy()
    summary = summary[~summary[_RATER_FACTOR].isna()]
    summary[_NEGATIVE] = summary[_RATER_FACTOR] < -0.3
    summary[_NEUTRAL] = (summary[_RATER_FACTOR] >= -0.3) & (summary[_RATER_FACTOR] <= 0.3)
    summary[_POSITIVE] = summary[_RATER_FACTOR] > 0.3
    summary[c.helpfulValueTsv] = summary[c.helpfulnessLevelKey] == c.helpfulValueTsv
    summary[c.somewhatHelpfulValueTsv] = summary[c.helpfulnessLevelKey] == c.somewhatHelpfulValueTsv
    summary[c.notHelpfulValueTsv] = summary[c.helpfulnessLevelKey] == c.notHelpfulValueTsv
    for viewpoint in [_NEGATIVE, _NEUTRAL, _POSITIVE]:
      for rating in [c.helpfulValueTsv, c.somewhatHelpfulValueTsv, c.notHelpfulValueTsv]:
        summary[f"{viewpoint}_{rating}"] = summary[viewpoint].multiply(summary[rating])
    summary = summary[[c.noteIdKey] + _BUCKET_COUNT_COLS]
    summary = summary.groupby(c.noteIdKey).sum().reset_index(drop=False)
    summary[_BUCKET_COUNT_COLS] = summary[_BUCKET_COUNT_COLS].astype(np.float64)
    summary = (
      notes[[c.noteIdKey]]
      .merge(summary, on=c.noteIdKey, how="left")
      .fillna(0.0)
      .astype(pd.Int64Dtype())
    )
    return summary

  def _get_helpful_rating_stats(self, notes: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate statistics about the Helpful ratings on a note.

    Args:
      notes: pd.DataFrame used to specify the universe of all notes to include.
      ratings: pd.DataFrame containing all ratings for feature extraction.

    Returns:
      pd.DataFrame with one row per note and 7 columns for aggregate statistics about rater
        factors for Helpful ratings.
    """
    # Prune ratings to only include Helpful ratings from users with a factor
    ratings = ratings[[c.noteIdKey, _RATER_FACTOR, c.helpfulnessLevelKey]].copy()
    ratings = ratings[~ratings[_RATER_FACTOR].isna()]
    ratings = ratings[ratings[c.helpfulnessLevelKey] == c.helpfulValueTsv]
    ratings = ratings.drop(columns=c.helpfulnessLevelKey)
    # Compute rating stats
    maxPosHelpful = (
      ratings[ratings[_RATER_FACTOR] > 0]
      .groupby(c.noteIdKey)
      .max()
      .reset_index()
      .rename(columns={_RATER_FACTOR: _MAX_POS_HELPFUL})
    )
    maxNegHelpful = (
      ratings[ratings[_RATER_FACTOR] <= 0]
      .groupby(c.noteIdKey)
      .min()
      .abs()
      .reset_index()
      .rename(columns={_RATER_FACTOR: _MAX_NEG_HELPFUL})
    )
    meanPosHelpful = (
      ratings[ratings[_RATER_FACTOR] > 0]
      .groupby(c.noteIdKey)
      .mean()
      .reset_index()
      .rename(columns={_RATER_FACTOR: _MEAN_POS_HELPFUL})
    )
    meanNegHelpful = (
      ratings[ratings[_RATER_FACTOR] <= 0]
      .groupby(c.noteIdKey)
      .mean()
      .abs()
      .reset_index()
      .rename(columns={_RATER_FACTOR: _MEAN_NEG_HELPFUL})
    )
    stdHelpful = (
      ratings.groupby(c.noteIdKey).std().reset_index().rename(columns={_RATER_FACTOR: _STD_HELPFUL})
    )
    # Compile into features per-note
    notes = notes[[c.noteIdKey]].merge(maxPosHelpful, on=c.noteIdKey, how="left")
    notes = notes.merge(maxNegHelpful, on=c.noteIdKey, how="left")
    notes = notes.merge(meanPosHelpful, on=c.noteIdKey, how="left")
    notes = notes.merge(meanNegHelpful, on=c.noteIdKey, how="left")
    notes = notes.merge(stdHelpful, on=c.noteIdKey, how="left")
    notes[_MAX_DIFF] = notes[_MAX_POS_HELPFUL] + notes[_MAX_NEG_HELPFUL]
    notes[_MEAN_DIFF] = notes[_MEAN_POS_HELPFUL] + notes[_MEAN_NEG_HELPFUL]
    return notes

  def _make_note_info_from_ratings(
    self,
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
  ) -> pd.DataFrame:
    """Generate features derived from peer or local ratings associated with a note.

    Generated features include user helpfulness and tag ratings, buckets of rating counts,
    stats about rater factor distributions and ratios of tags across ratings.

    Args:
      notes: DF specifying which notes should be included in the output
      ratings: DF containing peer or local ratings to base the features on
    """
    # Augment notes with features.  Note that attributes of the note (e.g. author,
    # creation time) should always be available because we filter to notes with the creation time
    # in the last year, inherently removing any deleted notes where the creation time is unavailable.
    helpfulnessRatings = self._get_helpfulness_ratings(notes[[c.noteIdKey]], ratings)
    notes = notes.merge(helpfulnessRatings, how="inner")
    userHelpfulTags = self._get_user_tag_ratings(
      notes[[c.noteIdKey]], ratings, _USER_HELPFUL_TAGS, c.helpfulTagsTSVOrder
    )
    notes = notes.merge(userHelpfulTags, how="inner")
    userNotHelpfulTags = self._get_user_tag_ratings(
      notes[[c.noteIdKey]], ratings, _USER_NOT_HELPFUL_TAGS, c.notHelpfulTagsTSVOrder
    )
    notes = notes.merge(userNotHelpfulTags, how="inner")
    bucketCounts = self._get_bucket_count_totals(notes[[c.noteIdKey]], ratings)
    notes = notes.merge(bucketCounts, how="inner")
    helpfulStats = self._get_helpful_rating_stats(notes[[c.noteIdKey]], ratings)
    notes = notes.merge(helpfulStats, how="inner")
    tagRatios = self._get_tag_ratios(notes[[c.noteIdKey]], ratings)
    notes = notes.merge(tagRatios, how="inner")
    return notes

  def _get_note_counts(
    self,
    scoredNotes: pd.DataFrame,
    notes: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    prepareForTraining: bool,
  ) -> pd.DataFrame:
    """Return counts of relevant events on peer notes.

    Counts should include: number of misleading and non-misleading notes created on the same post,
    number of misleading notes on the same post that went CRH and ratios of the above values.

    Note that aggregation works differently during training and in production, since training
    effectively has access to data from the future.


    Args:
      scoredNotes: DF containing {note, tweet} associations and effective scoring times for training.
      notes: DF containing set of all notes, not just those that are being scored.
      noteStatusHistory: DF specifying key scoring timestamps for peer notes.
      prepareForTraining: whether notes should be pruned to discard future data.
    """
    # Merge inputs into a single dataframe with relevant info
    assert scoredNotes[c.tweetIdKey].min() > 0  # tweet should be set for all notes being scored
    peerNotes = scoredNotes[[c.noteIdKey, c.tweetIdKey]].merge(
      notes[[c.tweetIdKey, c.noteIdKey, c.classificationKey]].rename(
        columns={c.noteIdKey: "peerNoteId"}
      )
    )
    assert len(scoredNotes) == peerNotes[c.noteIdKey].nunique()
    assert peerNotes[c.classificationKey].isna().sum() == 0
    peerNotes = peerNotes.merge(noteStatusHistory[[c.noteIdKey, c.createdAtMillisKey]])
    peerNotes = peerNotes.merge(
      noteStatusHistory[
        [
          c.noteIdKey,
          c.createdAtMillisKey,
          c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey,
          c.timestampMillisOfNoteFirstNonNMRLabelKey,
          c.firstNonNMRLabelKey,
        ]
      ].rename(columns={c.noteIdKey: "peerNoteId", c.createdAtMillisKey: "peerCreatedAtMillis"})
    )
    assert len(scoredNotes) == peerNotes[c.noteIdKey].nunique()

    # If we are in training, prune scope of what info should be available
    assert (_SCORING_CUTOFF_MTS in scoredNotes) == prepareForTraining
    if _SCORING_CUTOFF_MTS in scoredNotes:
      peerNotes = peerNotes.merge(scoredNotes[[c.noteIdKey, _SCORING_CUTOFF_MTS]])
      # Prune any notes created after the cutoff
      peerNotes = peerNotes[peerNotes["peerCreatedAtMillis"] <= peerNotes[_SCORING_CUTOFF_MTS]]
      # Null any scoring events that happened after the cutoff
      peerNotes.loc[
        peerNotes[c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey]
        > peerNotes[_SCORING_CUTOFF_MTS],
        c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey,
      ] = np.nan
      peerNotes.loc[
        peerNotes[c.timestampMillisOfNoteFirstNonNMRLabelKey] > peerNotes[_SCORING_CUTOFF_MTS],
        [c.timestampMillisOfNoteFirstNonNMRLabelKey, c.firstNonNMRLabelKey],
      ] = np.nan

    # Compute totals
    peerNotes = peerNotes[peerNotes[c.noteIdKey] != peerNotes["peerNoteId"]]
    totalPeerNotes = (
      peerNotes[c.noteIdKey]
      .value_counts()
      .to_frame()
      .reset_index(drop=False)
      .rename(columns={"count": _TOTAL_PEER_NOTES})
    )
    totalPeerMisleadingNotes = (
      peerNotes[peerNotes[c.classificationKey] == c.notesSaysTweetIsMisleadingKey][c.noteIdKey]
      .value_counts()
      .to_frame()
      .reset_index(drop=False)
      .rename(columns={"count": _TOTAL_PEER_MISLEADING_NOTES})
    )
    totalPeerNotMisleadingNotes = (
      peerNotes[peerNotes[c.classificationKey] == c.noteSaysTweetIsNotMisleadingKey][c.noteIdKey]
      .value_counts()
      .to_frame()
      .reset_index(drop=False)
      .rename(columns={"count": _TOTAL_PEER_NON_MISLEADING_NOTES})
    )
    totalPeerStabilizationNotes = (
      peerNotes[
        (peerNotes[c.classificationKey] == c.notesSaysTweetIsMisleadingKey)
        & (peerNotes[c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey].notna())
      ][c.noteIdKey]
      .value_counts()
      .to_frame()
      .reset_index(drop=False)
      .rename(columns={"count": _TOTAL_PEER_STABILIZATION_NOTES})
    )
    totalPeerCrhNotes = (
      peerNotes[
        (peerNotes[c.classificationKey] == c.notesSaysTweetIsMisleadingKey)
        & (peerNotes[c.firstNonNMRLabelKey] == c.currentlyRatedHelpful)
      ][c.noteIdKey]
      .value_counts()
      .to_frame()
      .reset_index(drop=False)
      .rename(columns={"count": _TOTAL_PEER_CRH_NOTES})
    )
    return (
      scoredNotes[[c.noteIdKey]]
      .merge(totalPeerNotes, how="left")
      .merge(totalPeerMisleadingNotes, how="left")
      .merge(totalPeerNotMisleadingNotes, how="left")
      .merge(totalPeerStabilizationNotes, how="left")
      .merge(totalPeerCrhNotes, how="left")
      .fillna(0)
      .astype(pd.Int64Dtype())
    )

  def _prepare_note_info(
    self,
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    prescoringRaterModelOutput: pd.DataFrame,
    prepareForTraining: bool,
    cutoff: Optional[str],
  ) -> pd.DataFrame:
    """Generate a DataFrame with one row per note containing all feature information.

    Note that some columns contain list values (e.g. a single column contains all
    Helpfulness ratings, where each rating is a unique string containing the rater ID
    and their Helpfulness rating).

    Args:
      notes: pd.DataFrame
      ratings: pd.DataFrame
      noteStatusHistory: pd.DataFrame
      prescoringRaterModelOutput: pd.DataFrame
      prepareForTraining: True if ratings should be filtered to discard data after the
        point of scoring to avoid skew.
      cutoff: Whether to prune ratings to those available when a note enters stabilization
        or gains CRH status.  None if prepareForTraining=False.

    Returns:
      pd.DataFrame containing all feature information with one row per note.
    """
    # Validate and normalize types
    assert ((prepareForTraining == False) & (cutoff is None)) | (
      prepareForTraining & (cutoff in {_MIN, _MAX})
    )
    notes[c.tweetIdKey] = notes[c.tweetIdKey].astype(pd.Int64Dtype())
    noteStatusHistory[c.createdAtMillisKey] = noteStatusHistory[c.createdAtMillisKey].astype(
      pd.Int64Dtype()
    )

    # Prep notes
    scoredNotes = self._get_notes(notes, noteStatusHistory)
    if prepareForTraining:
      assert cutoff is not None
      # Prune to recent notes
      scoredNotes = scoredNotes[scoredNotes[c.tweetIdKey] > _MIN_TWEET_ID]
      # Validate that stabilization timestamps are valid
      assert (
        noteStatusHistory[[c.noteIdKey, c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey]].merge(
          scoredNotes[[c.noteIdKey]]
        )[c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey]
        < 0
      ).sum() == 0
      # Compute flip labels
      labels = self._label_notes(noteStatusHistory.merge(scoredNotes[[c.noteIdKey]]))
      # Compute scoring cutoffs based on when the note entered and left stabilization
      scoringCutoff = self._compute_scoring_cutoff(
        noteStatusHistory.merge(scoredNotes[[c.noteIdKey]]),
        ratings.merge(scoredNotes[[c.noteIdKey]]),
        cutoff,
      )
      # Validate and merge data, effectively pruning to notes that have a label
      scoredNotes = scoredNotes.merge(scoringCutoff, on=c.noteIdKey)
      assert len(scoredNotes) == len(scoringCutoff)
      assert len(labels) == len(
        scoringCutoff.merge(labels)
      )  # labels should be a subset of scoringCutoff
      scoredNotes = scoredNotes.merge(labels)
      assert len(scoredNotes) == len(labels)
    totalScoredNotes = len(scoredNotes)

    # Prep ratings
    # Prune ratings to only include scored notes and other notes on the same post
    assert scoredNotes[c.tweetIdKey].min() > 0  # tweet should be set for all notes being scored
    adjacentNotes = notes[[c.noteIdKey, c.tweetIdKey]].merge(
      scoredNotes[[c.tweetIdKey]].drop_duplicates()
    )[[c.noteIdKey]]
    assert len(adjacentNotes) == adjacentNotes[c.noteIdKey].nunique()
    ratings = ratings.merge(adjacentNotes)
    assert len(ratings) == len(ratings[[c.noteIdKey, c.raterParticipantIdKey]].drop_duplicates())
    # Associate rater factors
    raterFactors = self._compute_rater_factors(prescoringRaterModelOutput)
    assert len(raterFactors) == raterFactors[c.raterParticipantIdKey].nunique()
    ratings = ratings.merge(raterFactors, how="left")
    # Generate rating datasets for self, peer misleading and peer non-misleading notes
    localRatings = self._prepare_local_ratings(ratings, scoredNotes[[c.noteIdKey]])
    peerRatings = self._prepare_peer_ratings(
      ratings,
      notes[[c.noteIdKey, c.tweetIdKey, c.classificationKey]],
      scoredNotes[[c.noteIdKey, c.tweetIdKey]],
    )
    peerMisleadingRatings = peerRatings[
      peerRatings[c.classificationKey] == c.notesSaysTweetIsMisleadingKey
    ]
    peerNonMisleadingRatings = peerRatings[
      peerRatings[c.classificationKey] == c.noteSaysTweetIsNotMisleadingKey
    ]
    if prepareForTraining:
      localRatings = self._apply_cutoff(localRatings, scoredNotes)
      peerMisleadingRatings = self._apply_cutoff(peerMisleadingRatings, scoredNotes)
      peerNonMisleadingRatings = self._apply_cutoff(peerNonMisleadingRatings, scoredNotes)

    # Extract featuers
    # Generate features that depend on self ratings only
    writingLatency = self._get_note_writing_latency(
      scoredNotes[[c.noteIdKey, _TWEET_CREATION_MILLIS, _NOTE_CREATION_MILLIS]]
    )
    scoredNotes = scoredNotes.merge(writingLatency, how="inner")
    noteAuthors = noteStatusHistory[[c.noteIdKey, c.noteAuthorParticipantIdKey]]
    scoredNotes = scoredNotes.merge(noteAuthors, how="inner")
    quickRatings = self._get_quick_rating_stats(
      scoredNotes[[c.noteIdKey, _NOTE_CREATION_MILLIS]], localRatings
    )
    scoredNotes = scoredNotes.merge(quickRatings, how="inner")
    burstRatings = self._get_burst_rating_stats(
      scoredNotes[[c.noteIdKey]], localRatings[[c.noteIdKey, c.createdAtMillisKey]]
    )
    scoredNotes = scoredNotes.merge(burstRatings, how="inner")
    recentRatings = self._get_recent_rating_stats(
      scoredNotes, localRatings[[c.noteIdKey, c.createdAtMillisKey]], prepareForTraining
    )
    scoredNotes = scoredNotes.merge(recentRatings, how="inner")
    peerNoteCounts = self._get_note_counts(
      scoredNotes, notes, noteStatusHistory, prepareForTraining
    )
    scoredNotes = scoredNotes.merge(peerNoteCounts, how="inner")
    # Generate features based on self and peer ratings
    for df, prefix in [
      (localRatings, _LOCAL),
      (peerMisleadingRatings, _PEER_MISLEADING),
      (peerNonMisleadingRatings, _PEER_NON_MISLEADING),
    ]:
      features = self._make_note_info_from_ratings(scoredNotes, df)
      overlapCols = (set(scoredNotes.columns) & set(features.columns)) - {c.noteIdKey}
      features = features[[col for col in features.columns if col not in overlapCols]]
      features = features.rename(
        columns={col: f"{prefix}_{col}" for col in features if col != c.noteIdKey}
      )
      scoredNotes = scoredNotes.merge(features, how="left")

    # Merge rating totals for debugging / info
    scoredNotes = scoredNotes.merge(
      localRatings[[c.noteIdKey]]
      .value_counts()
      .to_frame()
      .reset_index(drop=False)
      .rename(columns={"count": _LOCAL}),
      how="left",
    )
    scoredNotes = scoredNotes.merge(
      peerMisleadingRatings[[c.noteIdKey]]
      .value_counts()
      .to_frame()
      .reset_index(drop=False)
      .rename(columns={"count": _PEER_MISLEADING}),
      how="left",
    )
    scoredNotes = scoredNotes.merge(
      peerNonMisleadingRatings[[c.noteIdKey]]
      .value_counts()
      .to_frame()
      .reset_index(drop=False)
      .rename(columns={"count": _PEER_NON_MISLEADING}),
      how="left",
    )
    scoredNotes = scoredNotes.fillna(
      {_LOCAL: 0, _PEER_MISLEADING: 0, _PEER_NON_MISLEADING: 0}
    ).astype(
      {
        _LOCAL: pd.Int64Dtype(),
        _PEER_MISLEADING: pd.Int64Dtype(),
        _PEER_NON_MISLEADING: pd.Int64Dtype(),
      }
    )
    assert len(scoredNotes) == totalScoredNotes, f"{len(scoredNotes)} vs {totalScoredNotes}"
    return scoredNotes

  def _get_feature_pipeline(self, noteInfo: pd.DataFrame) -> Pipeline:
    # Begin with author pipeline
    columnPipes: List[Tuple[str, Any, Union[str, List[str]]]] = [
      (
        c.noteAuthorParticipantIdKey,
        Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]),
        [c.noteAuthorParticipantIdKey],
      )
    ]
    # Add pipelines for individual user helpfulness ratings
    for prefix in [_LOCAL, _PEER_MISLEADING, _PEER_NON_MISLEADING]:
      column = f"{prefix}_{_USER_HELPFULNESS_RATINGS}"
      pipe = Pipeline(
        [
          ("_set_to_list", FunctionTransformer(_set_to_list)),
          (
            "onehot",
            CountVectorizer(
              tokenizer=_identity, preprocessor=_identity, min_df=self._helpfulnessRaterMin
            ),
          ),
        ]
      )
      columnPipes.append((column, pipe, column))
    # Add pipelines for individual helpful tag ratings
    for prefix in [_LOCAL, _PEER_MISLEADING, _PEER_NON_MISLEADING]:
      column = f"{prefix}_{_USER_HELPFUL_TAGS}"
      pipe = Pipeline(
        [
          ("_set_to_list", FunctionTransformer(_set_to_list)),
          (
            "onehot",
            CountVectorizer(tokenizer=_identity, preprocessor=_identity, min_df=self._tagRaterMin),
          ),
          ("selection", SelectPercentile(chi2, percentile=self._helpfulTagPercentile)),
        ]
      )
      columnPipes.append((column, pipe, column))
    # Add pipelines for individual not-helpful tag ratings
    for prefix in [_LOCAL, _PEER_MISLEADING, _PEER_NON_MISLEADING]:
      column = f"{prefix}_{_USER_NOT_HELPFUL_TAGS}"
      pipe = Pipeline(
        [
          ("_set_to_list", FunctionTransformer(_set_to_list)),
          (
            "onehot",
            CountVectorizer(tokenizer=_identity, preprocessor=_identity, min_df=self._tagRaterMin),
          ),
          ("selection", SelectPercentile(chi2, percentile=self._notHelpfulTagPercentile)),
        ]
      )
      columnPipes.append((column, pipe, column))
    # Add pipelines for tag ratio columns
    for prefix in [_LOCAL, _PEER_MISLEADING, _PEER_NON_MISLEADING]:
      for tagset in [c.notHelpfulTagsTSVOrder, c.helpfulTagsTSVOrder]:
        for tag in tagset:
          column = f"{prefix}_{tag}"
          self._column_thresholds[column] = noteInfo[column].quantile(0.99)
          if noteInfo[column].min() == noteInfo[column].max():
            continue
          pipe = Pipeline(
            [
              ("reshape", FunctionTransformer(_reshape)),
              ("drop_constants", VarianceThreshold()),
              (
                "binize",
                KBinsDiscretizer(n_bins=self._tag_ratio_bins, encode="onehot", strategy="kmeans"),
              ),
            ]
          )
          columnPipes.append((column, pipe, column))
    # Add pipelines for rating counts across notes
    columns = []
    for prefix in [_LOCAL, _PEER_MISLEADING, _PEER_NON_MISLEADING]:
      for col in _BUCKET_COUNT_COLS:
        columns.append(f"{prefix}_{col}")
    assert noteInfo[columns].isna().sum().sum() == 0
    for col in columns:
      pipe = Pipeline(
        [
          ("log", FunctionTransformer(_feature_log)),
          (
            "binize",
            KBinsDiscretizer(n_bins=self._rating_count_bins, encode="onehot", strategy="kmeans"),
          ),
        ]
      )
      columnPipes.append((col, pipe, [col]))
    for degree in [2, 3]:
      pipe = Pipeline(
        [
          ("log", FunctionTransformer(_feature_log)),
          (
            "binize",
            KBinsDiscretizer(n_bins=self._rating_count_bins, encode="onehot", strategy="kmeans"),
          ),
          ("drop_rare", VarianceThreshold(threshold=0.001)),
          (
            "cross",
            PolynomialFeatures(degree=(degree, degree), interaction_only=True, include_bias=False),
          ),
          ("drop_rare_again", VarianceThreshold(threshold=0.001)),
        ]
      )
      columnPipes.append((f"cross_note_counts_degree_{degree}", pipe, columns))
    # Add pipelines for rating counts within notes
    for prefix in [_LOCAL, _PEER_MISLEADING, _PEER_NON_MISLEADING]:
      columns = []
      for col in _BUCKET_COUNT_COLS:
        columns.append(f"{prefix}_{col}")
      assert noteInfo[columns].isna().sum().sum() == 0
      pipe = Pipeline(
        [
          ("log", FunctionTransformer(_feature_log)),
          (
            "binize",
            KBinsDiscretizer(n_bins=self._rating_count_bins, encode="onehot", strategy="kmeans"),
          ),
          ("drop_rare", VarianceThreshold(threshold=0.001)),
          ("cross_0", PolynomialFeatures(degree=(2, 2), interaction_only=True, include_bias=False)),
          ("cross_1", PolynomialFeatures(degree=(2, 2), interaction_only=True, include_bias=False)),
          ("drop_rare_again", VarianceThreshold(threshold=0.001)),
        ]
      )
      columnPipes.append((f"{prefix}_cross_note_counts_degree_4", pipe, columns))
    # Add pipelines for rater factor stats
    for prefix in [_LOCAL, _PEER_MISLEADING, _PEER_NON_MISLEADING]:
      columns = []
      for col in _STATS_COLS:
        columns.append(f"{prefix}_{col}")
      pipe = Pipeline(
        [
          ("fill_nans_df", FunctionTransformer(_fill_na)),
          (
            "binize",
            KBinsDiscretizer(n_bins=self._factor_stats_bins, encode="onehot", strategy="kmeans"),
          ),
          ("cross", PolynomialFeatures(degree=(1, 2), interaction_only=True, include_bias=False)),
          ("drop_rare", VarianceThreshold(threshold=0.001)),
        ]
      )
      columnPipes.append((f"{prefix}_stat_cols", pipe, columns))
    # Add pipelines for rating bursts
    for colset in [_QUICK_RATING_COLS, _BURST_RATING_COLS, _RECENT_RATING_COLS]:
      for col in colset:
        assert noteInfo[col].isna().sum() == 0
        if "RATIO" in col:
          self._column_thresholds[col] = noteInfo[col].quantile(0.999)
          pipe = Pipeline(
            [
              (
                "binize",
                KBinsDiscretizer(n_bins=self._burst_bins, encode="onehot", strategy="kmeans"),
              ),
            ]
          )
        else:
          assert "TOTAL" in col
          self._column_thresholds[col] = noteInfo[col].quantile(0.999)
          pipe = Pipeline(
            [
              (
                "binize",
                KBinsDiscretizer(n_bins=self._burst_bins, encode="onehot", strategy="kmeans"),
              ),
            ]
          )
        columnPipes.append((col, pipe, [col]))
    # Add pipeline for note writing latency
    assert noteInfo[_NOTE_WRITING_LATENCY].isna().sum() == 0
    self._column_thresholds[_NOTE_WRITING_LATENCY] = noteInfo[_NOTE_WRITING_LATENCY].quantile(0.999)
    pipe = Pipeline(
      [
        ("binize", KBinsDiscretizer(n_bins=self._latency_bins, encode="onehot", strategy="kmeans")),
      ]
    )
    columnPipes.append((_NOTE_WRITING_LATENCY, pipe, [_NOTE_WRITING_LATENCY]))
    # Add columns for peer notes
    peerNoteCols = [
      _TOTAL_PEER_NOTES,
      _TOTAL_PEER_MISLEADING_NOTES,
      _TOTAL_PEER_NON_MISLEADING_NOTES,
      _TOTAL_PEER_CRH_NOTES,
      _TOTAL_PEER_STABILIZATION_NOTES,
    ]
    assert noteInfo[peerNoteCols].isna().sum().sum() == 0
    for col in peerNoteCols:
      self._column_thresholds[col] = noteInfo[col].quantile(0.9999)
      pipe = Pipeline(
        [
          (
            "binize",
            KBinsDiscretizer(n_bins=self._peer_note_count_bins, encode="onehot", strategy="kmeans"),
          ),
        ]
      )
      columnPipes.append((col, pipe, [col]))
    pipe = Pipeline(
      [
        ("log", FunctionTransformer(_feature_log)),
        (
          "binize",
          KBinsDiscretizer(n_bins=self._peer_note_count_bins, encode="onehot", strategy="kmeans"),
        ),
        ("cross", PolynomialFeatures(degree=(2, 2), interaction_only=True, include_bias=False)),
        ("drop_rare", VarianceThreshold(threshold=0.001)),
      ]
    )
    columnPipes.append(("peer_note_cross_degree_2", pipe, peerNoteCols))

    # Build and return column transformer
    return ColumnTransformer(columnPipes, verbose=True)

  def _get_model_pipeline(self, noteInfo: pd.DataFrame) -> Pipeline:
    """Return a full model pipeline including feature extraction and model.

    Note that the pipeline includes a VarianceThreshold filter between feature extraction
    and model training to drop any features that have zero variance (i.e. always have the
    same value).

    Returns:
      Pipeline containing feature extraction, variance threshold and model.
    """
    return Pipeline(
      [
        ("feature_extraction", self._get_feature_pipeline(noteInfo)),
        ("drop_constants", VarianceThreshold()),
        (
          "lr",
          LogisticRegression(
            penalty=self._penalty,
            C=self._C,
            max_iter=self._maxIter,
            verbose=self._verbose,
            class_weight=self._classWeight,
            solver=self._solver,
            tol=self._tol,
          ),
        ),
      ]
    )

  def _transform_note_info(self, noteInfo: pd.DataFrame) -> pd.DataFrame:
    noteInfo = noteInfo.copy()
    # Transform tag ratio columns
    for prefix in [_LOCAL, _PEER_MISLEADING, _PEER_NON_MISLEADING]:
      for tagset in [c.notHelpfulTagsTSVOrder, c.helpfulTagsTSVOrder]:
        for tag in tagset:
          column = f"{prefix}_{tag}"
          threshold = self._column_thresholds[column]
          noteInfo[column] = noteInfo[column].fillna(-0.25).clip(-0.25, threshold)
    # Transform for rating bursts
    for colset in [_QUICK_RATING_COLS, _BURST_RATING_COLS, _RECENT_RATING_COLS]:
      for column in colset:
        threshold = self._column_thresholds[column]
        if "RATIO" in column:
          noteInfo[column] = noteInfo[column].clip(0, threshold)
        else:
          assert "TOTAL" in column
          noteInfo[column] = np.log(1 + noteInfo[column].clip(0, threshold)) / np.log(2)
    # Transform for writing latency
    threshold = self._column_thresholds[_NOTE_WRITING_LATENCY]
    noteInfo[_NOTE_WRITING_LATENCY] = np.log(
      noteInfo[_NOTE_WRITING_LATENCY].clip(0, threshold)
    ) / np.log(2)
    # Transform for peer notes
    peerNoteCols = [
      _TOTAL_PEER_NOTES,
      _TOTAL_PEER_MISLEADING_NOTES,
      _TOTAL_PEER_NON_MISLEADING_NOTES,
      _TOTAL_PEER_CRH_NOTES,
      _TOTAL_PEER_STABILIZATION_NOTES,
    ]
    for column in peerNoteCols:
      threshold = self._column_thresholds[column]
      noteInfo[column] = np.log(1 + noteInfo[column].clip(0, threshold)) / np.log(2)
    return noteInfo

  def _get_label_vector(self, noteInfo: pd.DataFrame) -> np.array:
    """Extract a binary label vector from a noteInfo DataFrame."""
    return (noteInfo[LABEL] == FLIP).values.astype(np.int8)

  def _evaluate_model(
    self, noteInfo: pd.DataFrame, threshold: Optional[float] = None
  ) -> Tuple[float, float, float, float]:
    """Apply a pipeline to noteInfo and return the AUC, TPR, FPR and associated threshold.

    Assumes that the pipeline has already been fit.  If the threshold is specified as a
    command line argument then uses the provided threshold.  Otherwise, select the
    threshold to yield a 25% FPR.

    Args:
      noteInfo: pd.DataFrame containing raw feature information and labels.

    Returns:
      Tuple containing AUC, TPR and FPR.
    """
    assert self._pipeline is not None, "pipeline must be fit prior to evaluation"
    labels = self._get_label_vector(noteInfo)
    predictions = self._pipeline.decision_function(noteInfo)
    fpRates, tpRates, thresholds = roc_curve(labels, predictions)
    auc = area_under_curve(fpRates, tpRates)
    if threshold is None:
      idx = np.argmin(np.abs(fpRates - self._crhFpRate))
      threshold = thresholds[idx]
      fpr = fpRates[idx]
      tpr = tpRates[idx]
    else:
      tn, fp, fn, tp = confusion_matrix(
        labels, (predictions > threshold).astype(np.int8), labels=np.arange(2)
      ).ravel()
      fpr = fp / (fp + tn)
      tpr = tp / (tp + fn)
    logger.info(f"threshold={threshold}  tpr={tpr}  fpr={fpr}  auc={auc}")
    return (threshold, tpr, fpr, auc)

  def _convert_col_types(self, noteInfo: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas types to numpy types.

    This conversion is necessary for scikit-learn compatibility.
    """
    for col, dtype in noteInfo.dtypes.to_dict().items():
      if isinstance(dtype, pd.Int64Dtype):
        assert noteInfo[col].isna().sum() == 0
        noteInfo[col] = noteInfo[col].astype(np.int64)
      if isinstance(dtype, pd.Float64Dtype):
        noteInfo[col] = noteInfo[col].astype(np.float64)
    return noteInfo

  def _profile_pipeline(self, pipe: Pipeline, noteInfo: pd.DataFrame) -> str:
    """Generate a numerical profile of each extracted feature.

    For each feature, we examine the dimensionality and sparsity of the feature.  For low
    dimensional features representing discretized continuous values, we also profile the
    size and boundaries of each bin.
    """
    # Generate feature matrix
    matrix = pipe.transform(noteInfo)
    # Profile matrix
    start = 0
    lines = []
    for name, transformer, _ in pipe.transformers_:
      if name == "remainder":
        continue
      end = start + len(transformer[-1].get_feature_names_out())
      total = int(matrix[:, start:end].sum())
      colMin = int(matrix[:, start:end].sum(axis=0).min())
      colMean = total / (end - start)
      colMax = int(matrix[:, start:end].sum(axis=0).max())
      rowMin = int(matrix[:, start:end].sum(axis=1).min())
      rowMean = total / (matrix.shape[0])
      rowMax = int(matrix[:, start:end].sum(axis=1).max())
      columns = [
        f"{name:<60}pos=[{start:8} {end:8} {end-start:8}]",
        f"total={total:9}",
        f"col=[{colMin:8} {colMean:8.1f} {colMax:8}]",
        f"row=[{rowMin:8} {rowMean:8.1f} {rowMax:8}]",
      ]
      if (end - start) <= 10:
        columns.append(f"{str(matrix[:, start:end].sum(axis=0).astype(np.int64)):<80}")
        columns.append(str(transformer[-1].bin_edges_[0].round(3).tolist()))
      lines.append("    ".join(columns))
      start = end
    return "\n".join(lines)

  def fit(
    self,
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    prescoringRaterModelOutput: pd.DataFrame,
  ) -> None:
    """Fit and evaluate a sklearn pipeline for predicting note status.

    Args:
      notes: pd.DataFrame
      ratings: pd.DataFrame
      noteStatusHistory: pd.DataFrame
      prescoringRaterModelOutput: pd.DataFrame

    Returns:
      sklearn pipeline covering containing full process of feature extraction, feature
      selection and prediction.
    """
    # Apply seed if necessary
    if self._seed is not None:
      logger.info(f"seeding pflip: {self._seed}")
      np.random.seed(self._seed)
    # Prepare datasets
    noteInfo = pd.concat(
      [
        self._prepare_note_info(
          notes,
          ratings,
          noteStatusHistory,
          prescoringRaterModelOutput,
          prepareForTraining=True,
          cutoff=_MIN,
        ),
        self._prepare_note_info(
          notes,
          ratings,
          noteStatusHistory,
          prescoringRaterModelOutput,
          prepareForTraining=True,
          cutoff=_MAX,
        ),
      ]
    )
    noteInfo = self._convert_col_types(noteInfo)
    noteInfo = noteInfo.sort_values(c.noteIdKey)
    if len(noteInfo) == 0:
      return
    logger.info(f"noteInfo summary: {get_df_fingerprint(noteInfo, [c.noteIdKey])}")
    # Dividing training data temporally provides a more accurate measurement, but would also
    # require excluding the newest data from training.
    trainDataFrame, validationDataFrame = train_test_split(noteInfo, train_size=self._trainSize)
    logger.info(f"pflip training data size: {len(trainDataFrame)}")
    logger.info(f"trainDataFrame summary: {get_df_fingerprint(trainDataFrame, [c.noteIdKey])}")
    logger.info(f"pflip validation data size: {len(validationDataFrame)}")
    logger.info(
      f"validationDataFrame summary: {get_df_fingerprint(validationDataFrame, [c.noteIdKey])}"
    )
    # Fit model
    self._pipeline = self._get_model_pipeline(trainDataFrame)
    trainDataFrame = self._transform_note_info(trainDataFrame)
    validationDataFrame = self._transform_note_info(validationDataFrame)
    self._pipeline.fit(trainDataFrame, self._get_label_vector(trainDataFrame))
    featureProfile = self._profile_pipeline(self._pipeline[0], trainDataFrame)
    logger.info(f"\ntraining feature matrix profile:\n{featureProfile}")
    # Evaluate model
    logger.info("Training Results:")
    threshold, _, _, _ = self._evaluate_model(trainDataFrame)
    self._predictionThreshold = threshold
    logger.info("Validation Results:")
    self._evaluate_model(validationDataFrame, threshold=threshold)

  def serialize(self) -> bytes:
    """Return a serialized version of the PFlipModel object.

    Note that since the pflip pipeline includes CountVectorizer instances that have
    functions as parameters, joblib must be called from within this same module to be
    able to serialize the functions.

    Returns:
      bytes containing a serialized PFlipModel object
    """
    buffer = BytesIO()
    joblib.dump(self, buffer)
    return buffer.getvalue()

  def predict(
    self,
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    prescoringRaterModelOutput: pd.DataFrame,
    maxBatchSize: int = 10000,
  ) -> pd.DataFrame:
    """Given input DataFrames, predict which notes will flip and lose CRH status.

    Args:
      notes: pd.DataFrame
      ratings: pd.DataFrame
      noteStatusHistory: pd.DataFrame
      prescoringRaterModelOutput: pd.DataFrame

    Returns:
      pd.DataFrame containing noteIds and predicted labels
    """
    assert self._pipeline is not None, "pipeline must be initialized prior to prediction"
    assert (
      self._predictionThreshold is not None
    ), "threshold must be initialized prior to prediction"
    # Build list of unique tweetIds
    tweetIds = notes[[c.tweetIdKey]].drop_duplicates()
    tweetIds = tweetIds[tweetIds[c.tweetIdKey] != "-1"]
    # Iterate through batches
    results = [pd.DataFrame({c.noteIdKey: [], LABEL: []})]
    start = 0
    while start < len(tweetIds):
      logger.info(f"processing prediction batch: {start}")
      # Note that all peer tweets must be included in the same batch for correct feature extraction
      tweetBatch = tweetIds.iloc[start : (start + maxBatchSize)]
      noteBatch = (
        notes[[c.noteIdKey, c.tweetIdKey]].merge(tweetBatch)[[c.noteIdKey]].drop_duplicates()
      )
      noteInfo = self._prepare_note_info(
        notes.merge(noteBatch),
        ratings.merge(noteBatch),
        noteStatusHistory.merge(noteBatch),
        prescoringRaterModelOutput,
        prepareForTraining=False,
        cutoff=None,
      )
      noteInfo = self._convert_col_types(noteInfo)
      noteInfo = self._transform_note_info(noteInfo)
      predictions = self._pipeline.decision_function(noteInfo)
      results.append(
        pd.DataFrame(
          {
            c.noteIdKey: noteInfo[c.noteIdKey],
            LABEL: [FLIP if p > self._predictionThreshold else CRH for p in predictions],
          }
        )
      )
      start += maxBatchSize
    return pd.concat(results)
