"""Train and apply a model to predict which notes will lose CRH status.

This module implements a supervised model designed to be applied at the time that a note
is considered for receiving CRH status.  The model predicts whether the note will ultimately
lock to CRH status or flip back to NMR.  The features used by the model include:
* Individual user Helpfulness and tag ratings.
* Note author.
* Ratios of how common each tag is in ratings the note has received.
* Aggregate statistics about rater factors (e.g. standard deviation of rater factors for
  Helpful ratings, number of {Helpful, Somewhat Helpful, Not Helpful} ratings from users
  with {positive, center, negative} factors, etc.).

The training data includes all notes that were CRH at some point and have either locked to
CRH or NMR, except for any note that either drifted to NMR status after locking or is only
presently scored as CRH due to inertia.

The feature extraction and model training are implemented entirely in scikit-learn.
"""

from io import BytesIO
import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple

from . import constants as c
from .enums import Scorers
from .pandas_utils import get_df_fingerprint

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
logger = logging.getLogger("birdwatch.pflip_model")
logger.setLevel(logging.INFO)


# Exposed constants
LABEL = "LABEL"
CRH = "CRH"
FLIP = "FLIP"

# Internal column names
_RATER_FACTOR = "RATER_FACTOR"
_STATUS_MTS = "STATUS_MTS"
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


# Define helper functions at module level so feature extraction pipeline doesn't require
# any lambda functions (and consequently can be pickled.)
def _identity(x: Any) -> Any:
  """Used to create modified preprocessing and tokenization for CountVectorizer."""
  return x


def _feature_log(features: pd.Series) -> pd.Series:
  """Helper to log-scale features while allowing for zero valued features."""
  return np.log(1 + features)


def _fill_nans(features: pd.Series) -> pd.Series:
  """Fill NaNs in raw feature values with a constant prior to discretization.

  Args:
    features: pd.Series potentially containing NaN values to be replaced.

  Returns:
    feature array with NaN values replaced by a default, out-of-distribution value.
  """
  MAX_FLOAT = 4
  assert 0 == ((1.5 * features.dropna().abs().values) > MAX_FLOAT).sum()
  return features.fillna(-1 * MAX_FLOAT)


class PFlipModel(object):
  def __init__(
    self,
    helpfulnessRaterMin: int = 5,
    tagRaterMin: int = 5,
    helpfulTagPercentile: int = 1,
    notHelpfulTagPercentile: int = 50,
    tagRatioBins: int = 10,
    summaryBins: int = 10,
    helpfulStatsBins: int = 10,
    penalty: str = "l2",
    C: float = 0.002,
    maxIter: int = 2500,
    verbose: int = 1,
    classWeight: Dict[int, int] = {0: 1000, 1: 1},
    trainSize: float = 0.9,
    seed: Optional[int] = None,
    crhFpRate: float = 0.6,
  ):
    """Configure feature extraction and learning hyperparameters for pflip model.

    Args:
      helpfulnessRaterMin: Minimum number of ratings from a given rater with a given
        Helpfulness value for the {rater, value} pair to appear as a feature.
      tagRaterMin: Minimum number of uses of a tag from a given rater for the
        {rater, tag} pair to appear as a feature.
      helpfulTagPercentile: Percentage of {rater, tag} feature columns to retain for
        Helpful tags.
      notHelpfulTagPercentile: Percentage of {rater, tag} feature columns to retain
        for Not Helpful tags.
      tagRatioBins: Number of intervals for discretization of tag ratios.
      summaryBins: Number of intervals for discretization of rating count buckets.
      helpfulStatBins: Number of intervals for discretization of aggregate stats about
        rater factors from Helpful ratings.
      penalty: str specifying whether to apply "l1" or "l2" regularization.
      C: float specifying inverse of the regularization strength.
      maxIter: maximum number of training rounds
      verbose: logging level during training
      classWeight: loss weight assigned to positive (flip) and negative (crh) samples
      validation: amount of data to hold out for model validation
      seed: randomness seed used to make data split and model training deterministic
      chrFpRate: target FP rate for CRH notes
    """
    self._pipeline: Optional[Pipeline] = None
    self._predictionThreshold: Optional[float] = None
    self._helpfulnessRaterMin = helpfulnessRaterMin
    self._tagRaterMin = tagRaterMin
    self._helpfulTagPercentile = helpfulTagPercentile
    self._notHelpfulTagPercentile = notHelpfulTagPercentile
    self._tagRatioBins = tagRatioBins
    self._summaryBins = summaryBins
    self._helpfulStatsBins = helpfulStatsBins
    self._penalty = penalty
    self._C = C
    self._maxIter = maxIter
    self._verbose = verbose
    self._classWeight = classWeight
    self._trainSize = trainSize
    self._seed = seed
    self._crhFpRate = crhFpRate

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

  def _compute_rating_cutoff(self, noteStatusHistory: pd.DataFrame) -> pd.DataFrame:
    """Compute time limits on which ratings to include for each note.

    Given that the model is applied before a note receives status and the model is
    attempting to predict final locked status, we restrict the model to ratings that
    occur before the first non-NMR status issued to the note (or alternately include
    all ratings if the note never received a non-NMR status).

    Args:
      noteStatusHistory: pd.DataFrame used to determine time of the first CRH status,
        if applicable.

    Returns:
      pd.DataFrame with noteId and STATUS_MTS columns
    """
    ratingCutoff = noteStatusHistory[
      [
        c.noteIdKey,
        c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey,
        c.timestampMillisOfNoteFirstNonNMRLabelKey,
      ]
    ].copy()
    ratingCutoff.loc[
      ratingCutoff[c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey] == -1,
      c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey,
    ] = np.nan
    ratingCutoff[_STATUS_MTS] = ratingCutoff[
      [
        c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey,
        c.timestampMillisOfNoteFirstNonNMRLabelKey,
      ]
    ].min(axis=1)
    return ratingCutoff[[c.noteIdKey, _STATUS_MTS]].dropna()

  def _prepare_ratings(
    self,
    ratings: pd.DataFrame,
    raterFactors: pd.DataFrame,
    ratingCutoff: pd.DataFrame,
  ) -> pd.DataFrame:
    """Prune add rater factor and prune any ratings occurring after status cutoff.
    Note that the p(flip) model is applied at the time that note would enter the
    NmrDueToMinStableCrhTime waiting period.  Consequently, for all notes scored in
    production there are by definition no ratings available after a status would
    have been shown.  While filtering ratings is only necessary during training,
    there is no risk / effect of filtering ratings in production.

    Args:
      ratings: pd.DataFrame containing all ratings used in scoring
      raterFactors: pd.DataFrame mapping raters to a single factor.
      ratingCutoff: pd.DataFrame specifying rating cutoffs for each note when applicable

    Returns:
      pd.DataFrame filtered to include only ratings before a note is assigned status
        or enters the NmrDueToMinStableCrhTime waiting period and augmented with
        rater factors.
    """
    # Add factors and rating cutoff
    ratings = ratings.merge(raterFactors, how="left")
    ratings = ratings.merge(ratingCutoff, how="left")
    # Add rating cutoff and prune late ratings
    ratings = ratings[
      (ratings[_STATUS_MTS].isna()) | (ratings[c.createdAtMillisKey] < ratings[_STATUS_MTS])
    ]
    # Remove _STATUS_MTS column
    return ratings.drop(columns=_STATUS_MTS)

  def _get_recent_notes(
    self,
    notes: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    cutoffMts: int = (1000 * 60 * 60 * 24 * 365),
  ):
    """Return DataFrame containing notes authored within cutoffMts.

    Args:
      notes: Input DataFrame containing creation times.
      noteStatusHistory: Input DataFrame containing creation times.
      cutoffMts: Allowable creation window measured in milliseconds.
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
    threshold = c.epochMillis - cutoffMts
    return noteStatusHistory[noteStatusHistory[c.createdAtMillisKey] > threshold][[c.noteIdKey]]

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
      .agg(list)
      .reset_index(drop=False)
    )
    helpfulnessRatings = notes.merge(helpfulnessRatings, how="left")
    helpfulnessRatings[_USER_HELPFULNESS_RATINGS] = helpfulnessRatings[
      _USER_HELPFULNESS_RATINGS
    ].apply(lambda d: d if isinstance(d, list) else [])
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
      [f"{rater}:{tag}" for tag in tagStrs[np.where(row)[0]]]
      for rater, row in zip(ratingTags[c.raterParticipantIdKey], ratingTags[tagCols].values)
    ]
    ratingTags = (
      ratingTags[[c.noteIdKey, outCol]].groupby(c.noteIdKey).agg(list).reset_index(drop=False)
    )
    ratingTags[outCol] = ratingTags[outCol].apply(lambda userTags: list(itertools.chain(*userTags)))
    ratingTags = notes[[c.noteIdKey]].merge(ratingTags, how="left")
    ratingTags[outCol] = ratingTags[outCol].apply(
      lambda userTags: userTags if isinstance(userTags, list) else []
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
      notes[[c.noteIdKey]].merge(summary, on=c.noteIdKey, how="left").fillna(0.0).astype(np.int64)
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

  def _prepare_note_info(
    self,
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    prescoringRaterModelOutput: pd.DataFrame,
    pruneNotes: bool,
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
      pruneNotes: True IFF noteInfo should be filtered to only include labeled data

    Returns:
      pd.DataFrame containing all feature information with one row per note.
    """
    # Augment rating with factors, and exclude ratings after status
    logger.info(f"total ratings considered for pflip model: {len(ratings)}")
    raterFactors = self._compute_rater_factors(prescoringRaterModelOutput)
    ratingCutoff = self._compute_rating_cutoff(noteStatusHistory)
    ratings = self._prepare_ratings(ratings, raterFactors, ratingCutoff)
    logger.info(f"total ratings before initial note status for pflip model: {len(ratings)}")
    # Identify set of labeled notes and prune ratings accordingly
    notes = self._get_recent_notes(notes, noteStatusHistory)
    if pruneNotes:
      labels = self._label_notes(noteStatusHistory)
      notes = notes.merge(labels)
      labelCounts = notes[LABEL].value_counts(dropna=False)
      logger.info(f"labels after restricting to recent notes:\n{labelCounts}")
      ratings = ratings.merge(notes[[c.noteIdKey]])
    logger.info(f"total ratings included in pflip model: {len(ratings)}")
    # Augment notes with features.  Note that attributes of the note (e.g. author,
    # creation time) should always be available because we filter to notes with the creation time
    # in the last year, inherently removing any deleted notes where the creation time is unavailable.
    helpfulnessRatings = self._get_helpfulness_ratings(notes[[c.noteIdKey]], ratings)
    notes = notes.merge(helpfulnessRatings, how="inner")
    noteAuthors = noteStatusHistory[[c.noteIdKey, c.noteAuthorParticipantIdKey]]
    notes = notes.merge(noteAuthors, how="inner")
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

  def _get_feature_pipeline(self) -> Pipeline:
    """Returns a scikit-learn pipeline for converting noteInfo into a feature matrix.

    The feature extraction pipeline applies different transformations to different
    columns.  The extraction pipeline applies different transformations to different
    columns within the noteInfo DataFrame.  In general:
    * User helpfulness and tag ratings are represented with a one-hot encoding.
    * Authorship is represented with a one-hot encoding.
    * Tag ratios are discretized using uniform width buckets, then one-hot encoded.
    * Aggregate summary statistics about note ratings are crossed and discretized, then
      one-hot encoded.

    Note that since the pipeline also includes feature selection, fitting the pipeline
    requires access to both the noteInfo DataFrame and labels.

    Returns:
      ColumnTransformer Pipeline composed of 7 constituent Pipelines, each handling
      different columns.
    """

    # Convert user helpfulness and tag rating directly into model features.  Only include
    # {user, rating} pairs where the pair occurs at least 5 times, and apply additional
    # filtering to the tags.
    rating_pipeline = Pipeline(
      [
        (
          "onehot",
          CountVectorizer(
            tokenizer=_identity, preprocessor=_identity, min_df=self._helpfulnessRaterMin
          ),
        )
      ]
    )
    helpful_tag_pipeline = Pipeline(
      [
        (
          "onehot",
          CountVectorizer(tokenizer=_identity, preprocessor=_identity, min_df=self._tagRaterMin),
        ),
        ("selection", SelectPercentile(chi2, percentile=self._helpfulTagPercentile)),
      ]
    )
    not_helpful_tag_pipeline = Pipeline(
      [
        (
          "onehot",
          CountVectorizer(tokenizer=_identity, preprocessor=_identity, min_df=self._tagRaterMin),
        ),
        ("selection", SelectPercentile(chi2, percentile=self._notHelpfulTagPercentile)),
      ]
    )
    # Convert authorship to a feature.  Note the featurization process is different from
    # ratings because there is exactly one author per note.
    author_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    # Discretize tag ratios.
    tag_pipeline = Pipeline(
      [
        # During training there should never be a note that doesn't have any ratings in the
        # training set, but during prediction there could be notes without ratings, so we
        # impute 0.  It's unclear what impact this has on predictions, but it doesn't matter
        # because we are only using prediction values for notes that are being considered
        # for becoming CRH (and therefore have ratings, just like the training data).
        ("fill_nans_df", FunctionTransformer(_fill_nans)),
        ("drop_constants", VarianceThreshold()),
        (
          "binize",
          KBinsDiscretizer(n_bins=self._tagRatioBins, encode="onehot", strategy="quantile"),
        ),
      ]
    )
    # Log, cross and discretize rating counts of {helpful, somewhat helpful, not helpful}
    # x {left, center, right} for each note.
    summary_total_cross_first_pipeline = Pipeline(
      [
        ("log", FunctionTransformer(_feature_log)),
        ("cross", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        (
          "binize",
          KBinsDiscretizer(n_bins=self._summaryBins, encode="onehot", strategy="quantile"),
        ),
      ]
    )
    # Discretize and cross stats about the rater factors (e.g. max positive factor
    # that rated helpful).
    stats_bin_first_pipeline = Pipeline(
      [
        ("fill_nans_df", FunctionTransformer(_fill_nans)),
        (
          "binize",
          KBinsDiscretizer(n_bins=self._helpfulStatsBins, encode="onehot", strategy="uniform"),
        ),
        ("cross", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
      ]
    )
    preprocess = ColumnTransformer(
      [
        ("ratings", rating_pipeline, _USER_HELPFULNESS_RATINGS),
        ("helpful_tags", helpful_tag_pipeline, _USER_HELPFUL_TAGS),
        ("not_helpful_tags", not_helpful_tag_pipeline, _USER_NOT_HELPFUL_TAGS),
        ("authors", author_pipeline, [c.noteAuthorParticipantIdKey]),
        ("tags", tag_pipeline, c.helpfulTagsTSVOrder + c.notHelpfulTagsTSVOrder),
        ("summary", summary_total_cross_first_pipeline, _BUCKET_COUNT_COLS),
        ("stats", stats_bin_first_pipeline, _STATS_COLS),
      ]
    )
    return preprocess

  def _get_model_pipeline(self) -> Pipeline:
    """Return a full model pipeline including feature extraction and model.

    Note that the pipeline includes a VarianceThreshold filter between feature extraction
    and model training to drop any features that have zero variance (i.e. always have the
    same value).

    Returns:
      Pipeline containing feature extraction, variance threshold and model.
    """
    return Pipeline(
      [
        ("feature_extraction", self._get_feature_pipeline()),
        ("drop_constants", VarianceThreshold()),
        (
          "lr",
          LogisticRegression(
            penalty=self._penalty,
            C=self._C,
            max_iter=self._maxIter,
            verbose=self._verbose,
            class_weight=self._classWeight,
          ),
        ),
      ]
    )

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
    noteInfo = self._prepare_note_info(
      notes, ratings, noteStatusHistory, prescoringRaterModelOutput, pruneNotes=True
    ).sort_values(c.noteIdKey)
    if len(noteInfo) == 0:
      return
    logger.info(f"noteInfo summary: {get_df_fingerprint(noteInfo, [c.noteIdKey])}")
    trainDataFrame, validationDataFrame = train_test_split(noteInfo, train_size=self._trainSize)
    logger.info(f"pflip training data size: {len(trainDataFrame)}")
    logger.info(f"trainDataFrame summary: {get_df_fingerprint(trainDataFrame, [c.noteIdKey])}")
    logger.info(f"pflip validation data size: {len(validationDataFrame)}")
    logger.info(
      f"validationDataFrame summary: {get_df_fingerprint(validationDataFrame, [c.noteIdKey])}"
    )
    # Fit model
    self._pipeline = self._get_model_pipeline()
    self._pipeline.fit(trainDataFrame, self._get_label_vector(trainDataFrame))
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
  ) -> Optional[Pipeline]:
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
    noteInfo = self._prepare_note_info(
      notes, ratings, noteStatusHistory, prescoringRaterModelOutput, pruneNotes=False
    )
    predictions = self._pipeline.decision_function(noteInfo)
    return pd.DataFrame(
      {
        c.noteIdKey: noteInfo[c.noteIdKey],
        LABEL: [FLIP if p > self._predictionThreshold else CRH for p in predictions],
      }
    )
