"""Utilites for tag based scoring logic."""

import logging
from typing import Dict

from . import constants as c

import numpy as np
import pandas as pd


logger = logging.getLogger("birdwatch.tag_filter")
logger.setLevel(logging.INFO)


def _normalize_factors(rawFactors: pd.DataFrame, entityKey: str, factorKey: str) -> pd.DataFrame:
  """Performs Z-Normalization on embedding factors.

  Z-Normalization scales a vector to have mean=0 and stddev=1, removing any effects of scale differences
  in situations where scales can vary.  Since we map raters and notes into the same embedding space the
  mean and stddev for notes and raters tend to be similar, but normalizing helps control for any differences
  and ensures that distances will remain on a similar scale even if the scale of the embedding changes.

  Args:
    rawFactors: Pairs  of a entity (note or rater) and an associated factor (embedding)
    entityKey: String corresponding to the column which identifies the entity
    factorKey: String corresponding to the column which identifies the factor

  Returns:
    pd.DataFrame containing two columns with names matching the input, except the factor column has
    been Z-normalized.
  """
  normalizedFactors = pd.DataFrame(rawFactors[[entityKey, factorKey]])
  assert len(normalizedFactors) == len(normalizedFactors[entityKey].drop_duplicates())
  mean = np.mean(normalizedFactors[factorKey])
  std = np.std(normalizedFactors[factorKey])
  normalizedFactors[factorKey] = (normalizedFactors[factorKey] - mean) / std
  return normalizedFactors


def _get_weight_from_distance(distances: pd.Series) -> pd.Series:
  """Transforms non-negative distances to weights in the range (0, 1].

  This function transforms distances to weights between zero (exclusive) and one (inclusive).
  Distances closer to 0 get weight close to 1.0, and the weights drop off as distances increase.

  Args:
    distances: pd.Series of non-negative float values representing distances between notes and raters.

  Returns:
    pd.Series of rating weights corresponding to the input distances.
  """
  normalizationFactor = np.nanpercentile(distances, c.tagPercentileForNormalization)
  return 1.0 / (1 + (distances / normalizationFactor) ** 5)


def _get_rating_weight(
  ratings: pd.DataFrame, noteParams: pd.DataFrame, raterParams: pd.DataFrame
) -> pd.DataFrame:
  """Computes the weight of each rating based on the distance between the note and the rater.

  Args:
    ratings: initial input ratings DF containing all ratings
    noteParams: MF results for notes
    raterParams: MF results for raters

  Returns:
    pd.DataFrame containing the weight for reach rating, where a rating is identified by a
    {noteId, raterId} pair.
  """
  # obtain normalized factors
  normedNotes = _normalize_factors(noteParams, c.noteIdKey, c.internalNoteFactor1Key)
  normedRaters = _normalize_factors(raterParams, c.raterParticipantIdKey, c.internalRaterFactor1Key)
  # prune ratings to eliminate cases where the note or rater isn't included
  # and simultaneously add factor data
  ratings = ratings.merge(normedNotes, on=c.noteIdKey, how="inner")
  ratings = ratings.merge(normedRaters, on=c.raterParticipantIdKey, how="inner")
  # normalize impact factors
  ratings[c.ratingWeightKey] = _get_weight_from_distance(
    np.abs(ratings[c.internalRaterFactor1Key] - ratings[c.internalNoteFactor1Key])
  )
  # return relevant columns
  return ratings[[c.noteIdKey, c.raterParticipantIdKey, c.ratingWeightKey]]


def get_note_tag_aggregates(
  ratings: pd.DataFrame, noteParams: pd.DataFrame, raterParams: pd.DataFrame
) -> pd.DataFrame:
  """Computes non-helpful tag aggregates for each note.

  Args:
    ratings: initial input ratings DF containing all ratings
    noteParams: MF results for notes
    raterParams: MF results for raters

  Returns:
    pd.DataFrame containing one row per note that was scored during MF.  Columns correspond to
    aggregates for the Not-Helpful tags, including raw totals, totals adjusted based on the
    distance between the rater and the note and ratios based on the adjusted weight totals.
  """
  # Obtain a weight for each rating.  Note that since weights are defined using the note and rater
  # embedding factors, obtaining a weight inherently filters out ratings where either the note or
  # rater were not included in matrix factorization.
  ratingWeights = _get_rating_weight(ratings, noteParams, raterParams)
  # Filter ratings to only the columns which we need.
  ratings = ratings[[c.noteIdKey, c.raterParticipantIdKey] + c.notHelpfulTagsTSVOrder]
  # Add weights to ratings, which will inherently filter out ratings where either the note or
  # rater were not included in matrix factorization.
  ratings = ratings.merge(ratingWeights, on=[c.noteIdKey, c.raterParticipantIdKey], how="inner")
  # Introduce scaled tag columns, and drop original columns
  adjustedRatings = ratings[c.notHelpfulTagsTSVOrder].multiply(
    ratings[c.ratingWeightKey], axis="rows"
  )
  adjustedRatings.columns = c.notHelpfulTagsAdjustedColumns
  ratings = ratings.drop(columns=c.notHelpfulTagsTSVOrder)
  ratings = pd.concat([ratings, adjustedRatings], axis="columns")
  # Aggregate by note to get adjusted tag and weight totals for each note.
  ratings = ratings.drop(columns=c.raterParticipantIdKey)
  noteTagAggregates = ratings.groupby(c.noteIdKey).sum().reset_index()
  adjustedRatioRatings = noteTagAggregates[c.notHelpfulTagsAdjustedColumns].divide(
    noteTagAggregates[c.ratingWeightKey], axis="rows"
  )
  adjustedRatioRatings.columns = c.notHelpfulTagsAdjustedRatioColumns
  noteTagAggregates = pd.concat([noteTagAggregates, adjustedRatioRatings], axis="columns")
  return noteTagAggregates


def get_tag_thresholds(ratings: pd.DataFrame, percentile: int) -> Dict[str, float]:
  """Determine a Nth percentile threshold for each adjusted ratio tag.

  Args:
    ratings: DataFrame containing adjusted ratio columns
    percnetile: int in the range [0, 100)

  Returns:
    Dictionary mapping adjusted ratio columns to a threshold value
  """
  thresholds = {}
  for column in c.notHelpfulTagsAdjustedRatioColumns:
    if len(ratings[column]) == 0:
      logger.info(
        f"Warning: No ratings for column {column} in get_tag_thresholds. Setting threshold to 0.0 arbitrarily."
      )
      thresholds[column] = 0.0
    else:
      thresholds[column] = np.quantile(ratings[column], np.arange(0, 1, 0.01))[percentile]
  return thresholds
