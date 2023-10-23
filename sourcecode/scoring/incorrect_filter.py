"""Utilites for tag based scoring logic."""

from typing import List, Optional

from . import constants as c

import numpy as np
import pandas as pd


def _get_user_incorrect_ratio(nhTagRatings: pd.DataFrame) -> pd.DataFrame:
  """Computes empirical p(incorrect | not helpful tags assigned) per rater.

  Args:
    nhTagRatings: DF containing all ratings with some NH tag

  Returns:
    pd.DataFrame containing one row per user who assigned not helpful tags with their empirical propensity
    to assign "incorrect" tag
  """

  user_incorrect = (
    nhTagRatings[[c.raterParticipantIdKey, c.notHelpfulIncorrectTagKey]]
    .groupby(c.raterParticipantIdKey)
    .agg("sum")
  )
  user_nh_rating_count = (
    nhTagRatings[[c.raterParticipantIdKey, c.noteIdKey]]
    .groupby(c.raterParticipantIdKey)
    .agg("count")
  )
  user_nh_rating_count.rename(columns={c.noteIdKey: "cnt"}, inplace=True)
  user_totals = user_incorrect.merge(user_nh_rating_count, on=c.raterParticipantIdKey)

  return user_totals


def _get_incorrect_tfidf_ratio(
  augmented_ratings: pd.DataFrame, user_filter: Optional[bool], suffix: str
) -> pd.DataFrame:
  """Computes empirical p(incorrect | note) / p(incorrect | raters over all notes) subject to rater-note inclusion function.

  Args:
    augmented_ratings: ratings DF with note and rater factors and user incorrect TF
    filter: inclusion criteria for "incorrect" voters
    suffix: suffix for incorrect and count column names for this filter

  Returns:
    pd.DataFrame with one row for each note, with computed sum(tf_idf_incorrect) score for raters
    included in filter
  """
  if user_filter is not None:
    ratings_w_user_totals = augmented_ratings[user_filter]
  else:
    ratings_w_user_totals = augmented_ratings

  note_nh_count = (
    ratings_w_user_totals[[c.raterParticipantIdKey, c.noteIdKey]]
    .groupby(c.noteIdKey)
    .agg("count")
    .reset_index()
  )
  note_nh_count.rename(columns={c.raterParticipantIdKey: "num_voters"}, inplace=True)

  columns_to_attempt_to_drop = [
    c.internalRaterFactor1Key,
    c.internalNoteFactor1Key,
    c.raterParticipantIdKey,
  ]
  columns_to_drop = ratings_w_user_totals.columns.intersection(columns_to_attempt_to_drop)
  ratings_w_user_totals.drop(columns_to_drop, inplace=True, axis=1)

  ratings_w_user_totals["p_incorrect_user"] = (
    ratings_w_user_totals["notHelpfulIncorrect_total"] / ratings_w_user_totals["cnt"]
  )

  rating_aggs = ratings_w_user_totals.groupby(c.noteIdKey).agg("sum").reset_index()
  rating_aggs_w_cnt = rating_aggs.merge(note_nh_count, on=c.noteIdKey)

  rating_aggs_w_cnt["tf_idf_incorrect"] = (rating_aggs_w_cnt[c.notHelpfulIncorrectTagKey]) / np.log(
    1 + (rating_aggs_w_cnt["p_incorrect_user"])
  )  # p(incorrect over all rater ratings)
  rating_aggs_w_cnt.drop(["notHelpfulIncorrect_total", "cnt"], inplace=True, axis=1)
  rating_aggs_w_cnt.columns = [c.noteIdKey] + [
    f"{col}{suffix}" for col in rating_aggs_w_cnt.columns[1:]
  ]
  return rating_aggs_w_cnt


def get_incorrect_aggregates(
  ratingsOrig: pd.DataFrame,
  noteParams: pd.DataFrame,
  raterParams: pd.DataFrame,
  applyFilter: bool = True,
  extraCols: List[str] = [],
  colSuffix: str = "",
) -> pd.DataFrame:
  """Computes non-helpful tag aggregates for each note.

  Args:
    ratings: initial input ratings DF containing all ratings
    noteParams: MF results for notes
    raterParams: MF results for raters
    applyFilter: bool indicating whether to filter included ratings based on factor
    extraCols: list of tags to include along with notHelpfulIncorrect
    colSuffix: str which will be added to the end of each column other than noteId

  Returns:
    pd.DataFrame containing one row per note that was scored during MF.  Columns correspond to
    aggregates for the Not-Helpful tags, including raw totals, totals adjusted based on the
    distance between the rater and the note and ratios based on the adjusted weight totals.
  """
  # augment notHelpfulIncorrect with any additional columns
  ratings = ratingsOrig.copy()
  if extraCols:
    for column in extraCols:
      assert column is not c.notHelpfulIncorrectTagKey
      ratings[c.notHelpfulIncorrectTagKey] += ratings[column]
    ratings[c.notHelpfulIncorrectTagKey] = (
      ratings[c.notHelpfulIncorrectTagKey].clip(0, 1).astype(np.int64)
    )

  # consider only ratings with some NH tag
  notHelpfulTaggedRatings = ratings.loc[ratings[c.notHelpfulTagsTSVOrder].sum(axis=1) > 0]

  # get per user incorrect term frequency
  user_totals = _get_user_incorrect_ratio(notHelpfulTaggedRatings)
  # add user and note factors
  ratings_w_user_totals = (
    notHelpfulTaggedRatings[[c.raterParticipantIdKey, c.noteIdKey, c.notHelpfulIncorrectTagKey]]
    .merge(user_totals, on=c.raterParticipantIdKey, suffixes=(None, "_total"))
    .merge(noteParams[[c.noteIdKey, c.internalNoteFactor1Key]], on=c.noteIdKey)
    .merge(
      raterParams[[c.raterParticipantIdKey, c.internalRaterFactor1Key]], on=c.raterParticipantIdKey
    )
  )

  interval_filter = None
  if applyFilter:
    interval_filter = (
      np.abs(
        ratings_w_user_totals[c.internalRaterFactor1Key].clip(-0.4, 0.4)
        - ratings_w_user_totals[c.internalNoteFactor1Key].clip(-0.4, 0.4)
      )
      < c.intervalHalfWidth
    )

  incorrectAggregates = _get_incorrect_tfidf_ratio(
    ratings_w_user_totals, interval_filter, "_interval"
  )

  # apply column suffix
  columns = incorrectAggregates.columns
  cols = [f"{col}{colSuffix}" if col is not c.noteIdKey else col for col in columns]
  incorrectAggregates.columns = cols

  return incorrectAggregates
