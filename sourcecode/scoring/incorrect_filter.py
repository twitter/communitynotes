"""Utilites for tag based scoring logic."""

from . import constants as c

import numpy as np
import pandas as pd


def _get_user_incorrect_ratio(ratings: pd.DataFrame) -> pd.DataFrame:
  """Computes empirical p(incorrect | not helpful tags assigned) per rater.

  Args:
    ratings: initial input ratings DF containing all ratings

  Returns:
    pd.DataFrame containing one row per user who assigned not helpful tags with their empirical propensity
    to assign "incorrect" tag
  """

  notHelpfulTaggedRatings = ratings.loc[ratings[c.notHelpfulTagsTSVOrder].sum(axis=1) > 0]
  user_incorrect = (
    notHelpfulTaggedRatings[[c.raterParticipantIdKey, "notHelpfulIncorrect"]]
    .groupby(c.raterParticipantIdKey)
    .agg("sum")
  )
  user_nh_rating_count = (
    notHelpfulTaggedRatings[[c.raterParticipantIdKey, c.noteIdKey]]
    .groupby(c.raterParticipantIdKey)
    .agg("count")
  )
  user_nh_rating_count.rename(columns={c.noteIdKey: "cnt"}, inplace=True)
  user_totals = user_incorrect.merge(user_nh_rating_count, on=c.raterParticipantIdKey)

  note_totals = (
    notHelpfulTaggedRatings[[c.raterParticipantIdKey, c.noteIdKey]]
    .groupby(c.noteIdKey)
    .agg("count")
    .reset_index()
  )
  note_totals.rename(columns={c.raterParticipantIdKey: "num_voters"}, inplace=True)
  return user_totals, note_totals


def _get_incorrect_tfidf_ratio(
  augmented_ratings: pd.DataFrame, note_nh_count: pd.DataFrame, user_filter: bool, suffix: str
) -> pd.DataFrame:
  """Computes empirical p(incorrect | note) / p(incorrect | raters over all notes) subject to rater-note inclusion function.

  Args:
    augmented_ratings: ratings DF with note and rater factors and user incorrect TF
    note_nh_count: DF with total number of NH votes per note
    filter: inclusion criteria for "incorrect" voters
    suffix: suffix for incorrect and count column names for this filter

  Returns:
    pd.DataFrame with one row for each note, with computed sum(tf_idf_incorrect) score for raters
    included in filter
  """

  ratings_w_user_totals = augmented_ratings[user_filter]
  ratings_w_user_totals.drop(
    [c.internalRaterFactor1Key, c.internalNoteFactor1Key], inplace=True, axis=1
  )
  rating_aggs = ratings_w_user_totals.groupby(c.noteIdKey).agg("sum").reset_index()
  rating_aggs_w_cnt = rating_aggs.merge(note_nh_count, on=c.noteIdKey)

  rating_aggs_w_cnt["tf_idf_incorrect"] = (
    rating_aggs_w_cnt["notHelpfulIncorrect"] / rating_aggs_w_cnt["num_voters"]
  ) / np.log(
    1 + (rating_aggs_w_cnt["notHelpfulIncorrect_total"] / rating_aggs_w_cnt["cnt"])
  )  # p(incorrect over all rater ratings)
  rating_aggs_w_cnt.drop("notHelpfulIncorrect_total", inplace=True, axis=1)
  rating_aggs_w_cnt.columns = [c.noteIdKey] + [
    f"{col}{suffix}" for col in rating_aggs_w_cnt.columns[1:]
  ]
  return rating_aggs_w_cnt


def get_incorrect_aggregates(
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

  # get per user incorrect term frequency
  user_totals, note_totals = _get_user_incorrect_ratio(ratings)
  # add user and note factors
  ratings_w_user_totals = (
    ratings[[c.raterParticipantIdKey, c.noteIdKey, "notHelpfulIncorrect"]]
    .merge(user_totals, on=c.raterParticipantIdKey, suffixes=(None, "_total"))
    .merge(noteParams[[c.noteIdKey, c.internalNoteFactor1Key]], on=c.noteIdKey)
    .merge(
      raterParams[[c.raterParticipantIdKey, c.internalRaterFactor1Key]], on=c.raterParticipantIdKey
    )
  )

  interval_filter = (
    np.abs(
      ratings_w_user_totals[c.internalRaterFactor1Key]
      - ratings_w_user_totals[c.internalNoteFactor1Key]
    )
    < c.intervalHalfWidth
  )
  interval_scores = _get_incorrect_tfidf_ratio(
    ratings_w_user_totals, note_totals, interval_filter, "_interval"
  )
  same_factor_filter = (
    (ratings_w_user_totals[c.internalRaterFactor1Key] > 0)
    & (ratings_w_user_totals[c.internalNoteFactor1Key] > 0)
  ) | (
    (ratings_w_user_totals[c.internalRaterFactor1Key] < 0)
    & (ratings_w_user_totals[c.internalNoteFactor1Key] < 0)
  )
  same_scores = _get_incorrect_tfidf_ratio(
    ratings_w_user_totals, note_totals, same_factor_filter, "_same"
  )

  incorrectAggregates = interval_scores.merge(same_scores, on=c.noteIdKey)
  return incorrectAggregates
