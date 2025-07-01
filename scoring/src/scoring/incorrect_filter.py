"""Utilites for tag based scoring logic."""

from typing import Optional

from . import constants as c

import numpy as np
import pandas as pd


def get_user_incorrect_ratio(ratings: pd.DataFrame) -> pd.DataFrame:
  """Computes empirical p(incorrect | not helpful tags assigned) per rater.
  Called during prescoring only, since it uses entire rating history.

  Args:
    ratings: DF containing ratings.

  Returns:
    pd.DataFrame containing one row per user who assigned not helpful tags with their empirical propensity
    to assign "incorrect" tag
  """
  # Filter down to just ratings with some nh tags used.
  nhTagRatings = ratings.loc[ratings[c.notHelpfulTagsTSVOrder].sum(axis=1) > 0]

  user_incorrect = (
    (
      nhTagRatings[[c.raterParticipantIdKey, c.notHelpfulIncorrectTagKey]]
      .groupby(c.raterParticipantIdKey)
      .agg("sum")
    )
    .rename(columns={c.notHelpfulIncorrectTagKey: c.incorrectTagRatingsMadeByRaterKey})
    .reset_index()
  )

  user_nh_rating_count = (
    (
      nhTagRatings[[c.raterParticipantIdKey, c.noteIdKey]]
      .groupby(c.raterParticipantIdKey)
      .agg("count")
    )
    .rename(columns={c.noteIdKey: c.totalRatingsMadeByRaterKey})
    .reset_index()
  )

  user_totals = user_incorrect.merge(user_nh_rating_count, on=c.raterParticipantIdKey)

  return user_totals


def _get_incorrect_tfidf_ratio(
  augmented_ratings: pd.DataFrame, interval_filter: Optional[bool]
) -> pd.DataFrame:
  """Computes empirical p(incorrect | note) / p(incorrect | raters over all notes) subject to rater-note inclusion function.

  Args:
    augmented_ratings: ratings DF with note and rater factors and user incorrect TF
    interval_filter: inclusion criteria: only keep the ratings where the rater and note factors are within a certain interval

  Returns:
    pd.DataFrame with one row for each note, with computed sum(tf_idf_incorrect) score for raters
    included in filter
  """
  if interval_filter is not None:
    ratings_w_user_totals = augmented_ratings[interval_filter]
  else:
    ratings_w_user_totals = augmented_ratings

  note_nh_count = (
    ratings_w_user_totals[[c.raterParticipantIdKey, c.noteIdKey]]
    .groupby(c.noteIdKey)
    .agg("count")
    .reset_index()
  )
  note_nh_count.rename(columns={c.raterParticipantIdKey: c.numVotersKey}, inplace=True)

  columns_to_attempt_to_drop = [
    c.internalRaterFactor1Key,
    c.internalNoteFactor1Key,
    c.raterParticipantIdKey,
  ]
  columns_to_drop = ratings_w_user_totals.columns.intersection(columns_to_attempt_to_drop)
  ratings_w_user_totals.drop(columns_to_drop, inplace=True, axis=1)

  ratings_w_user_totals[c.incorrectTagRateByRaterKey] = (
    ratings_w_user_totals[c.incorrectTagRatingsMadeByRaterKey]
    / ratings_w_user_totals[c.totalRatingsMadeByRaterKey]
  )

  # Setup columns to be aggregated so they are not dropped during aggregation
  ratings_w_user_totals[c.incorrectTagRateByRaterKey].fillna(0, inplace=True)
  ratings_w_user_totals[c.incorrectTagRateByRaterKey] = ratings_w_user_totals[
    c.incorrectTagRateByRaterKey
  ].astype(np.double)
  ratings_w_user_totals[c.incorrectTagRatingsMadeByRaterKey].fillna(0, inplace=True)
  ratings_w_user_totals[c.incorrectTagRatingsMadeByRaterKey] = ratings_w_user_totals[
    c.incorrectTagRatingsMadeByRaterKey
  ].astype(np.double)
  ratings_w_user_totals[c.totalRatingsMadeByRaterKey].fillna(0, inplace=True)
  ratings_w_user_totals[c.totalRatingsMadeByRaterKey] = ratings_w_user_totals[
    c.totalRatingsMadeByRaterKey
  ].astype(np.double)

  rating_aggs = ratings_w_user_totals.groupby(c.noteIdKey).agg("sum").reset_index()
  rating_aggs_w_cnt = rating_aggs.merge(note_nh_count, on=c.noteIdKey)

  rating_aggs_w_cnt[c.noteTfIdfIncorrectScoreKey] = (
    rating_aggs_w_cnt[c.notHelpfulIncorrectTagKey]
  ) / np.log(
    1 + (rating_aggs_w_cnt[c.incorrectTagRateByRaterKey])
  )  # p(incorrect over all rater ratings)
  rating_aggs_w_cnt.drop(
    [c.totalRatingsMadeByRaterKey, c.incorrectTagRatingsMadeByRaterKey], inplace=True, axis=1
  )

  rating_aggs_w_cnt.rename(
    columns={
      c.notHelpfulIncorrectTagKey: c.notHelpfulIncorrectIntervalKey,
      c.incorrectTagRateByRaterKey: c.sumOfIncorrectTagRateByRaterIntervalKey,
      c.numVotersKey: c.numVotersIntervalKey,
      c.noteTfIdfIncorrectScoreKey: c.noteTfIdfIncorrectScoreIntervalKey,
    },
    inplace=True,
  )
  return rating_aggs_w_cnt


def get_incorrect_aggregates_final_scoring(
  ratings: pd.DataFrame,
  noteParams: pd.DataFrame,
  raterParamsWithRatingCounts: pd.DataFrame,
) -> pd.DataFrame:
  """Computes non-helpful tag aggregates for each note. Intended to be called in final scoring.

  Args:
    ratings: initial input ratings DF containing all ratings
    noteParams: MF results for notes
    raterParamsWithRatingCounts: MF results for raters. Should include c.incorrectTagRatingsMadeByRaterKey and c.totalRatingsMadeByRaterKey.
    raterIncorrectTagRatingCounts: should contain: c.raterParticipantIdKey,

  Returns:
    pd.DataFrame containing one row per note that was scored during MF.  Columns correspond to
    aggregates for the Not-Helpful tags, including raw totals, totals adjusted based on the
    distance between the rater and the note and ratios based on the adjusted weight totals.
  """
  # consider only ratings with some NH tag
  notHelpfulTaggedRatings = ratings.loc[ratings[c.notHelpfulTagsTSVOrder].sum(axis=1) > 0]

  # join user totals, note factors, and rater factors with each rating
  ratings_w_user_totals = (
    notHelpfulTaggedRatings[[c.raterParticipantIdKey, c.noteIdKey, c.notHelpfulIncorrectTagKey]]
    .merge(noteParams[[c.noteIdKey, c.internalNoteFactor1Key]], on=c.noteIdKey)
    .merge(
      raterParamsWithRatingCounts[
        [
          c.raterParticipantIdKey,
          c.internalRaterFactor1Key,
          c.incorrectTagRatingsMadeByRaterKey,
          c.totalRatingsMadeByRaterKey,
        ]
      ],
      on=c.raterParticipantIdKey,
    )
  )

  # Keep users with clipped factors within a certain interval of notes' (e.g. within 0.3)
  interval_filter = (
    np.abs(
      ratings_w_user_totals[c.internalRaterFactor1Key].clip(-0.4, 0.4)
      - ratings_w_user_totals[c.internalNoteFactor1Key].clip(-0.4, 0.4)
    )
    < c.intervalHalfWidth
  )

  incorrectAggregates = _get_incorrect_tfidf_ratio(ratings_w_user_totals, interval_filter)
  return incorrectAggregates


def get_incorrect_aggregates(
  ratings: pd.DataFrame,
  noteParams: pd.DataFrame,
  raterParams: pd.DataFrame,
) -> pd.DataFrame:
  """
  Legacy version of this function, computable all at once instead of being called separately in prescoring
  vs. final scoring.
  """
  # get per user incorrect term frequency -- normally called during prescoring
  raterParamsWithRatingCounts = raterParams.merge(get_user_incorrect_ratio(ratings))

  return get_incorrect_aggregates_final_scoring(ratings, noteParams, raterParamsWithRatingCounts)
