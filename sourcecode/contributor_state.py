from time import time

import constants as c
from note_ratings import get_ratings_with_scores, get_valid_ratings

import pandas as pd


def _get_visible_rating_counts(
  scoredNotes: pd.DataFrame, ratings: pd.DataFrame, noteStatusHistory: pd.DataFrame
) -> pd.DataFrame:
  """
  Given scored notes from the algorithm, all ratings, and note status history, this function
  analyzes how succesfully a user rates notes. It aggregates how successfully/unsucessfully
  a notes ratings aligns with a contributors ratings.

  Args:
      scoredNotes (pd.DataFrame): Notes scored from MF + contributor stats
      ratings (pd.DataFrame): all ratings
      statusHistory (pd.DataFrame): history of note statuses
  Returns:
      pd.DataFrame: noteCounts The visible rating counts
  """
  ratingCountRows = [
    c.successfulRatingHelpfulCount,
    c.successfulRatingNotHelpfulCount,
    c.successfulRatingTotal,
    c.unsuccessfulRatingHelpfulCount,
    c.unsuccessfulRatingNotHelpfulCount,
    c.unsuccessfulRatingTotal,
  ]
  validRatings = get_valid_ratings(ratings, noteStatusHistory, scoredNotes)
  ratingCounts = validRatings.groupby(c.raterParticipantIdKey).sum()[ratingCountRows]

  ratingsWithScores = get_ratings_with_scores(ratings, noteStatusHistory, scoredNotes)
  historyCounts = ratingsWithScores.groupby(c.raterParticipantIdKey).sum()[
    [c.afterDecisionBoolKey, c.awaitingMoreRatingsBoolKey]
  ]
  historyCounts[c.ratedAfterDecision] = historyCounts[c.afterDecisionBoolKey]
  historyCounts[c.ratingsAwaitingMoreRatings] = historyCounts[c.awaitingMoreRatingsBoolKey]

  ratingCounts = ratingCounts.merge(historyCounts, on=c.raterParticipantIdKey, how="outer")
  for rowName in ratingCountRows:
    ratingCounts[rowName] = ratingCounts[rowName].fillna(0)
  return ratingCounts


def _sum_first_n(n):
  """
  A helper function that sums the first n values in a series.

  Args:
      n (int): The number of values to sum
  Returns:
      function: The function
  """

  def _sum(x):
    return x.iloc[:n].sum()

  return _sum


class DictCopyMissing(dict):
  def __missing__(self, key):
    return key


def _sort_nmr_status_last(x: pd.Series) -> pd.Series:
  """
  A helper that sorts notes with NMR status last. This key function is used by sort_values
  to transform the ratingStatus to the ints in nmrSortLast
  """
  # We perform this complex sort because we need to make sure to count NMR notes for users that
  # have no CRH / CRNH notes. Explicitly filtering out these notes would lead to situations where
  # the user would end up without an enrollment state. We perform a key based sorting in descending
  # order. The nmrSortLast transforms CRH + CRNH notes to the beginning of the frame. The noteIdkey
  # (snowflake id) acts a secondary filter to make sure that we are checking for recent notes.
  nmrSortLast = DictCopyMissing(
    {c.needsMoreRatings: 0, c.currentlyRatedHelpful: 1, c.currentlyRatedNotHelpful: 1}
  )
  return x.map(nmrSortLast)


def _get_visible_note_counts(
  scoredNotes: pd.DataFrame, lastNNotes: int = -1, countNMRNotesLast: bool = False
):
  """
  Given scored notes from the algorithm, this function aggregates the note status by note author.

  Args:
      scoredNotes: Notes scored from MF + contributor stats
      lastNNotes: Only count the last n notes
      countNMRNotesLast: Count the NMR notes last. Only affects lastNNNotes counts.
  Returns:
      pd.DataFrame: noteCounts The visible note counts
  """
  sort_by = [c.ratingStatusKey, c.noteIdKey] if countNMRNotesLast else c.noteIdKey
  key_function = _sort_nmr_status_last if countNMRNotesLast else None
  groupAuthorCounts = (
    scoredNotes.sort_values(sort_by, ascending=False, key=key_function)
    .groupby(c.noteAuthorParticipantIdKey)
    .agg(
      {
        c.currentlyRatedHelpfulBoolKey: _sum_first_n(lastNNotes),
        c.currentlyRatedNotHelpfulBoolKey: _sum_first_n(lastNNotes),
        c.awaitingMoreRatingsBoolKey: _sum_first_n(lastNNotes),
        c.numRatingsKey: _sum_first_n(lastNNotes),
      }
    )
    if lastNNotes > 0
    else scoredNotes.groupby(c.noteAuthorParticipantIdKey).sum()
  )
  authorCounts = groupAuthorCounts[
    [
      c.currentlyRatedHelpfulBoolKey,
      c.currentlyRatedNotHelpfulBoolKey,
      c.awaitingMoreRatingsBoolKey,
      c.numRatingsKey,
    ]
  ]
  authorCounts[c.notesCurrentlyRatedHelpful] = authorCounts[c.currentlyRatedHelpfulBoolKey]
  authorCounts[c.notesCurrentlyRatedNotHelpful] = authorCounts[c.currentlyRatedNotHelpfulBoolKey]
  authorCounts[c.notesAwaitingMoreRatings] = authorCounts[c.awaitingMoreRatingsBoolKey]
  authorCounts[c.aggregateRatingReceivedTotal] = authorCounts[c.numRatingsKey]
  authorCounts.fillna(
    inplace=True,
    value={
      c.notesCurrentlyRatedHelpful: 0,
      c.notesCurrentlyRatedNotHelpful: 0,
      c.notesAwaitingMoreRatings: 0,
    },
  )
  return authorCounts


def get_contributor_scores(
  scoredNotes: pd.DataFrame,
  ratings: pd.DataFrame,
  statusHistory: pd.DataFrame,
  lastNNotes=-1,
  countNMRNotesLast: bool = False,
  logging: bool = True,
) -> pd.DataFrame:
  """
  Given the outputs of the MF model, this function aggregates stats over notes and ratings. The
  contributor scores are merged and attached to helfpulness scores in the algorithm.

  Args:
      scoredNotes (pd.DataFrame): scored notes
      ratings (pd.DataFrame): all ratings
      statusHistory (pd.DataFrame): history of note statuses
      lastNNotes (int): count over the last n notes
      countNMRNotesLast (bool): count NMR notes last. Useful when you want to calculate over a limited set of CRH + CRNH notes
      logging (bool): Should we log?
  Returns:
      pd.DataFrame: contributorScores - rating + note aggregates per contributor.
  """
  visibleRatingCounts = _get_visible_rating_counts(scoredNotes, ratings, statusHistory)
  visibleNoteCounts = _get_visible_note_counts(scoredNotes, lastNNotes, countNMRNotesLast)
  contributorCounts = (
    visibleRatingCounts.join(visibleNoteCounts, lsuffix="note", rsuffix="rater", how="outer")
    .reset_index()
    .rename({"index": c.raterParticipantIdKey}, axis=1)[
      [
        c.raterParticipantIdKey,
        c.notesCurrentlyRatedHelpful,
        c.notesCurrentlyRatedNotHelpful,
        c.notesAwaitingMoreRatings,
        c.successfulRatingHelpfulCount,
        c.successfulRatingNotHelpfulCount,
        c.successfulRatingTotal,
        c.unsuccessfulRatingHelpfulCount,
        c.unsuccessfulRatingNotHelpfulCount,
        c.unsuccessfulRatingTotal,
        c.ratedAfterDecision,
        c.ratingsAwaitingMoreRatings,
        c.aggregateRatingReceivedTotal,
      ]
    ]
  )

  if logging:
    print("Number Contributor Counts: ", len(contributorCounts))

  return contributorCounts
