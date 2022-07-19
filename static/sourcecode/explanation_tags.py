import constants as c

import numpy as np
import pandas as pd


def _top_tags(row: pd.Series, minRatingsToGetTag: int, minTagsNeededForStatus: int) -> pd.Series:
  """Given a particular row of the scoredNotes DataFrame, determine which two
  explanation tags to assign to the note based on its ratings.

  See https://twitter.github.io/birdwatch/ranking-notes/#determining-note-status-explanation-tags

  Args:
      row (pd.Series): row of the scoredNotes dataframe, including a count of each tag
      minRatingsToGetTag (int): min ratings needed
      minTagsNeededForStatus (int): min tags needed before a note gets a status

  Returns:
      Tuple: return the whole row back, with rating tag fields filled in.
  """
  if row[c.ratingStatusKey] == c.currentlyRatedHelpful:
    tagCounts = pd.DataFrame(row[c.helpfulTagsTiebreakOrder])
  elif row[c.ratingStatusKey] == c.currentlyRatedNotHelpful:
    tagCounts = pd.DataFrame(row[c.notHelpfulTagsTiebreakOrder])
  else:
    return row
  tagCounts.columns = [c.tagCountsKey]
  tagCounts[c.tiebreakOrderKey] = range(len(tagCounts))
  tagCounts = tagCounts[tagCounts[c.tagCountsKey] >= minRatingsToGetTag]
  topTags = tagCounts.sort_values(by=[c.tagCountsKey, c.tiebreakOrderKey], ascending=False)[:2]
  if len(topTags) < minTagsNeededForStatus:
    row[c.ratingStatusKey] = c.needsMoreRatings
  else:
    row[c.firstTagKey] = topTags.index[0]
    row[c.secondTagKey] = topTags.index[1]
  return row


def get_rating_status_and_explanation_tags(
  ratings: pd.DataFrame,
  noteParams: pd.DataFrame,
  minRatingsNeeded: int = c.minRatingsNeeded,
  minRatingsToGetTag: int = c.minRatingsToGetTag,
  minTagsNeededForStatus: int = c.minTagsNeededForStatus,
  crhThreshold: float = c.crhThreshold,
  crnhThresholdIntercept: float = c.crnhThresholdIntercept,
  crnhThresholdNoteFactorMultiplier: float = c.crnhThresholdNoteFactorMultiplier,
  logging: bool = True,
) -> pd.DataFrame:
  """Given the parameters (intercept score) each note received from the algorithm,
  assign a status to each note. Also assign rating explanation tags to each note
  that's currently rated helpful or not helpful.

  See https://twitter.github.io/birdwatch/ranking-notes/#note-status

  Args:
      ratings (pd.DataFrame): all ratings
      noteParams (pd.DataFrame): scored notes
      minRatingsNeeded (int): min overall ratings needed to get status or tags
      minRatingsToGetTag (int): min times a particular tag needs to be given
      minTagsNeededForStatus (int): min number of tags a note needs to receive a status
      crhThreshold (float): above this score, notes get currently rated helpful status
      crnhThresholdIntercept (float): intercept to determine crnh status
      crnhThresholdNoteFactorMultiplier (float): slope/multiplier to determine crnh status
      logging (bool, optional): debug output. Defaults to True.

  Returns:
      pd.DataFrame: scoredNotes, including note params, tags, and aggregated ratings.
  """
  ratingsToUse = ratings[
    [c.noteIdKey, c.helpfulNumKey] + c.helpfulTagsTSVOrder + c.notHelpfulTagsTSVOrder
  ].copy()

  ratingsToUse[c.numRatingsKey] = 1
  scoredNotes = ratingsToUse.groupby(c.noteIdKey).sum()
  scoredNotes = scoredNotes.merge(
    noteParams[[c.noteIdKey, c.noteInterceptKey, c.noteFactor1Key]], on=c.noteIdKey
  )

  scoredNotes[c.ratingStatusKey] = c.needsMoreRatings
  scoredNotes.loc[
    (scoredNotes[c.numRatingsKey] >= minRatingsNeeded)
    & (scoredNotes[c.noteInterceptKey] >= crhThreshold),
    c.ratingStatusKey,
  ] = c.currentlyRatedHelpful
  scoredNotes.loc[
    (scoredNotes[c.numRatingsKey] >= minRatingsNeeded)
    & (
      scoredNotes[c.noteInterceptKey]
      <= crnhThresholdIntercept
      + crnhThresholdNoteFactorMultiplier * np.abs(scoredNotes[c.noteFactor1Key])
    ),
    c.ratingStatusKey,
  ] = c.currentlyRatedNotHelpful

  scoredNotes[c.firstTagKey] = np.nan
  scoredNotes[c.secondTagKey] = np.nan

  assert scoredNotes.columns.tolist() == c.scoredNotesColumns
  scoredNotes = scoredNotes.apply(
    lambda row: _top_tags(row, minRatingsToGetTag, minTagsNeededForStatus), axis=1
  )

  if logging:
    print("Num Scored Notes:", len(scoredNotes))
  return scoredNotes
