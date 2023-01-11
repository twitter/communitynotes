from collections import Counter
from typing import List, Optional

import constants as c

import pandas as pd


def top_tags(
  row: pd.Series,
  minRatingsToGetTag: int,
  minTagsNeededForStatus: int,
  tagsConsidered: Optional[List[str]] = None,
) -> pd.Series:
  """Given a particular row of the scoredNotes DataFrame, determine which two
  explanation tags to assign to the note based on its ratings.

  See https://twitter.github.io/communitynotes/ranking-notes/#determining-note-status-explanation-tags

  Args:
      row (pd.Series): row of the scoredNotes dataframe, including a count of each tag
      minRatingsToGetTag (int): min ratings needed
      minTagsNeededForStatus (int): min tags needed before a note gets a status
      tagsConsidered (list[str]): set of tags to consider for *all* notes

  Returns:
      Tuple: return the whole row back, with rating tag fields filled in.
  """
  if tagsConsidered:
    tagCounts = pd.DataFrame(row[tagsConsidered])
  elif row[c.ratingStatusKey] == c.currentlyRatedHelpful:
    tagCounts = pd.DataFrame(row[c.helpfulTagsTiebreakOrder])
  elif row[c.ratingStatusKey] == c.currentlyRatedNotHelpful:
    tagCounts = pd.DataFrame(row[c.notHelpfulTagsTiebreakOrder])
  else:
    return row
  tagCounts.columns = [c.tagCountsKey]
  tagCounts[c.tiebreakOrderKey] = range(len(tagCounts))
  tagCounts = tagCounts[tagCounts[c.tagCountsKey] >= minRatingsToGetTag]
  topTags = tagCounts.sort_values(by=[c.tagCountsKey, c.tiebreakOrderKey], ascending=False)[:2]
  # Note: this currently only allows for minTagsNeededForStatus between 0-2
  if len(topTags) >= minTagsNeededForStatus:
    if len(topTags):
      row[c.firstTagKey] = topTags.index[0]
    if len(topTags) > 1:
      row[c.secondTagKey] = topTags.index[1]

  return row


def get_top_nonhelpful_tags_per_author(
  noteStatusHistory: pd.DataFrame, reputationFilteredRatings: pd.DataFrame
):
  """Identifies the top non-helpful tags per author.

  We identify the top two non-helpful tags per author by:
  1. Identifying the top two non-helpful tags for each note.
  2. Grouping notes by author to find the top two tags across all notes
     by the author.

  Args:
      noteStatusHistory (pd.DataFrame): DF mapping notes to authors
      reputationFilteredRatings (pd.DataFrame): DF including ratings and helpfulness tags

  Returns:
    pd.DataFrame with one row per author containing the author ID and top
    two non-helpful tags associated with the author
  """
  # Finds the top two non-helpful tags per note.
  noteTagTotals = (
    reputationFilteredRatings.groupby(c.noteIdKey).sum(numeric_only=True).reset_index()
  )
  # Default tags to the empty string.
  noteTagTotals[c.firstTagKey] = ""
  noteTagTotals[c.secondTagKey] = ""
  noteTopTags = noteTagTotals.apply(
    lambda row: top_tags(
      row,
      c.minRatingsToGetTag,
      c.minTagsNeededForStatus,
      tagsConsidered=c.notHelpfulTagsTiebreakOrder,
    ),
    axis=1,
  )[[c.noteIdKey, c.firstTagKey, c.secondTagKey]]
  # Aggreagtes top two tags per author.
  notesToUse = noteStatusHistory[[c.noteAuthorParticipantIdKey, c.noteIdKey]]
  authorTagsAgg = (
    notesToUse.merge(noteTopTags, on=[c.noteIdKey])
    .groupby(c.noteAuthorParticipantIdKey)
    .agg(Counter)
  )
  # Chooses top two tags per author.
  def _set_top_tags(row: pd.Series) -> pd.Series:
    # Note that row[c.firstTagKey] and row[c.secondTagKey] are both Counter
    # objects mapping tags to counts due to the aggregation above.
    tagTuples = []
    for tag, count in (row[c.firstTagKey] + row[c.secondTagKey]).items():
      if not tag:
        # Skip the empty string, which indicates no tag was assigned.
        continue
      tagTuples.append(
        (count, c.notHelpfulTagsTiebreakMapping[tag], c.notHelpfulTagsEnumMapping[tag])
      )
    tagTuples = sorted(tagTuples, reverse=True)
    topNotHelpfulTags = ",".join(
      map(str, sorted([tagTuples[i][2] for i in range(min(len(tagTuples), 2))]))
    )
    row[c.authorTopNotHelpfulTagValues] = topNotHelpfulTags
    return row

  return authorTagsAgg.apply(_set_top_tags, axis=1).reset_index()[
    [c.noteAuthorParticipantIdKey, c.authorTopNotHelpfulTagValues]
  ]
