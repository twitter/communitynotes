from typing import List

from . import constants as c

import numpy as np
import pandas as pd


def get_top_two_tags_for_note(
  noteStats: pd.DataFrame,
  minRatingsToGetTag: int,
  minTagsNeededToGetStatus: int,
  tagsConsideredInTiebreakOrder: List[str],
) -> pd.DataFrame:
  """Given a scoredNotes DataFrame, determine which two
  explanation tags to assign to each note based on its ratings.

  See https://twitter.github.io/communitynotes/ranking-notes/#determining-note-status-explanation-tags

  Args:
      noteStats (pd.DataFrame): row of the scoredNotes dataframe, including a count of each tag
      minRatingsToGetTag (int): min ratings needed
      minTagsNeededForStatus (int): min tags needed before a note gets a status
      tagsConsideredInTiebreakOrder (list[str]): set of tags to consider for *all* notes

  Returns:
      A dataframe back, filtered to just the rows that are assigned tags, with
      c.firstTagKey and c.secondTagKey columns set, and the index set to noteId.
  """
  assert (
    minTagsNeededToGetStatus == 2
  ), f"minTagsNeededToGetStatus was {minTagsNeededToGetStatus} but only implemented for minTagsNeededToGetStatus=2"

  with c.time_block("NH Tags: Top 2 per note"):
    noteStats.set_index(c.noteIdKey, inplace=True)
    noteTagTotals = noteStats[tagsConsideredInTiebreakOrder[::-1]]  # Put winning tags at front.

    # Filter tags and apply minimum rating threshold
    filteredTags = noteTagTotals.where(lambda x: x >= minRatingsToGetTag)
    filteredTags.dropna(
      thresh=minTagsNeededToGetStatus, inplace=True
    )  # only keep rows with at least 2 non-NaN entries

    negativeTags = -1 * filteredTags.to_numpy(dtype=np.float64, na_value=np.nan)

    # Create a small value for tie-breaking, proportional to the column indices
    # The small value should be smaller than the smallest difference between any two elements (ints)
    epsilon = 1e-3
    tieBreakers = np.arange(negativeTags.shape[1]) * epsilon

    # Add the tie_breaker to the array
    negativeTieBrokenTags = tieBreakers + negativeTags

    # Fill nans with 0 (higher than all other values with nonzero tag counts)
    negativeTieBrokenTags = np.nan_to_num(negativeTieBrokenTags)

    # Use argsort on the modified array
    sortedIndices = np.argsort(negativeTieBrokenTags, axis=1)

    # Extract indices of the two largest values in each row
    topTwoIndices = sortedIndices[:, :2]
    noteTopTags = pd.DataFrame(
      np.array(filteredTags.columns)[topTwoIndices], columns=[c.firstTagKey, c.secondTagKey]
    )
    noteTopTags[c.noteIdKey] = filteredTags.index
    noteTopTags.index = filteredTags.index

    return noteTopTags


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
  with c.time_block("NH Tags: Top 2 per note"):
    # Aggregate ratings by note ID
    tagCols = c.notHelpfulTagsTiebreakOrder[::-1]  # Put winning tags at front.
    noteTagTotals = reputationFilteredRatings[[c.noteIdKey] + tagCols].groupby(c.noteIdKey).sum()

    # Filter tags and apply minimum rating threshold
    filteredTags = noteTagTotals.where(lambda x: x >= c.minRatingsToGetTag)
    filteredTags.dropna(thresh=2, inplace=True)  # only keep rows with at least 2 non-NaN entries

    negativeTags = -1 * filteredTags.to_numpy(dtype=np.float64, na_value=np.nan)

    # Create a small value for tie-breaking, proportional to the column indices
    # The small value should be smaller than the smallest difference between any two elements (ints)
    epsilon = 1e-3
    tieBreakers = np.arange(negativeTags.shape[1]) * epsilon

    # Add the tie_breaker to the array
    negativeTieBrokenTags = tieBreakers + negativeTags

    # Fill nans with 0 (higher than all other values with nonzero tag counts)
    negativeTieBrokenTags = np.nan_to_num(negativeTieBrokenTags)

    # Use argsort on the modified array
    sortedIndices = np.argsort(negativeTieBrokenTags, axis=1)

    # Extract indices of the two largest values in each row
    topTwoIndices = sortedIndices[:, :2]
    noteTopTags = pd.DataFrame(
      np.array(filteredTags.columns)[topTwoIndices], columns=[c.firstTagKey, c.secondTagKey]
    )
    noteTopTags[c.noteIdKey] = filteredTags.index

  with c.time_block("NH Tags: Top 2 per author"):
    # Count how often each tag appears as a note's first or second tag across each author's
    # notes, then keep the two most frequent tags per author (ties broken by tiebreak priority).
    notesToUse = noteStatusHistory[[c.noteAuthorParticipantIdKey, c.noteIdKey]]
    authorNoteTags = notesToUse.merge(noteTopTags, on=[c.noteIdKey], how="inner")
    tagKey = "tag"
    countKey = "count"
    tieBreakKey = "tieBreak"
    authorTagCounts = (
      pd.concat(
        [
          authorNoteTags[[c.noteAuthorParticipantIdKey, tagCol]].rename(columns={tagCol: tagKey})
          for tagCol in [c.firstTagKey, c.secondTagKey]
        ]
      )
      .groupby([c.noteAuthorParticipantIdKey, tagKey], sort=False, dropna=True)
      .size()
      .rename(countKey)
      .reset_index()
    )
    authorTagCounts[tieBreakKey] = authorTagCounts[tagKey].map(c.notHelpfulTagsTiebreakMapping)
    authorTagCounts[c.authorTopNotHelpfulTagValues] = (
      authorTagCounts[tagKey].map(c.notHelpfulTagsEnumMapping).astype(str)
    )
    topTags = (
      authorTagCounts.sort_values([countKey, tieBreakKey], ascending=False)
      .groupby(c.noteAuthorParticipantIdKey, sort=False)
      .head(2)
    )
    topTagValues = topTags.groupby(c.noteAuthorParticipantIdKey, sort=False)[
      c.authorTopNotHelpfulTagValues
    ].agg(",".join)

  with c.time_block("NH Tags: Set Top Tags"):
    # One row per author (sorted, matching groupby semantics); authors without tags get "".
    authors = (
      notesToUse[c.noteAuthorParticipantIdKey]
      .dropna()
      .drop_duplicates()
      .sort_values()
      .reset_index(drop=True)
    )
    topTwoPerAuthor = pd.DataFrame(
      {
        c.noteAuthorParticipantIdKey: authors,
        c.authorTopNotHelpfulTagValues: authors.map(topTagValues).fillna("").astype(object),
      }
    )
  return topTwoPerAuthor
