import constants as c

import numpy as np
import pandas as pd


def _author_helpfulness(
  scoredNotes: pd.DataFrame,
  CRNHMultiplier: float = 5.0,
) -> pd.DataFrame:
  """Computes author helpfulness scores as described in:
  https://twitter.github.io/birdwatch/contributor-scores/#author-helpfulness-scores

  Args:
      scoredNotes (pd.DataFrame): one row per note, containing preliminary note statuses
      CRNHMultiplier (float): how much more to penalize CRNH notes written vs. reward CRH notes

  Returns:
      pd.DataFrame: one row per author, containing columns for author helpfulness scores
  """
  authorCounts = scoredNotes.groupby(c.participantIdKey).sum()[
    [
      c.currentlyRatedHelpfulBoolKey,
      c.currentlyRatedNotHelpfulBoolKey,
      c.noteCountKey,
      c.noteInterceptKey,
    ]
  ]
  authorCounts[c.crhRatioKey] = (
    authorCounts[c.currentlyRatedHelpfulBoolKey] / authorCounts[c.noteCountKey]
  )
  authorCounts[c.crnhRatioKey] = (
    authorCounts[c.currentlyRatedNotHelpfulBoolKey] / authorCounts[c.noteCountKey]
  )
  authorCounts[c.crhCrnhRatioDifferenceKey] = authorCounts[c.crhRatioKey] - (
    authorCounts[c.crnhRatioKey] * CRNHMultiplier
  )
  authorCounts[c.meanNoteScoreKey] = authorCounts[c.noteInterceptKey] / authorCounts[c.noteCountKey]

  return authorCounts


def _get_ratings_before_note_status_and_public_tsv(
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  logging: bool = True,
) -> pd.DataFrame:
  """Determine which ratings are made before note's most recent non-NMR status,
  and before we could've released any information in the public TSV (48 hours after note creation).

  For old notes (created pre-tombstones launch May 19, 2022), take first 5 ratings.

  Args:
      ratings (pd.DataFrame)
      noteStatusHistory (pd.DataFrame)
      logging (bool, optional). Defaults to True.
  Returns:
      pd.DataFrame ratings that were created early enough to be valid ratings
  """
  ratingsWithNoteLabelInfo = ratings[
    [c.raterParticipantIdKey, c.noteIdKey, c.helpfulNumKey, c.createdAtMillisKey]
  ].merge(
    noteStatusHistory[
      [c.noteIdKey, c.createdAtMillisKey, c.timestampMillisOfNoteMostRecentNonNMRLabelKey]
    ],
    on=c.noteIdKey,
    how="left",
    suffixes=("", "_note"),
  )

  ratingsWithNoteLabelInfo[c.ratingCreatedBeforeMostRecentNMRLabelKey] = (
    pd.isna(ratingsWithNoteLabelInfo[c.timestampMillisOfNoteMostRecentNonNMRLabelKey])
  ) | (
    ratingsWithNoteLabelInfo[c.createdAtMillisKey]
    < ratingsWithNoteLabelInfo[c.timestampMillisOfNoteMostRecentNonNMRLabelKey]
  )

  ratingsWithNoteLabelInfo[c.ratingCreatedBeforePublicTSVReleasedKey] = (
    ratingsWithNoteLabelInfo[c.createdAtMillisKey]
    - ratingsWithNoteLabelInfo[c.createdAtMillisKey + "_note"]
    < c.publicTSVTimeDelay
  )

  noteCreatedBeforeNoteStatusHistory = (
    ratingsWithNoteLabelInfo[c.createdAtMillisKey + "_note"] < c.deletedNoteTombstonesLaunchTime
  )

  first5RatingsOldNotes = (
    ratingsWithNoteLabelInfo[
      (
        noteCreatedBeforeNoteStatusHistory
        & ratingsWithNoteLabelInfo[c.ratingCreatedBeforePublicTSVReleasedKey]
      )
    ][[c.raterParticipantIdKey, c.noteIdKey, c.createdAtMillisKey]]
    .sort_values(c.createdAtMillisKey)
    .groupby(c.noteIdKey)
    .head(c.maxHistoricalValidRatings)
  )[[c.raterParticipantIdKey, c.noteIdKey]].merge(ratingsWithNoteLabelInfo)

  ratingsBeforeStatusNewNotes = ratingsWithNoteLabelInfo[
    (
      np.invert(noteCreatedBeforeNoteStatusHistory)
      & ratingsWithNoteLabelInfo[c.ratingCreatedBeforePublicTSVReleasedKey]
      & ratingsWithNoteLabelInfo[c.ratingCreatedBeforeMostRecentNMRLabelKey]
    )
  ]

  combinedRatingsBeforeStatus = pd.concat([ratingsBeforeStatusNewNotes, first5RatingsOldNotes])

  if logging:
    print(
      f"Total ratings: {np.invert(noteCreatedBeforeNoteStatusHistory).sum()} post-tombstones and {(noteCreatedBeforeNoteStatusHistory).sum()} pre-tombstones"
    )
    print(
      f"Total ratings created before statuses: {len(combinedRatingsBeforeStatus)}, including {len(ratingsBeforeStatusNewNotes)} post-tombstones and {len(first5RatingsOldNotes)} pre-tombstones."
    )

  assert len(combinedRatingsBeforeStatus) <= len(ratingsWithNoteLabelInfo)
  return combinedRatingsBeforeStatus


def _get_valid_ratings(
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  scoredNotes: pd.DataFrame,
  logging: bool = True,
) -> pd.DataFrame:
  """Determine which ratings are "valid" (used to determine rater helpfulness score)

  See definition here: https://twitter.github.io/birdwatch/contributor-scores/#valid-ratings

  Args:
      ratings (pd.DataFrame)
      noteStatusHistory (pd.DataFrame)
      scoredNotes (pd.DataFrame)
      logging (bool, optional): Defaults to True.
  Returns:
      pd.DataFrame: valid ratings
  """
  ratingsBeforeNoteStatus = _get_ratings_before_note_status_and_public_tsv(
    ratings, noteStatusHistory, logging
  )

  ratingsWithScores = ratingsBeforeNoteStatus[
    [c.raterParticipantIdKey, c.helpfulNumKey, c.noteIdKey]
  ].merge(
    scoredNotes[
      [
        c.noteIdKey,
        c.noteInterceptKey,
        c.currentlyRatedHelpfulBoolKey,
        c.currentlyRatedNotHelpfulBoolKey,
      ]
    ],
    on=c.noteIdKey,
  )
  ratingsWithScores[c.ratingCountKey] = 1

  binaryRatingsOnNotesWithStatusLabels = ratingsWithScores[
    (
      ratingsWithScores[c.currentlyRatedHelpfulBoolKey]
      | ratingsWithScores[c.currentlyRatedNotHelpfulBoolKey]
    )
    & ((ratingsWithScores[c.helpfulNumKey] == 1) | (ratingsWithScores[c.helpfulNumKey] == 0))
  ].copy()

  helpfulRatingOnCrhNote = (
    binaryRatingsOnNotesWithStatusLabels[c.currentlyRatedHelpfulBoolKey]
    & binaryRatingsOnNotesWithStatusLabels[c.helpfulNumKey]
    == 1
  )
  notHelpfulRatingOnCrnhNote = (
    binaryRatingsOnNotesWithStatusLabels[c.currentlyRatedNotHelpfulBoolKey]
    & binaryRatingsOnNotesWithStatusLabels[c.helpfulNumKey]
    == 0
  )
  binaryRatingsOnNotesWithStatusLabels[c.ratingAgreesWithNoteStatusKey] = (
    helpfulRatingOnCrhNote | notHelpfulRatingOnCrnhNote
  )

  if logging:
    print(f"Total valid ratings: {len(binaryRatingsOnNotesWithStatusLabels)}")

  return binaryRatingsOnNotesWithStatusLabels


def _rater_helpfulness(
  scoredNotes: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
) -> pd.DataFrame:
  """Computes rater helpfulness scores as described in:
  https://twitter.github.io/birdwatch/contributor-scores/#rater-helpfulness-score

  Args:
      scoredNotes (pd.DataFrame): one row per note containing preliminary note statuses
      ratings (pd.DataFrame)
      noteStatusHistory (pd.DataFrame)

  Returns:
      pd.DataFrame: one row per rater, containing rater helpfulness scores in columns
  """
  validRatings = _get_valid_ratings(ratings, noteStatusHistory, scoredNotes)

  raterCounts = validRatings.groupby(c.raterParticipantIdKey).sum()[
    [c.ratingAgreesWithNoteStatusKey, c.ratingCountKey]
  ]
  raterCounts[c.raterAgreeRatioKey] = (
    raterCounts[c.ratingAgreesWithNoteStatusKey] / raterCounts[c.ratingCountKey]
  )
  return raterCounts


def _compute_scored_notes_for_helpfulness_scores(
  noteParams: pd.DataFrame, noteStatusHistory: pd.DataFrame
) -> pd.DataFrame:
  """Preliminarily score notes, which will be used to determine user helpfulness scores.

  Args:
      noteParams (pd.DataFrame): note parameters output from the matrix factorization
      noteStatusHistory (pd.DataFrame)

  Returns:
      pd.DataFrame: one row per note containing preliminary note scores
  """
  scoredNotes = noteParams.merge(
    noteStatusHistory[[c.noteIdKey, c.participantIdKey]], on=c.noteIdKey
  )
  assert len(scoredNotes) == len(noteParams)

  scoredNotes[c.currentlyRatedHelpfulBoolKey] = False
  scoredNotes[c.currentlyRatedNotHelpfulBoolKey] = False
  scoredNotes.loc[
    (scoredNotes[c.noteInterceptKey] >= c.crhThreshold), c.currentlyRatedHelpfulBoolKey
  ] = True
  scoredNotes.loc[
    (
      scoredNotes[c.noteInterceptKey]
      <= c.crnhThresholdIntercept
      + c.crnhThresholdNoteFactorMultiplier * np.abs(scoredNotes[c.noteFactor1Key])
    ),
    c.currentlyRatedNotHelpfulBoolKey,
  ] = True
  scoredNotes[c.noteCountKey] = 1

  return scoredNotes


def compute_general_helpfulness_scores(
  noteParams: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  logging: bool = True,
) -> pd.DataFrame:
  """Given the note parameters output from matrix factorization, compute helpfulness scores.
  Author helpfulness scores are based on the scores of the notes you wrote.
  Rater helpfulness scores are based on how the ratings you made match up with note scores.
  See https://twitter.github.io/birdwatch/contributor-scores/.

  Args:
      noteParams pandas.DataFrame: note parameters learned from matrix factorization.
      ratings pandas.DataFrame: raw ratings
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
      CRNHMultiplier (float, optional): How much to penalize CRNH relative to CRH in author reputation. Defaults to 5.0.
      logging (bool, optional). Whether to log debug output. Defaults to True.
  Returns:
      helpfulness_scores pandas.DataFrame: 1 row per user, with helpfulness scores as columns.
  """

  scoredNotes = _compute_scored_notes_for_helpfulness_scores(noteParams, noteStatusHistory)
  authorCounts = _author_helpfulness(scoredNotes)
  raterCounts = _rater_helpfulness(scoredNotes, ratings, noteStatusHistory)

  helpfulnessScores = (
    authorCounts.join(raterCounts, how="outer", lsuffix="_author", rsuffix="_rater")
    .reset_index()
    .rename({"index": c.raterParticipantIdKey}, axis=1)[
      [
        c.raterParticipantIdKey,
        c.crhCrnhRatioDifferenceKey,
        c.meanNoteScoreKey,
        c.raterAgreeRatioKey,
      ]
    ]
  )

  helpfulnessScores["aboveThreshold"] = (
    (
      (helpfulnessScores[c.crhCrnhRatioDifferenceKey] >= c.minCRHVsCRNHRatio)
      & (helpfulnessScores[c.meanNoteScoreKey] >= c.minMeanNoteScore)
    )
    | (
      pd.isna(helpfulnessScores[c.crhCrnhRatioDifferenceKey])
      & pd.isna(helpfulnessScores[c.meanNoteScoreKey])
    )
    | (
      pd.isna(helpfulnessScores[c.crhCrnhRatioDifferenceKey])
      & helpfulnessScores[c.meanNoteScoreKey]
      >= c.minMeanNoteScore
    )
  ) & (helpfulnessScores[c.raterAgreeRatioKey] >= c.minRaterAgreeRatio)

  return helpfulnessScores


def filter_ratings_by_helpfulness_scores(
  ratingsForTraining: pd.DataFrame,
  helpfulnessScores: pd.DataFrame,
  logging: bool = True,
):
  """Filter out ratings from raters whose helpfulness scores are too low.
  See https://twitter.github.io/birdwatch/contributor-scores/#filtering-ratings-based-on-helpfulness-scores.

  Args:
      ratingsForTraining pandas.DataFrame: unfiltered input ratings
      helpfulnessScores pandas.DataFrame: helpfulness scores to use to determine which raters to filter out.
      logging (bool, optional): debug output. Defaults to True.

  Returns:
      filtered_ratings pandas.DataFrame: same schema as input ratings, but filtered.
  """

  includedUsers = helpfulnessScores[helpfulnessScores["aboveThreshold"]][[c.raterParticipantIdKey]]

  ratingsHelpfulnessScoreFiltered = includedUsers.merge(
    ratingsForTraining, on=c.raterParticipantIdKey
  )

  if logging:
    print("Unique Raters: ", len(np.unique(ratingsForTraining[c.raterParticipantIdKey])))
    print("People (Authors or Raters) With Helpfulness Scores: ", len(helpfulnessScores))
    print("Raters Included Based on Helpfulness Scores: ", len(includedUsers))
    print(
      "Included Raters who have rated at least 1 note in the final dataset: ",
      len(np.unique(ratingsHelpfulnessScoreFiltered[c.raterParticipantIdKey])),
    )
    print("Number of Ratings Used For 1st Training: ", len(ratingsForTraining))
    print("Number of Ratings for Final Training: ", len(ratingsHelpfulnessScoreFiltered))

  return ratingsHelpfulnessScoreFiltered
