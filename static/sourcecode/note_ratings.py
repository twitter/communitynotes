from datetime import datetime, timedelta, timezone
from typing import Callable

import constants as c, scoring_rules
from explanation_tags import top_tags
from scoring_rules import RuleID

import numpy as np
import pandas as pd


def is_crh(scoredNotes, minRatingsNeeded, crhThreshold) -> pd.Series:
  return (scoredNotes[c.numRatingsKey] >= minRatingsNeeded) & (
    scoredNotes[c.noteInterceptKey] >= crhThreshold
  )


def is_crnh(
  scoredNotes, minRatingsNeeded, crnhThresholdIntercept, crnhThresholdNoteFactorMultiplier
) -> pd.Series:
  return (scoredNotes[c.numRatingsKey] >= minRatingsNeeded) & (
    scoredNotes[c.noteInterceptKey]
    <= crnhThresholdIntercept
    + crnhThresholdNoteFactorMultiplier * np.abs(scoredNotes[c.noteFactor1Key])
  )


def get_ratings_before_note_status_and_public_tsv(
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  logging: bool = True,
  doTypeCheck: bool = True,
) -> pd.DataFrame:
  """Determine which ratings are made before note's most recent non-NMR status,
  and before we could've released any information in the public TSV (48 hours after note creation).

  For old notes (created pre-tombstones launch May 19, 2022), take first 5 ratings.

  Args:
      ratings (pd.DataFrame)
      noteStatusHistory (pd.DataFrame)
      logging (bool, optional). Defaults to True.
      doTypeCheck (bool): do asserts to check types.
  Returns:
      pd.DataFrame combinedRatingsBeforeStatus ratings that were created early enough to be valid ratings
  """
  right_suffix = "_note"
  ratingsWithNoteLabelInfo = ratings[
    [c.raterParticipantIdKey, c.noteIdKey, c.helpfulNumKey, c.createdAtMillisKey]
  ].merge(
    noteStatusHistory[
      [c.noteIdKey, c.createdAtMillisKey, c.timestampMillisOfNoteMostRecentNonNMRLabelKey]
    ],
    on=c.noteIdKey,
    how="left",
    suffixes=("", right_suffix),
  )
  # Note that the column types for c.createdAtMillisKey and
  # c.timestampMillisOfNoteMostRecentNonNMRLabelKey are determined at runtime and cannot be statically
  # determined from the code above.  If noteStatusHistory is missing any noteIdKey which is found in
  # ratings, then the missing rows will have NaN values for c.createdAtMillisKey and
  # c.timestampMillisOfNoteMostRecentNonNMRLabelKey, forcing the entire colum to have type np.float.
  # However, if there are no missing values in column noteIdKey then c.createdAtMillisKey and
  # c.timestampMillisOfNoteMostRecentNonNMRLabelKey will retain their int64 types.  The code below
  # coerces both columns to always have float types so the typecheck below will pass.
  ratingsWithNoteLabelInfo[
    [c.createdAtMillisKey + right_suffix, c.timestampMillisOfNoteMostRecentNonNMRLabelKey]
  ] *= 1.0

  if doTypeCheck:
    ratingsWithNoteLabelInfoTypes = c.ratingTSVTypeMapping
    ratingsWithNoteLabelInfoTypes[
      c.createdAtMillisKey + "_note"
    ] = np.float  # float because nullable after merge.
    ratingsWithNoteLabelInfoTypes[
      c.timestampMillisOfNoteMostRecentNonNMRLabelKey
    ] = np.float  # float because nullable.
    ratingsWithNoteLabelInfoTypes[c.helpfulNumKey] = np.float

    assert len(ratingsWithNoteLabelInfo) == len(ratings)

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


def get_ratings_with_scores(
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  scoredNotes: pd.DataFrame,
  logging: bool = True,
  doTypeCheck: bool = True,
) -> pd.DataFrame:
  """
  This funciton merges the note status history, ratings, and scores for later aggregation.

  Args:
      ratings (pd.DataFrame): all ratings
      noteStatusHistory (pd.DataFrame): history of note statuses
      scoredNotes (pd.DataFrame): Notes scored from MF + contributor stats
  Returns:
      pd.DataFrame: binaryRatingsOnNotesWithStatusLabels Binary ratings with status labels
  """
  ratingsBeforeNoteStatus = get_ratings_before_note_status_and_public_tsv(
    ratings, noteStatusHistory, logging, doTypeCheck
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
        c.awaitingMoreRatingsBoolKey,
        c.afterDecisionBoolKey,
      ]
    ],
    on=c.noteIdKey,
  )
  return ratingsWithScores


def get_valid_ratings(
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  scoredNotes: pd.DataFrame,
  logging: bool = True,
  doTypeCheck: bool = True,
) -> pd.DataFrame:
  """Determine which ratings are "valid" (used to determine rater helpfulness score)

  See definition here: https://twitter.github.io/birdwatch/contributor-scores/#valid-ratings

  Args:
      ratings (pd.DataFrame)
      noteStatusHistory (pd.DataFrame)
      scoredNotes (pd.DataFrame)
      logging (bool, optional): Defaults to True.
      doTypeCheck (bool): do asserts to check types.
  Returns:
      pd.DataFrame: binaryRatingsOnNotesWithStatusLabels CRH/CRNH notes group by helpfulness
  """
  ratingsWithScores = get_ratings_with_scores(
    ratings, noteStatusHistory, scoredNotes, logging, doTypeCheck
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
  ) & (binaryRatingsOnNotesWithStatusLabels[c.helpfulNumKey] == 1)
  notHelpfulRatingOnCrhNote = (
    binaryRatingsOnNotesWithStatusLabels[c.currentlyRatedHelpfulBoolKey]
  ) & (binaryRatingsOnNotesWithStatusLabels[c.helpfulNumKey] == 0)
  helpfulRatingOnCrnhNote = (
    binaryRatingsOnNotesWithStatusLabels[c.currentlyRatedNotHelpfulBoolKey]
  ) & (binaryRatingsOnNotesWithStatusLabels[c.helpfulNumKey] == 1)
  notHelpfulRatingOnCrnhNote = (
    binaryRatingsOnNotesWithStatusLabels[c.currentlyRatedNotHelpfulBoolKey]
  ) & (binaryRatingsOnNotesWithStatusLabels[c.helpfulNumKey] == 0)

  binaryRatingsOnNotesWithStatusLabels[c.successfulRatingHelpfulCount] = False
  binaryRatingsOnNotesWithStatusLabels[c.successfulRatingNotHelpfulCount] = False
  binaryRatingsOnNotesWithStatusLabels[c.successfulRatingTotal] = False
  binaryRatingsOnNotesWithStatusLabels[c.unsuccessfulRatingHelpfulCount] = False
  binaryRatingsOnNotesWithStatusLabels[c.unsuccessfulRatingNotHelpfulCount] = False
  binaryRatingsOnNotesWithStatusLabels[c.unsuccessfulRatingTotal] = False
  binaryRatingsOnNotesWithStatusLabels[c.ratingAgreesWithNoteStatusKey] = False

  binaryRatingsOnNotesWithStatusLabels.loc[
    helpfulRatingOnCrhNote,
    c.successfulRatingHelpfulCount,
  ] = True
  binaryRatingsOnNotesWithStatusLabels.loc[
    notHelpfulRatingOnCrnhNote,
    c.successfulRatingNotHelpfulCount,
  ] = True
  binaryRatingsOnNotesWithStatusLabels.loc[
    helpfulRatingOnCrhNote | notHelpfulRatingOnCrnhNote,
    c.successfulRatingTotal,
  ] = True
  binaryRatingsOnNotesWithStatusLabels.loc[
    notHelpfulRatingOnCrhNote,
    c.unsuccessfulRatingHelpfulCount,
  ] = True
  binaryRatingsOnNotesWithStatusLabels.loc[
    helpfulRatingOnCrnhNote,
    c.unsuccessfulRatingNotHelpfulCount,
  ] = True
  binaryRatingsOnNotesWithStatusLabels.loc[
    notHelpfulRatingOnCrhNote | helpfulRatingOnCrnhNote,
    c.unsuccessfulRatingTotal,
  ] = True
  binaryRatingsOnNotesWithStatusLabels.loc[
    helpfulRatingOnCrhNote | notHelpfulRatingOnCrnhNote, c.ratingAgreesWithNoteStatusKey
  ] = True

  if logging:
    print(f"Total valid ratings: {len(binaryRatingsOnNotesWithStatusLabels)}")

  return binaryRatingsOnNotesWithStatusLabels


def compute_scored_notes(
  ratings: pd.DataFrame,
  noteParams: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  minRatingsNeeded: int = c.minRatingsNeeded,
  crhThreshold: float = c.crhThreshold,
  crnhThresholdIntercept: float = c.crnhThresholdIntercept,
  crnhThresholdNoteFactorMultiplier: float = c.crnhThresholdNoteFactorMultiplier,
  allNotes: bool = False,
  # TODO: We might want to consider inputing only the series here, instead of the whole callable
  is_crh_function: Callable[..., pd.Series] = is_crh,
  is_crnh_function: Callable[..., pd.Series] = is_crnh,
) -> pd.DataFrame:
  """
  Merges note status history, ratings, and model output. It annotes the data frame with
  different note statuses, and features needed to calculate contributor stats.

  Args:
      ratings (pd.DataFrame): all ratings
      noteStatusHistory (pd.DataFrame): history of note statuses
      scoredNotes (pd.DataFrame): Notes scored from MF + contributor stats
  Returns:
      pd.DataFrame: scoredNotes The scored notes
  """
  last28Days = (
    1000 * (datetime.now(tz=timezone.utc) - timedelta(days=c.emergingWriterDays)).timestamp()
  )
  ratingsToUse = ratings[
    [c.noteIdKey, c.helpfulNumKey, c.createdAtMillisKey]
    + c.helpfulTagsTSVOrder
    + c.notHelpfulTagsTSVOrder
  ]
  ratingsToUse[c.numRatingsKey] = 1
  ratingsToUse[c.numRatingsLast28DaysKey] = False
  ratingsToUse.loc[ratings[c.createdAtMillisKey] > last28Days, c.numRatingsLast28DaysKey] = True
  noteStats = ratingsToUse.groupby(c.noteIdKey).sum()
  noteStats = noteStats.merge(
    noteParams[[c.noteIdKey, c.noteInterceptKey, c.noteFactor1Key]], on=c.noteIdKey
  )

  how = "outer" if allNotes else "left"
  noteStats = noteStats.merge(
    noteStatusHistory[
      [
        c.noteIdKey,
        c.createdAtMillisKey,
        c.timestampMillisOfNoteMostRecentNonNMRLabelKey,
        c.noteAuthorParticipantIdKey,
        c.participantIdKey,
      ]
    ],
    on=c.noteIdKey,
    how=how,
    suffixes=("", "_note"),
  )

  noteStats[c.noteCountKey] = 1
  noteStats[c.afterDecisionBoolKey] = False
  noteStats.loc[
    (noteStats[c.createdAtMillisKey] > noteStats[c.timestampMillisOfNoteMostRecentNonNMRLabelKey])
    & np.invert(pd.isna(c.timestampMillisOfNoteMostRecentNonNMRLabelKey)),
    c.afterDecisionBoolKey,
  ] = True

  rules = [
    scoring_rules.DefaultRule(RuleID.INITIAL_NMR, c.needsMoreRatings, set()),
    scoring_rules.RuleFromFunction(
      RuleID.GENERAL_CRH,
      c.currentlyRatedHelpful,
      {RuleID.INITIAL_NMR},
      lambda noteStats: is_crh_function(noteStats, minRatingsNeeded, crhThreshold),
    ),
    scoring_rules.RuleFromFunction(
      RuleID.GENERAL_CRNH,
      c.currentlyRatedNotHelpful,
      {RuleID.INITIAL_NMR},
      lambda noteStats: is_crnh_function(
        noteStats, minRatingsNeeded, crnhThresholdIntercept, crnhThresholdNoteFactorMultiplier
      ),
    ),
  ]
  scoredNotes = scoring_rules.apply_scoring_rules(noteStats, rules)

  # We need to add apply the top tag counter, because it changes the ratingStatusKey
  # to needsMoreRatings for rows that do not have a critical number of statuses.
  scoredNotes[c.firstTagKey] = np.nan
  scoredNotes[c.secondTagKey] = np.nan
  scoredNotes = scoredNotes.apply(
    lambda row: top_tags(row, c.minRatingsToGetTag, c.minTagsNeededForStatus), axis=1
  )

  return scoredNotes
