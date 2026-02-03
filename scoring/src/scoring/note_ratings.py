from datetime import datetime, timedelta, timezone
import logging
from typing import Callable, Dict, Optional

from . import constants as c, incorrect_filter, scoring_rules, tag_filter
from .scoring_rules import RuleID

import numpy as np
import pandas as pd


logger = logging.getLogger("birdwatch.note_ratings")
logger.setLevel(logging.INFO)


# Threshold limiting the number of ratings which can be counted as "valid" for the purpose of
# determining rating performance for notes which were created before noteStatusHistory was
# introduced.  Notice that this value coincides with the minimum number of ratings necessary to
# achieve status.
_maxHistoricalValidRatings = 5


def is_crh(scoredNotes, minRatingsNeeded, crhThreshold) -> pd.Series:
  return (scoredNotes[c.numRatingsKey] >= minRatingsNeeded) & (
    scoredNotes[c.internalNoteInterceptKey] >= crhThreshold
  )


def is_crnh_ucb(scoredNotes, minRatingsNeeded, crnhThresholdUCBIntercept) -> pd.Series:
  enoughRatings = scoredNotes[c.numRatingsKey] >= minRatingsNeeded
  if c.noteInterceptMaxKey in scoredNotes.columns:
    return enoughRatings & (scoredNotes[c.noteInterceptMaxKey] < crnhThresholdUCBIntercept)
  else:
    # all False
    return enoughRatings & (~enoughRatings)


def is_crnh_diamond(
  scoredNotes, minRatingsNeeded, crnhThresholdIntercept, crnhThresholdNoteFactorMultiplier
) -> pd.Series:
  return (scoredNotes[c.numRatingsKey] >= minRatingsNeeded) & (
    scoredNotes[c.internalNoteInterceptKey]
    <= crnhThresholdIntercept
    + crnhThresholdNoteFactorMultiplier * np.abs(scoredNotes[c.internalNoteFactor1Key])
  )


def is_crnh_ratio(
  scoredNotes,
  minRatingsNeeded,
  maxHelpfulNumPerSide=0.4,
  maxAvgHelpfulNumPerSide=0.3,
  minMinSignCount=3,
  do=True,
):
  if not do:
    # return all False
    return (scoredNotes[c.numRatingsKey] < 0) & (scoredNotes[c.numRatingsKey] > 0)
  return (
    (scoredNotes[c.numRatingsKey] >= minRatingsNeeded)
    & (~pd.isna(scoredNotes[c.minSignCountKey]))
    & (~pd.isna(scoredNotes[c.negFactorMeanHelpfulNumKey]))
    & (~pd.isna(scoredNotes[c.posFactorMeanHelpfulNumKey]))
    & (scoredNotes[c.minSignCountKey] >= minMinSignCount)
    & (scoredNotes[c.negFactorMeanHelpfulNumKey] <= maxHelpfulNumPerSide)
    & (scoredNotes[c.posFactorMeanHelpfulNumKey] <= maxHelpfulNumPerSide)
    & (
      ((scoredNotes[c.negFactorMeanHelpfulNumKey] + scoredNotes[c.posFactorMeanHelpfulNumKey]) / 2)
      <= maxAvgHelpfulNumPerSide
    )
  )


def get_ratings_before_note_status_and_public_tsv(
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  log: bool = True,
  doTypeCheck: bool = True,
) -> pd.DataFrame:
  """Determine which ratings are made before note's most recent non-NMR status,
  and before we could've released any information in the public TSV (48 hours after note creation).

  For old notes (created pre-tombstones launch May 19, 2022), take first 5 ratings.

  Args:
      ratings (pd.DataFrame)
      noteStatusHistory (pd.DataFrame)
      log (bool, optional). Defaults to True.
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
    unsafeAllowed={c.createdAtMillisKey},
  )
  # Note that the column types for c.createdAtMillisKey and
  # c.timestampMillisOfNoteMostRecentNonNMRLabelKey are determined at runtime and cannot be statically
  # determined from the code above.  If noteStatusHistory is missing any noteIdKey which is found in
  # ratings, then the missing rows will have NaN values for c.createdAtMillisKey and
  # c.timestampMillisOfNoteMostRecentNonNMRLabelKey, forcing the entire colum to have type float.
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
    ] = float  # float because nullable after merge.
    ratingsWithNoteLabelInfoTypes[
      c.timestampMillisOfNoteMostRecentNonNMRLabelKey
    ] = float  # float because nullable.
    ratingsWithNoteLabelInfoTypes[c.helpfulNumKey] = float

    assert len(ratingsWithNoteLabelInfo) == len(ratings)
    mismatches = [
      (col, dtype, ratingsWithNoteLabelInfoTypes[col])
      for col, dtype in zip(ratingsWithNoteLabelInfo, ratingsWithNoteLabelInfo.dtypes)
      if ("participantid" not in col.lower()) and (dtype != ratingsWithNoteLabelInfoTypes[col])
    ]
    assert not len(mismatches), f"Mismatch columns: {mismatches}"

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
    .head(_maxHistoricalValidRatings)
  )[[c.raterParticipantIdKey, c.noteIdKey]].merge(ratingsWithNoteLabelInfo)

  ratingsBeforeStatusNewNotes = ratingsWithNoteLabelInfo[
    (
      np.invert(noteCreatedBeforeNoteStatusHistory)
      & ratingsWithNoteLabelInfo[c.ratingCreatedBeforePublicTSVReleasedKey]
      & ratingsWithNoteLabelInfo[c.ratingCreatedBeforeMostRecentNMRLabelKey]
    )
  ]

  combinedRatingsBeforeStatus = pd.concat([ratingsBeforeStatusNewNotes, first5RatingsOldNotes])

  if log:
    logger.info(
      f"Total ratings: {np.invert(noteCreatedBeforeNoteStatusHistory).sum()} post-tombstones and {(noteCreatedBeforeNoteStatusHistory).sum()} pre-tombstones"
    )
    logger.info(
      f"Total ratings created before statuses: {len(combinedRatingsBeforeStatus)}, including {len(ratingsBeforeStatusNewNotes)} post-tombstones and {len(first5RatingsOldNotes)} pre-tombstones."
    )

  assert len(combinedRatingsBeforeStatus) <= len(ratingsWithNoteLabelInfo)
  return combinedRatingsBeforeStatus


def get_ratings_with_scores(
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  scoredNotes: pd.DataFrame,
  log: bool = True,
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
    ratings, noteStatusHistory, log, doTypeCheck
  )

  ratingsWithScores = ratingsBeforeNoteStatus[
    [c.raterParticipantIdKey, c.helpfulNumKey, c.noteIdKey, c.createdAtMillisKey]
  ].merge(
    scoredNotes[
      [
        c.noteIdKey,
        c.currentlyRatedHelpfulBoolKey,
        c.currentlyRatedNotHelpfulBoolKey,
        c.awaitingMoreRatingsBoolKey,
      ]
    ],
    on=c.noteIdKey,
  )
  return ratingsWithScores


def get_valid_ratings(
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  scoredNotes: pd.DataFrame,
  log: bool = True,
  doTypeCheck: bool = True,
) -> pd.DataFrame:
  """Determine which ratings are "valid" (used to determine rater helpfulness score)

  See definition here: https://twitter.github.io/communitynotes/contributor-scores/#valid-ratings

  Args:
      ratings (pd.DataFrame)
      noteStatusHistory (pd.DataFrame)
      scoredNotes (pd.DataFrame)
      log (bool, optional): Defaults to True.
      doTypeCheck (bool): do asserts to check types.
  Returns:
      pd.DataFrame: binaryRatingsOnNotesWithStatusLabels CRH/CRNH notes group by helpfulness
  """
  ratingsWithScores = get_ratings_with_scores(
    ratings, noteStatusHistory, scoredNotes, log, doTypeCheck
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

  if log:
    logger.info(f"Total valid ratings: {len(binaryRatingsOnNotesWithStatusLabels)}")

  return binaryRatingsOnNotesWithStatusLabels


def compute_note_stats(ratings: pd.DataFrame, noteStatusHistory: pd.DataFrame) -> pd.DataFrame:
  """Compute aggregate note statics over available ratings and merge in noteStatusHistory fields.

  This function computes note aggregates over ratings and then merges additional fields from
  noteStatusHistory.  In general, we do not expect that every note in noteStatusHistory will
  also appear in ratings (e.g. some notes have no ratings) so the aggregate values for some
  notes will be NaN.  We do expect that all notes observed in ratings will appear in
  noteStatusHistory, and verify that expectation with an assert.

  Note that the content of both ratings and noteStatusHistory may vary across callsites.  For
  example:
  * Scoring models operating on subsets of notes and ratings may pre-filter both
    ratings and noteStatusHistory to only include notes/ratings that are in-scope.
  * During meta scoring we may invoke compute_note_stats with the full set of ratings
    and notes to compute note stats supporting contributor helpfulness aggregates.

  Args:
    ratings (pd.DataFrame): all ratings
    noteStatusHistory (pd.DataFrame): history of note statuses
  Returns:
    pd.DataFrame containing stats about each note
  """
  last28Days = (
    1000
    * (
      datetime.fromtimestamp(c.epochMillis / 1000, tz=timezone.utc)
      - timedelta(days=c.emergingWriterDays)
    ).timestamp()
  )
  ratingsToUse = pd.DataFrame(
    ratings[[c.noteIdKey] + c.helpfulTagsTSVOrder + c.notHelpfulTagsTSVOrder]
  )
  ratingsToUse.loc[:, c.numRatingsKey] = 1
  ratingsToUse.loc[:, c.numPopulationSampledRatingsKey] = 0
  ratingsToUse.loc[
    ratings[c.ratingSourceBucketedKey] == c.ratingSourcePopulationSampledValueTsv,
    c.numPopulationSampledRatingsKey,
  ] = 1
  ratingsToUse.loc[:, c.numRatingsLast28DaysKey] = False
  ratingsToUse.loc[ratings[c.createdAtMillisKey] > last28Days, c.numRatingsLast28DaysKey] = True
  noteStats = ratingsToUse.groupby(c.noteIdKey).sum()

  noteStats = noteStats.merge(
    noteStatusHistory[
      [
        c.noteIdKey,
        c.createdAtMillisKey,
        c.noteAuthorParticipantIdKey,
        c.classificationKey,
        c.currentLabelKey,
        c.lockedStatusKey,
      ]
    ],
    on=c.noteIdKey,
    how="outer",
    unsafeAllowed=set(
      [
        c.numRatingsKey,
        c.numRatingsLast28DaysKey,
        c.numPopulationSampledRatingsKey,
      ]
      + c.helpfulTagsTSVOrder
      + c.notHelpfulTagsTSVOrder
    ),
  )

  # Fill in nan values resulting from the outer merge with zero since these values were not
  # present during aggregation.
  columns = [
    c.numRatingsKey,
    c.numRatingsLast28DaysKey,
    c.numPopulationSampledRatingsKey,
  ] + (c.helpfulTagsTSVOrder + c.notHelpfulTagsTSVOrder)
  noteStats = noteStats.fillna({col: 0 for col in columns})
  noteStats[columns] = noteStats[columns].astype(np.int64)

  # Validate that notes in ratings were a subset of noteStatusHistory.
  assert len(noteStats) == len(noteStatusHistory), "noteStatusHistory should contain all notes"
  return noteStats


def get_note_counts_by_rater_sign(raterModelOutput, ratings):
  raterModelOutput[c.raterParticipantIdKey].astype(ratings[c.raterParticipantIdKey].dtype)

  if c.helpfulNumKey not in ratings.columns:
    ratings[c.helpfulNumKey] = 0.5
    ratings.loc[ratings[c.helpfulnessLevelKey] == "HELPFUL", c.helpfulNumKey] = 1.0
    ratings.loc[ratings[c.helpfulnessLevelKey] == "NOT_HELPFUL", c.helpfulNumKey] = 0.0

  ratingsToUse = pd.DataFrame(ratings[[c.noteIdKey, c.raterParticipantIdKey, c.helpfulNumKey]])
  raterModelOutputToUse = pd.DataFrame(
    raterModelOutput[[c.raterParticipantIdKey, c.internalRaterFactor1Key]]
  )

  mergedRatings = ratingsToUse.merge(raterModelOutputToUse, on=c.raterParticipantIdKey)
  origLength = len(mergedRatings)
  mergedRatings = mergedRatings[mergedRatings[c.internalRaterFactor1Key].notna()]
  logger.info(
    f"dropped {origLength - len(mergedRatings)} out of {origLength} ratings due to NaN factor."
  )

  negFactorKey = "negFactor"
  posFactorKey = "posFactor"
  raterFactorBucketKey = "raterFactorBucket"

  mergedRatings["raterFactorBucket"] = np.where(
    mergedRatings[c.internalRaterFactor1Key] < 0, negFactorKey, posFactorKey
  )

  noteCountsByRaterSign = (
    mergedRatings.groupby([c.noteIdKey, raterFactorBucketKey])
    .size()
    .unstack(fill_value=0)
    .reset_index()
  ).rename(
    columns={negFactorKey: c.negFactorRatingCountKey, posFactorKey: c.posFactorRatingCountKey}
  )

  if c.negFactorRatingCountKey not in noteCountsByRaterSign.columns:
    noteCountsByRaterSign[c.negFactorRatingCountKey] = 0
  if c.posFactorRatingCountKey not in noteCountsByRaterSign.columns:
    noteCountsByRaterSign[c.posFactorRatingCountKey] = 0

  noteCountsByRaterSign[c.minSignCountKey] = noteCountsByRaterSign[
    [c.negFactorRatingCountKey, c.posFactorRatingCountKey]
  ].min(axis=1)

  meanHelpfulnessByRaterSign = (
    mergedRatings.groupby([c.noteIdKey, raterFactorBucketKey])[c.helpfulNumKey]
    .mean()
    .unstack()
    .reset_index()
  ).rename(
    columns={negFactorKey: c.negFactorMeanHelpfulNumKey, posFactorKey: c.posFactorMeanHelpfulNumKey}
  )

  noteCountsByRaterSign = noteCountsByRaterSign.merge(
    meanHelpfulnessByRaterSign, on=[c.noteIdKey], how="left", unsafeAllowed=c.minSignCountKey
  )
  if c.negFactorMeanHelpfulNumKey not in noteCountsByRaterSign.columns:
    noteCountsByRaterSign[c.negFactorMeanHelpfulNumKey] = np.nan
  if c.posFactorMeanHelpfulNumKey not in noteCountsByRaterSign.columns:
    noteCountsByRaterSign[c.posFactorMeanHelpfulNumKey] = np.nan

  # Merge in net minority helpfulness
  counts = mergedRatings[[c.noteIdKey, raterFactorBucketKey, c.helpfulNumKey]]
  counts = pd.crosstab(
    index=counts[c.noteIdKey], columns=[counts[raterFactorBucketKey], counts[c.helpfulNumKey]]
  )
  counts.columns = [f"{col1}_{col2}" for col1, col2 in counts.columns]
  counts = counts.reset_index(drop=False)
  for bucket in [negFactorKey, posFactorKey]:
    for level in [0.0, 0.5, 1.0]:
      col = f"{bucket}_{level}"
      if col not in counts:
        counts[col] = 0
  counts[f"{negFactorKey}_{c.netMinHelpfulKey}"] = (
    counts[f"{negFactorKey}_1.0"] - counts[f"{negFactorKey}_0.0"]
  )
  counts[f"{posFactorKey}_{c.netMinHelpfulKey}"] = (
    counts[f"{posFactorKey}_1.0"] - counts[f"{posFactorKey}_0.0"]
  )
  counts[c.netMinHelpfulKey] = (
    counts[[f"{negFactorKey}_{c.netMinHelpfulKey}", f"{posFactorKey}_{c.netMinHelpfulKey}"]]
    .min(axis=1)
    .clip(lower=0)
  )
  counts = counts.merge(
    mergedRatings[[c.noteIdKey]]
    .value_counts()
    .reset_index(drop=False)
    .rename(columns={"count": "total"})
  )
  counts[c.netMinHelpfulRatioKey] = counts[c.netMinHelpfulKey] / counts["total"]
  noteCountsByRaterSign = noteCountsByRaterSign.merge(
    counts[[c.noteIdKey, c.netMinHelpfulKey, c.netMinHelpfulRatioKey]]
  )

  # Downcast types from 64=>32
  noteCountsByRaterSign[c.minSignCountKey] = noteCountsByRaterSign[c.minSignCountKey].astype(
    np.int32
  )
  noteCountsByRaterSign[c.netMinHelpfulKey] = noteCountsByRaterSign[c.netMinHelpfulKey].astype(
    np.int32
  )
  noteCountsByRaterSign[c.netMinHelpfulRatioKey] = noteCountsByRaterSign[
    c.netMinHelpfulRatioKey
  ].astype(np.float32)
  noteCountsByRaterSign[c.negFactorRatingCountKey] = noteCountsByRaterSign[
    c.negFactorRatingCountKey
  ].astype(np.int32)
  noteCountsByRaterSign[c.posFactorRatingCountKey] = noteCountsByRaterSign[
    c.posFactorRatingCountKey
  ].astype(np.int32)
  noteCountsByRaterSign[c.negFactorMeanHelpfulNumKey] = noteCountsByRaterSign[
    c.negFactorMeanHelpfulNumKey
  ].astype(np.float32)
  noteCountsByRaterSign[c.posFactorMeanHelpfulNumKey] = noteCountsByRaterSign[
    c.posFactorMeanHelpfulNumKey
  ].astype(np.float32)

  return noteCountsByRaterSign[
    [
      c.noteIdKey,
      c.minSignCountKey,
      c.negFactorMeanHelpfulNumKey,
      c.posFactorMeanHelpfulNumKey,
      c.netMinHelpfulKey,
      c.netMinHelpfulRatioKey,
    ]
  ].rename_axis(None, axis=1)


def get_population_sampled_counts_by_rater_sign(raterModelOutput, ratings):
  raterModelOutput[c.raterParticipantIdKey].astype(ratings[c.raterParticipantIdKey].dtype)

  ratingsToUse = pd.DataFrame(
    ratings[[c.noteIdKey, c.raterParticipantIdKey, c.ratingSourceBucketedKey]]
  )
  # Filter to population sampled ratings
  ratingsToUse = ratingsToUse[
    ratingsToUse[c.ratingSourceBucketedKey] == c.ratingSourcePopulationSampledValueTsv
  ]

  if len(ratingsToUse) == 0:
    return pd.DataFrame(
      {
        c.noteIdKey: pd.Series([], dtype=np.int64),
        c.negFactorPopulationSampledRatingCountKey: pd.Series([], dtype=np.int64),
        c.posFactorPopulationSampledRatingCountKey: pd.Series([], dtype=np.int64),
      }
    )

  raterModelOutputToUse = pd.DataFrame(
    raterModelOutput[[c.raterParticipantIdKey, c.internalRaterFactor1Key]]
  )
  mergedRatings = ratingsToUse.merge(raterModelOutputToUse, on=c.raterParticipantIdKey)
  origLength = len(mergedRatings)
  mergedRatings = mergedRatings[mergedRatings[c.internalRaterFactor1Key].notna()]
  logger.info(
    f"dropped {origLength - len(mergedRatings)} out of {origLength} population sampled ratings due to NaN factor."
  )

  negFactorKey = "negFactor"
  posFactorKey = "posFactor"
  raterFactorBucketKey = "raterFactorBucket"

  mergedRatings[raterFactorBucketKey] = np.where(
    mergedRatings[c.internalRaterFactor1Key] < 0, negFactorKey, posFactorKey
  )

  counts = (
    mergedRatings.groupby([c.noteIdKey, raterFactorBucketKey])
    .size()
    .unstack(fill_value=0)
    .reset_index()
  )
  if negFactorKey not in counts.columns:
    counts[negFactorKey] = 0
  if posFactorKey not in counts.columns:
    counts[posFactorKey] = 0

  counts = counts.rename(
    columns={
      negFactorKey: c.negFactorPopulationSampledRatingCountKey,
      posFactorKey: c.posFactorPopulationSampledRatingCountKey,
    }
  )

  counts[c.negFactorPopulationSampledRatingCountKey] = counts[
    c.negFactorPopulationSampledRatingCountKey
  ].astype(np.int64)
  counts[c.posFactorPopulationSampledRatingCountKey] = counts[
    c.posFactorPopulationSampledRatingCountKey
  ].astype(np.int64)

  return counts[
    [
      c.noteIdKey,
      c.negFactorPopulationSampledRatingCountKey,
      c.posFactorPopulationSampledRatingCountKey,
    ]
  ]


# TODO: compute_scored_notes is only called from matrix_factorization_scorer and the behavior is directly
# coupled to the matrix_factorization_scorer results.  When we define a model object, this function should
# become a member of matrix_factorization_scorer and the composure of "rules" should be explicitly factored
# out into a helper function.  This approach would better encapsulate our code / logic and reduce how much
# is in the global namespace, and allow variants of matrix_factorization_scorer to more easily override
# the scoring rule behaviors.
def compute_scored_notes(
  ratings: pd.DataFrame,
  noteParams: pd.DataFrame,
  raterParams: Optional[pd.DataFrame],
  noteStatusHistory: pd.DataFrame,
  minRatingsNeeded: int,
  crhThreshold: float,
  crnhThresholdIntercept: float,
  crnhThresholdNoteFactorMultiplier: float,
  crnhThresholdNMIntercept: float,
  crnhThresholdUCBIntercept: float,
  crhSuperThreshold: Optional[float],
  inertiaDelta: float,
  tagFilterThresholds: Optional[Dict[str, float]],
  incorrectFilterThreshold: float,
  finalRound: bool = False,
  # TODO: We might want to consider inputing only the series here, instead of the whole callable
  is_crh_function: Callable[..., pd.Series] = is_crh,
  is_crnh_diamond_function: Callable[..., pd.Series] = is_crnh_diamond,
  is_crnh_ucb_function: Callable[..., pd.Series] = is_crnh_ucb,
  is_crnh_ratio_function: Callable[..., pd.Series] = is_crnh_ratio,
  lowDiligenceThreshold: float = 0.263,
  factorThreshold: float = 0.5,
  firmRejectThreshold: Optional[float] = None,
  minMinorityNetHelpfulRatings: Optional[int] = None,
  minMinorityNetHelpfulRatio: Optional[float] = None,
  crhThresholdNoHighVol: Optional[float] = None,
  crhThresholdNoCorrelated: Optional[float] = None,
) -> pd.DataFrame:
  """
  Merges note status history, ratings, and model output. It annotes the data frame with
  different note statuses, and features needed to calculate contributor stats.

  Args:
      ratings: All ratings from Community Notes contributors.
      noteParams: Note intercepts and factors returned from matrix factorization.
      raterParams: Rater intercepts and factors returned from matrix factorization.
      noteStatusHistory: History of note statuses.
      minRatingsNeeded: Minimum number of ratings for a note to achieve status.
      crhThrehsold: Minimum intercept for most notes to achieve CRH status.
      crnhThresholdIntercept: Minimum intercept for most notes to achieve CRNH status.
      crnhThresholdNoteFactorMultiplier: Scaling factor making controlling the relationship between
        CRNH threshold and note intercept.  Note that this constant is set negative so that notes with
        larger (magnitude) factors must have proportionally lower intercepts to become CRNH.
      crnhThresholdNMIntercept: Minimum intercept for notes which do not claim a tweet is misleading
        to achieve CRNH status.
      crnhThresholdUCBIntercept: Maximum UCB of the intercept (determined with pseudoraters) for
        notes to achieve CRNH status.
        notes to achieve CRH status.
      crhSuperThreshold: Minimum intercept for notes which have consistent and common patterns of
        repeated reason tags in not-helpful ratings to achieve CRH status.
      inertiaDelta: Minimum amount which a note that has achieve CRH status must drop below the
        applicable threshold to lose CRH status.
      finalRound: If true, enable additional status assignment logic which is only applied when
        determining final status.  Given that these mechanisms add complexity we don't apply them
        in earlier rounds.
      is_crh_function: Function specifying default CRH critierai.
      is_crnh_diamond_function: Function specifying default CRNH critierai.
      is_crnh_ucb_function: Function specifying default CRNH critierai, ORed together with previous.
      is_crnh_ratio_function:
  Returns:
      pd.DataFrame: scoredNotes The scored notes
  """
  # Compute noteStats and drop columns which won't be needed
  noteStats = compute_note_stats(ratings, noteStatusHistory)
  noteStats = noteStats.drop(
    columns=[
      c.numRatingsLast28DaysKey,
      c.createdAtMillisKey,
    ]
  )

  # Get meanHelpfulNum per side and minSignCount for each note
  noteStats = noteStats.merge(
    get_note_counts_by_rater_sign(raterParams, ratings),
    how="left",
    on=c.noteIdKey,
    unsafeAllowed={c.minSignCountKey, c.netMinHelpfulKey},
  )
  noteStats[[c.minSignCountKey]] = noteStats[[c.minSignCountKey]].fillna(0).astype(np.int32)
  noteStats[[c.netMinHelpfulKey]] = noteStats[[c.netMinHelpfulKey]].fillna(0).astype(np.int32)
  noteStats[[c.netMinHelpfulRatioKey]] = (
    noteStats[[c.netMinHelpfulRatioKey]].fillna(0.0).astype(np.float32)
  )

  # Merge with noteParams as necessary
  noteParamsColsToKeep = [c.noteIdKey, c.internalNoteInterceptKey, c.internalNoteFactor1Key]
  if finalRound:
    noteParamsColsToKeep += [c.lowDiligenceNoteInterceptKey]
  for col in c.noteParameterUncertaintyTSVColumns:
    if col in noteParams.columns:
      noteParamsColsToKeep.append(col)
  if crhThresholdNoHighVol is not None:
    noteParamsColsToKeep.append(c.internalNoteInterceptNoHighVolKey)
  if crhThresholdNoCorrelated is not None:
    noteParamsColsToKeep.append(c.internalNoteInterceptNoCorrelatedKey)
  if c.internalNoteInterceptPopulationSampledKey in noteParams.columns:
    noteParamsColsToKeep.append(c.internalNoteInterceptPopulationSampledKey)
  noteStats = noteStats.merge(
    noteParams[noteParamsColsToKeep],
    on=c.noteIdKey,
    how="left",
    unsafeAllowed={"ratingCount_all", "ratingCount_neg_fac", "ratingCount_pos_fac"},
  )

  # Merge per-sign population sampled counts
  popSampledCounts = get_population_sampled_counts_by_rater_sign(raterParams, ratings)
  noteStats = noteStats.merge(
    popSampledCounts,
    on=c.noteIdKey,
    how="left",
    unsafeAllowed={
      c.negFactorPopulationSampledRatingCountKey,
      c.posFactorPopulationSampledRatingCountKey,
    },
  )

  noteStats[
    [
      c.negFactorPopulationSampledRatingCountKey,
      c.posFactorPopulationSampledRatingCountKey,
    ]
  ] = (
    noteStats[
      [
        c.negFactorPopulationSampledRatingCountKey,
        c.posFactorPopulationSampledRatingCountKey,
      ]
    ]
    .fillna(0)
    .astype(np.int64)
  )

  rules = [
    scoring_rules.DefaultRule(RuleID.INITIAL_NMR, set(), c.needsMoreRatings),
    scoring_rules.RuleFromFunction(
      RuleID.GENERAL_CRH,
      {RuleID.INITIAL_NMR},
      c.currentlyRatedHelpful,
      lambda noteStats: is_crh_function(noteStats, minRatingsNeeded, crhThreshold),
      onlyApplyToNotesThatSayTweetIsMisleading=True,
    ),
    scoring_rules.RuleFromFunction(
      RuleID.GENERAL_CRNH,
      {RuleID.INITIAL_NMR},
      c.currentlyRatedNotHelpful,
      lambda noteStats: is_crnh_diamond_function(
        noteStats, minRatingsNeeded, crnhThresholdIntercept, crnhThresholdNoteFactorMultiplier
      ),
      onlyApplyToNotesThatSayTweetIsMisleading=False,
    ),
    scoring_rules.RuleFromFunction(
      RuleID.UCB_CRNH,
      {RuleID.INITIAL_NMR},
      c.currentlyRatedNotHelpful,
      lambda noteStats: is_crnh_ucb_function(
        noteStats, minRatingsNeeded, crnhThresholdUCBIntercept
      ),
      onlyApplyToNotesThatSayTweetIsMisleading=False,
    ),
    scoring_rules.RuleFromFunction(
      RuleID.RATIO_CRNH,
      {RuleID.INITIAL_NMR},
      c.currentlyRatedNotHelpful,
      lambda noteStats: is_crnh_ratio_function(noteStats, minRatingsNeeded, do=finalRound),
      onlyApplyToNotesThatSayTweetIsMisleading=False,
    ),
    scoring_rules.NMtoCRNH(
      RuleID.NM_CRNH, {RuleID.INITIAL_NMR}, c.currentlyRatedNotHelpful, crnhThresholdNMIntercept
    ),
  ]
  if finalRound:
    with c.time_block("compute_scored_notes: compute tag aggregates"):
      # Compute tag aggregates only if they are required for tag filtering.
      tagAggregates = tag_filter.get_note_tag_aggregates(ratings, noteParams, raterParams)

      # set pandas option to display all columns
      pd.set_option("display.max_columns", None)
      assert len(tagAggregates) == len(noteParams), f"""there should be one aggregate per scored note
      len(noteParams) == {len(noteParams)}; len(np.unique(noteParams[c.noteIdKey])) == {len(np.unique(noteParams[c.noteIdKey]))}
      len(tagAggregates) == {len(tagAggregates)}; len(np.unique(tagAggregates[c.noteIdKey])) == {len(np.unique(tagAggregates[c.noteIdKey]))}

      The first 30 notes that appear in noteParams but not in tagAggregates are:
      {noteParams[~noteParams[c.noteIdKey].isin(tagAggregates[c.noteIdKey])].head(30)}

      The first 30 notes that appear in tagAggregates but not in noteParams are:
      {tagAggregates[~tagAggregates[c.noteIdKey].isin(noteParams[c.noteIdKey])].head(30)}
      """

      noteStats = tagAggregates.merge(noteStats, on=c.noteIdKey, how="outer")

    with c.time_block("compute_scored_notes: compute incorrect aggregates"):
      incorrectAggregates = incorrect_filter.get_incorrect_aggregates_final_scoring(
        ratings, noteParams, raterParams
      )
      noteStats = noteStats.merge(
        incorrectAggregates,
        on=c.noteIdKey,
        how="outer",
        unsafeAllowed={
          c.notHelpfulIncorrectIntervalKey,
          c.numVotersIntervalKey,
        },
      )
    assert tagFilterThresholds is not None

    # Add tag filtering and sticky scoring logic.
    rules.extend(
      [
        scoring_rules.AddCRHInertia(
          RuleID.GENERAL_CRH_INERTIA,
          {RuleID.GENERAL_CRH},
          c.currentlyRatedHelpful,
          crhThreshold - inertiaDelta,
          crhThreshold,
          minRatingsNeeded,
        ),
        scoring_rules.FilterTagOutliers(
          RuleID.TAG_OUTLIER,
          {RuleID.GENERAL_CRH},
          c.firmReject if firmRejectThreshold is not None else c.needsMoreRatings,
          tagFilterThresholds=tagFilterThresholds,
        ),
      ]
    )
    if crhSuperThreshold is not None:
      rules.extend(
        [
          scoring_rules.RuleFromFunction(
            RuleID.ELEVATED_CRH,
            {RuleID.INITIAL_NMR},
            c.currentlyRatedHelpful,
            lambda noteStats: is_crh_function(noteStats, minRatingsNeeded, crhSuperThreshold),
            onlyApplyToNotesThatSayTweetIsMisleading=True,
          ),
          scoring_rules.AddCRHInertia(
            RuleID.ELEVATED_CRH_INERTIA,
            {RuleID.TAG_OUTLIER},
            c.currentlyRatedHelpful,
            crhSuperThreshold - inertiaDelta,
            crhSuperThreshold,
            minRatingsNeeded,
          ),
        ]
      )
    rules.extend(
      [
        scoring_rules.FilterIncorrect(
          RuleID.INCORRECT_OUTLIER,
          {RuleID.TAG_OUTLIER},
          c.firmReject if firmRejectThreshold is not None else c.needsMoreRatings,
          tagThreshold=2,
          voteThreshold=3,
          weightedTotalVotes=incorrectFilterThreshold,
        ),
        scoring_rules.FilterLowDiligence(
          RuleID.LOW_DILIGENCE,
          {RuleID.INCORRECT_OUTLIER},
          c.firmReject if firmRejectThreshold is not None else c.needsMoreRatings,
          interceptThreshold=lowDiligenceThreshold,
        ),
        scoring_rules.FilterLargeFactor(
          RuleID.LARGE_FACTOR,
          {RuleID.LOW_DILIGENCE},
          c.firmReject if firmRejectThreshold is not None else c.needsMoreRatings,
          factorThreshold=factorThreshold,
        ),
      ]
    )
    if firmRejectThreshold is not None:
      rules.append(
        scoring_rules.RejectLowIntercept(
          RuleID.LOW_INTERCEPT,
          {RuleID.LARGE_FACTOR},
          c.firmReject,
          firmRejectThreshold,
        )
      )
    if minMinorityNetHelpfulRatings is not None:
      rules.append(
        scoring_rules.RequireMinMinorityRaters(
          RuleID.MIN_MINORITY_RATERS,
          {RuleID.LARGE_FACTOR},
          c.needsYourHelp,
          minMinorityNetHelpfulRatings,
        )
      )
    if minMinorityNetHelpfulRatio is not None:
      rules.append(
        scoring_rules.RequireRaterBalance(
          RuleID.RATER_BALANCE,
          {RuleID.MIN_MINORITY_RATERS},
          c.needsYourHelp,
          minMinorityNetHelpfulRatio,
        )
      )
    if crhThresholdNoHighVol is not None:
      rules.append(
        scoring_rules.NoHighVolIntercept(
          RuleID.NO_HIGH_VOL_INTERCEPT,
          {RuleID.LARGE_FACTOR},
          c.needsYourHelp,
          crhThresholdNoHighVol,
        )
      )
    if crhThresholdNoCorrelated is not None:
      rules.append(
        scoring_rules.NoCorrelatedIntercept(
          RuleID.NO_CORRELATED_INTERCEPT,
          {RuleID.LARGE_FACTOR},
          c.needsYourHelp,
          crhThresholdNoCorrelated,
        )
      )
  scoredNotes = scoring_rules.apply_scoring_rules(
    noteStats, rules, c.internalRatingStatusKey, c.internalActiveRulesKey
  )
  # Discard the locked status column since it is captured in noteStatusHistory and
  # not necessary for the rest of scoring.
  scoredNotes = scoredNotes.drop(columns=[c.lockedStatusKey])
  # Discard extra columns related to the RatioCRNH rule that are no longer needed for the rest of scoring.
  scoredNotes = scoredNotes.drop(
    columns=[
      c.minSignCountKey,
      c.negFactorMeanHelpfulNumKey,
      c.posFactorMeanHelpfulNumKey,
      c.netMinHelpfulKey,
      c.netMinHelpfulRatioKey,
      c.numPopulationSampledRatingsKey,
    ]
  )

  return scoredNotes
