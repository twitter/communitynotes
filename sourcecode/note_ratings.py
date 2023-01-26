from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import constants as c, scoring_rules, tag_filter
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
    mismatches = [
      (c, dtype, ratingsWithNoteLabelInfoTypes[c])
      for c, dtype in zip(ratingsWithNoteLabelInfo, ratingsWithNoteLabelInfo.dtypes)
      if dtype != ratingsWithNoteLabelInfoTypes[c]
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

  See definition here: https://twitter.github.io/communitynotes/contributor-scores/#valid-ratings

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
  raterParams: Optional[pd.DataFrame],
  noteStatusHistory: pd.DataFrame,
  minRatingsNeeded: int = c.minRatingsNeeded,
  crhThreshold: float = c.crhThreshold,
  crnhThresholdIntercept: float = c.crnhThresholdIntercept,
  crnhThresholdNoteFactorMultiplier: float = c.crnhThresholdNoteFactorMultiplier,
  finalRound: bool = False,
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
    1000
    * (
      datetime.fromtimestamp(c.epochMillis / 1000, tz=timezone.utc)
      - timedelta(days=c.emergingWriterDays)
    ).timestamp()
  )
  ratingsToUse = pd.DataFrame(
    ratings[[c.noteIdKey, c.helpfulNumKey] + c.helpfulTagsTSVOrder + c.notHelpfulTagsTSVOrder]
  )
  ratingsToUse.loc[:, c.numRatingsKey] = 1
  ratingsToUse.loc[:, c.numRatingsLast28DaysKey] = False
  ratingsToUse.loc[ratings[c.createdAtMillisKey] > last28Days, c.numRatingsLast28DaysKey] = True
  # BUG: The line below causes us to sum over createdAtMillisKey, adding up timestamps which we
  # later compare against timestampMillisOfNoteMostRecentNonNMRLabelKey.
  noteStats = ratingsToUse.groupby(c.noteIdKey).sum()

  noteParamsColsToKeep = [c.noteIdKey, c.noteInterceptKey, c.noteFactor1Key]
  for col in c.noteParameterUncertaintyTSVColumns:
    if col in noteParams.columns:
      noteParamsColsToKeep.append(col)
  noteStats = noteStats.merge(noteParams[noteParamsColsToKeep], on=c.noteIdKey)

  how = "outer" if finalRound else "left"
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
    how=how,
  )
  noteStats[c.noteCountKey] = 1

  rules = [
    scoring_rules.DefaultRule(RuleID.INITIAL_NMR, set(), c.needsMoreRatings),
    scoring_rules.RuleFromFunction(
      RuleID.GENERAL_CRH,
      {RuleID.INITIAL_NMR},
      c.currentlyRatedHelpful,
      lambda noteStats: is_crh_function(noteStats, minRatingsNeeded, crhThreshold),
    ),
    scoring_rules.RuleFromFunction(
      RuleID.GENERAL_CRNH,
      {RuleID.INITIAL_NMR},
      c.currentlyRatedNotHelpful,
      lambda noteStats: is_crnh_function(
        noteStats, minRatingsNeeded, crnhThresholdIntercept, crnhThresholdNoteFactorMultiplier
      ),
    ),
    scoring_rules.NMtoCRNH(RuleID.NM_CRNH, {RuleID.INITIAL_NMR}, c.currentlyRatedNotHelpful),
  ]
  if finalRound:
    # Compute tag aggregates only if they are required for tag filtering.
    tagAggregates = tag_filter.get_note_tag_aggregates(ratings, noteParams, raterParams)
    assert len(tagAggregates) == len(noteParams), "there should be one aggregate per scored note"
    noteStats = tagAggregates.merge(noteStats, on=c.noteIdKey, how="outer")

    # Add tag filtering and sticky scoring logic.
    rules.extend(
      [
        scoring_rules.AddCRHInertia(
          RuleID.GENERAL_CRH_INERTIA,
          {RuleID.GENERAL_CRH},
          c.currentlyRatedHelpful,
          c.crhThreshold - c.inertiaDelta,
          c.crhThreshold,
        ),
        scoring_rules.FilterTagOutliers(
          RuleID.TAG_OUTLIER,
          {RuleID.GENERAL_CRH},
          c.needsMoreRatings,
          c.tagFilteringPercentile,
          c.minAdjustedTagWeight,
          c.crhSuperThreshold,
        ),
        scoring_rules.AddCRHInertia(
          RuleID.ELEVATED_CRH_INERTIA,
          {RuleID.TAG_OUTLIER},
          c.currentlyRatedHelpful,
          c.crhSuperThreshold - c.inertiaDelta,
          c.crhSuperThreshold,
        ),
        scoring_rules.InsufficientExplanation(
          RuleID.INSUFFICIENT_EXPLANATION,
          {
            RuleID.GENERAL_CRH,
            RuleID.GENERAL_CRNH,
            RuleID.GENERAL_CRH_INERTIA,
            RuleID.ELEVATED_CRH_INERTIA,
          },
          c.needsMoreRatings,
          c.minRatingsToGetTag,
          c.minTagsNeededForStatus,
        ),
        scoring_rules.ScoringDriftGuard(
          RuleID.SCORING_DRIFT_GUARD,
          {RuleID.INSUFFICIENT_EXPLANATION},
        ),
      ]
    )
  else:
    rules.append(
      scoring_rules.InsufficientExplanation(
        RuleID.INSUFFICIENT_EXPLANATION,
        {
          RuleID.GENERAL_CRH,
          RuleID.GENERAL_CRNH,
        },
        c.needsMoreRatings,
        c.minRatingsToGetTag,
        c.minTagsNeededForStatus,
      ),
    )
  noteStats[c.firstTagKey] = np.nan
  noteStats[c.secondTagKey] = np.nan
  scoredNotes = scoring_rules.apply_scoring_rules(noteStats, rules)
  # Discard the locked status column since it is captured in noteStatusHistory and
  # not necessary for the rest of scoring.
  scoredNotes = scoredNotes.drop(columns=[c.lockedStatusKey])

  return scoredNotes
