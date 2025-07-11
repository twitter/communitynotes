import logging

from . import constants as c, explanation_tags
from .helpfulness_scores import author_helpfulness
from .note_ratings import get_ratings_with_scores, get_valid_ratings

import pandas as pd


logger = logging.getLogger("birdwatch.contributor_state")
logger.setLevel(logging.INFO)


def is_not_api_user(contributorScoresWithEnrollment: pd.DataFrame):
  """
  Check if a contributor is not an api user.

  Args:
    contributorScoresWithEnrollment: Scored Notes + User Enrollment status
  """
  return (
    (contributorScoresWithEnrollment[c.enrollmentState] != c.apiTestUser)
    & (contributorScoresWithEnrollment[c.enrollmentState] != c.apiEarnedIn)
    & (contributorScoresWithEnrollment[c.enrollmentState] != c.apiEarnedOut)
  )


def should_earn_in(contributorScoresWithEnrollment: pd.DataFrame):
  """
  The participant should earn in when they are in the earnedOutAcknowledged, earnedoutNoAck and newUser state.
  To earn in, we need to check that the rating impact is larger than the succesfully ratings
  needed to earn in. This constant is fixed for new users (ratingImpactForEarnIn), for
  earnedOutNoAcknowledge it will be set int the CombineEventAndSnapshot job to +5 their current
  rating impact with a minimum of ratingImpactForEarnIn.

  Args:
    authorEnrollmentCounts (pd.DataFrame): Scored Notes + User Enrollment status
  """
  return (
    (contributorScoresWithEnrollment[c.enrollmentState] != c.removed)
    & (contributorScoresWithEnrollment[c.enrollmentState] != c.earnedIn)
    & (contributorScoresWithEnrollment[c.enrollmentState] != c.atRisk)
    & (is_not_api_user(contributorScoresWithEnrollment))
    & (
      contributorScoresWithEnrollment[c.ratingImpact]
      >= contributorScoresWithEnrollment[c.successfulRatingNeededToEarnIn]
    )
  )


def newly_at_risk(authorEnrollmentCounts: pd.DataFrame):
  """
  The author is at risk when they have written 2 CRNH notes of the last 5 notes. NewUser
  EarnedOutNoAck, and EarnedOutAcknowledged states cannot transition to this state because they cannot
  write notes, and must first Earn in to Birdwatch.

  Args:
    authorEnrollmentCounts (pd.DataFrame): Scored Notes + User Enrollment status
  """
  return (
    (authorEnrollmentCounts[c.enrollmentState] != c.removed)
    & (authorEnrollmentCounts[c.enrollmentState] != c.newUser)
    & (authorEnrollmentCounts[c.enrollmentState] != c.earnedOutNoAcknowledge)
    & (authorEnrollmentCounts[c.enrollmentState] != c.earnedOutAcknowledged)
    & (authorEnrollmentCounts[c.enrollmentState] != c.atRisk)
    & (is_not_api_user(authorEnrollmentCounts))
    & (authorEnrollmentCounts[c.notesCurrentlyRatedNotHelpful] == c.isAtRiskCRNHCount)
  )


def is_earned_out(authorEnrollmentCounts: pd.DataFrame):
  """
  The author is at earned out when they have written 3+ CRNH notes of the last 5 notes. The user
  loses their ability to write notes once they acknowledge earn out. (EarnedOutAcknowledged) NewUser
  and EarnedOutAcknowledged states cannot transition to this state because they cannot
  write notes, and must first Earn in to Birdwatch.

  Args:
    authorEnrollmentCounts (pd.DataFrame): Scored Notes + User Enrollment status
  """
  return (
    (authorEnrollmentCounts[c.enrollmentState] != c.removed)
    & (authorEnrollmentCounts[c.enrollmentState] != c.newUser)
    & (authorEnrollmentCounts[c.enrollmentState] != c.earnedOutAcknowledged)
    & (is_not_api_user(authorEnrollmentCounts))
    & (authorEnrollmentCounts[c.notesCurrentlyRatedNotHelpful] > c.isAtRiskCRNHCount)
  )


def newly_earned_in(authorEnrollmentCounts):
  """
  The author is at earned out when they have written <2 CRNH notes of the last 5 notes.
  NewUser, EarnedOutNoAck, and EarnedOutAcknowledged states cannot transition to this state because they cannot
  write notes, and must first Earn in to Birdwatch.

  Args:
    authorEnrollmentCounts (pd.DataFrame): Scored Notes + User Enrollment status
  """
  return (
    (authorEnrollmentCounts[c.enrollmentState] != c.removed)
    & (authorEnrollmentCounts[c.enrollmentState] != c.newUser)
    & (authorEnrollmentCounts[c.enrollmentState] != c.earnedOutAcknowledged)
    & (authorEnrollmentCounts[c.enrollmentState] != c.earnedOutNoAcknowledge)
    & (authorEnrollmentCounts[c.enrollmentState] != c.earnedIn)
    & (is_not_api_user(authorEnrollmentCounts))
    & (authorEnrollmentCounts[c.notesCurrentlyRatedNotHelpful] < c.isAtRiskCRNHCount)
  )


def is_top_writer(authorEnrollmentCounts):
  """
  The author is a top writer when they have at least 10 WI and 4% hit rate

  Args:
    authorEnrollmentCounts (pd.DataFrame): Scored Notes + User Enrollment status
  """
  # check whether any notes have been written at all to avoid divide by zero
  totalNotes = (
    authorEnrollmentCounts[c.notesCurrentlyRatedHelpful]
    + authorEnrollmentCounts[c.notesCurrentlyRatedNotHelpful]
    + authorEnrollmentCounts[c.notesAwaitingMoreRatings]
  ).apply(lambda row: max([row, 1]), 1)
  writingImpact = (
    authorEnrollmentCounts[c.notesCurrentlyRatedHelpful]
    - authorEnrollmentCounts[c.notesCurrentlyRatedNotHelpful]
  )
  return (writingImpact >= c.topWriterWritingImpact) & (
    (writingImpact / totalNotes) >= c.topWriterHitRate
  )


def _get_rated_after_decision(
  ratings: pd.DataFrame, noteStatusHistory: pd.DataFrame
) -> pd.DataFrame:
  """Calculates how many notes each rater reviewed after the note was assigned a status.

  Args:
    ratings: DataFrame containing all ratings from all users
    noteStatusHistory: DataFrame containing times when each note was first assigned CRH/CRNH status

  Returns:
    DataFrame mapping raterParticipantId to number of notes rated after status
  """
  ratingInfos = ratings[[c.noteIdKey, c.raterParticipantIdKey, c.createdAtMillisKey]].merge(
    noteStatusHistory[[c.noteIdKey, c.timestampMillisOfNoteMostRecentNonNMRLabelKey]],
    how="inner",
  )
  assert (
    len(ratingInfos) == len(ratings)
  ), f"assigning a status timestamp shouldn't decrease number of ratings: {len(ratingInfos)} vs. {len(ratings)}"
  logger.info("Calculating ratedAfterDecision:")
  logger.info(f"  Total ratings: {len(ratingInfos)}")
  ratingInfos = ratingInfos[~pd.isna(ratingInfos[c.timestampMillisOfNoteMostRecentNonNMRLabelKey])]
  logger.info(f"  Total ratings on notes with status: {len(ratingInfos)}")
  ratingInfos = ratingInfos[
    ratingInfos[c.createdAtMillisKey] > ratingInfos[c.timestampMillisOfNoteMostRecentNonNMRLabelKey]
  ]
  logger.info(f"  Total ratings after status: {len(ratingInfos)}")
  ratingInfos[c.ratedAfterDecision] = 1
  ratedAfterDecision = (
    ratingInfos[[c.raterParticipantIdKey, c.ratedAfterDecision]]
    .groupby(c.raterParticipantIdKey)
    .sum()
  )
  logger.info(f"  Total raters rating after decision: {len(ratedAfterDecision)}")
  return ratedAfterDecision


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
    [c.awaitingMoreRatingsBoolKey]
  ]
  historyCounts[c.ratingsAwaitingMoreRatings] = historyCounts[c.awaitingMoreRatingsBoolKey]
  ratedAfterDecision = _get_rated_after_decision(ratings, noteStatusHistory)
  historyCounts = historyCounts.merge(
    ratedAfterDecision,
    on=c.raterParticipantIdKey,
    how="left",
    unsafeAllowed=c.ratedAfterDecision,
  )
  # Fill in zero for any rater who didn't rate any notes after status was assigned and consequently
  # doesn't appear in the dataframe.
  historyCounts = historyCounts.fillna({c.ratedAfterDecision: 0})

  ratingCounts = ratingCounts.merge(
    historyCounts, on=c.raterParticipantIdKey, how="outer", unsafeAllowed=set(ratingCountRows)
  )
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
    {
      c.needsMoreRatings: 0,
      c.currentlyRatedHelpful: 1,
      c.currentlyRatedNotHelpful: 1,
    }
  )
  return x.map(nmrSortLast)


def _get_visible_note_counts(
  scoredNotes: pd.DataFrame,
  lastNNotes: int = -1,
  countNMRNotesLast: bool = False,
  sinceLastEarnOut: bool = False,
):
  """
  Given scored notes from the algorithm, this function aggregates the note status by note author.

  Args:
      scoredNotes: Notes scored from MF + contributor stats
      lastNNotes: Only count the last n notes
      countNMRNotesLast: Count the NMR notes last. Only affects lastNNNotes counts.
      sinceLastEarnOut: Only count notes since the last time the contributor earned out
  Returns:
      pd.DataFrame: noteCounts The visible note counts
  """
  sort_by = [c.finalRatingStatusKey, c.noteIdKey] if countNMRNotesLast else c.noteIdKey
  key_function = _sort_nmr_status_last if countNMRNotesLast else None
  if not sinceLastEarnOut:
    aggNotes = scoredNotes
  else:
    aggNotes = scoredNotes.loc[
      scoredNotes[c.createdAtMillisKey] > scoredNotes[c.timestampOfLastEarnOut]
    ].copy()

  groupAuthorCounts = (
    aggNotes.sort_values(sort_by, ascending=False, key=key_function)
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
    else aggNotes.groupby(c.noteAuthorParticipantIdKey).sum(numeric_only=True)
  )
  authorCounts = pd.DataFrame(
    groupAuthorCounts[
      [
        c.currentlyRatedHelpfulBoolKey,
        c.currentlyRatedNotHelpfulBoolKey,
        c.awaitingMoreRatingsBoolKey,
        c.numRatingsKey,
      ]
    ]
  )
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


def _transform_to_thrift_code(f):
  """
  TODO: Fix MH importer or CombineEventAndSnapshot.
  This is a bit of tech debt that should be addressed at some point. The MH importer expects
  a Thrift code, and the CombineEventAndSnapshot outputs a string. This function ensures that
  all strings are correctly converted.
  """
  if f in c.enrollmentStateToThrift:
    return c.enrollmentStateToThrift[f]
  return f


def is_emerging_writer(scoredNotes: pd.DataFrame):
  """
  A function that checks if a user is an emerging writer. Emerging writers have a
  high helpfulness scores over a number of ratings in the last 28 days.
  Args:
      scoredNotes (pd.DataFrame): scored notes
  Returns:
    pd.DataFrame: emergingWriter The contributor scores with enrollments
  """
  authorCounts = author_helpfulness(scoredNotes, c.coreNoteInterceptKey)
  raterCounts = scoredNotes.groupby(c.noteAuthorParticipantIdKey).sum(numeric_only=True)[
    [c.numRatingsLast28DaysKey]
  ]
  emergingWriter = (
    authorCounts.join(raterCounts, how="outer", lsuffix="_author", rsuffix="_rater")
    .reset_index()
    .rename({"index": c.noteAuthorParticipantIdKey}, axis=1)
  )
  emergingWriter[c.isEmergingWriterKey] = False
  emergingWriter.loc[
    (emergingWriter[c.meanNoteScoreKey] > c.emergingMeanNoteScore)
    & (emergingWriter[c.numRatingsLast28DaysKey] >= c.emergingRatingCount),
    c.isEmergingWriterKey,
  ] = True
  return emergingWriter[[c.noteAuthorParticipantIdKey, c.isEmergingWriterKey]]


def single_trigger_earn_out(contributorScoresWithEnrollment: pd.DataFrame) -> pd.DataFrame:
  """
  A function that earns out users with a negative writing impact upon any CRNH note
  Args:
      contributorScoresWithEnrollment (pd.DataFrame): contributor scores with state and current enrollment
  Returns:
    pd.DataFrame: updated contributor scores reflecting single trigger earned out users
  """
  earnedOutUsers = (
    (
      contributorScoresWithEnrollment[c.notesCurrentlyRatedNotHelpful].fillna(0, inplace=False)
      > contributorScoresWithEnrollment[c.notesCurrentlyRatedHelpful].fillna(0, inplace=False)
    )
    & (contributorScoresWithEnrollment[c.hasCrnhSinceEarnOut] == True)
    & (
      contributorScoresWithEnrollment[c.enrollmentState]
      != c.enrollmentStateToThrift[c.earnedOutNoAcknowledge]
    )
    & (
      contributorScoresWithEnrollment[c.enrollmentState]
      != c.enrollmentStateToThrift[c.earnedOutAcknowledged]
    )
    & (contributorScoresWithEnrollment[c.enrollmentState] != c.enrollmentStateToThrift[c.newUser])
    & (contributorScoresWithEnrollment[c.enrollmentState] != c.enrollmentStateToThrift[c.removed])
  )

  contributorScoresWithEnrollment.loc[earnedOutUsers, c.numberOfTimesEarnedOutKey] = (
    contributorScoresWithEnrollment.loc[earnedOutUsers, c.numberOfTimesEarnedOutKey] + 1
  )

  # use earned out no ack internally to identify newly earned out users
  contributorScoresWithEnrollment.loc[
    earnedOutUsers, c.enrollmentState
  ] = c.enrollmentStateToThrift[c.earnedOutNoAcknowledge]
  contributorScoresWithEnrollment.loc[earnedOutUsers, c.timestampOfLastStateChange] = c.epochMillis

  return contributorScoresWithEnrollment.drop(columns=[c.hasCrnhSinceEarnOut])


def calculate_ri_to_earn_in(contributorScoresWithEnrollment: pd.DataFrame) -> pd.DataFrame:
  """
  A function that updates rating impact needed to earn in for earned out users
  Args:
      contributorScoresWithEnrollment (pd.DataFrame): contributor scores with state and current enrollment
  Returns:
    pd.DataFrame: dataframe with updated rating impact required to earn in for earned out users
  """
  earnedOutUsers = (
    contributorScoresWithEnrollment[c.enrollmentState]
    == c.enrollmentStateToThrift[c.earnedOutNoAcknowledge]
  )

  contributorScoresWithEnrollment.loc[
    earnedOutUsers, c.successfulRatingNeededToEarnIn
  ] = contributorScoresWithEnrollment.apply(
    lambda row: c.ratingImpactForEarnIn
    + max([row[c.ratingImpact], 0])
    + (c.ratingImpactForEarnIn * max(row[c.numberOfTimesEarnedOutKey] - 1, 0)),
    axis=1,
  ).loc[earnedOutUsers]

  # for top writers, overwrite the score required to earn in with non-escalating version
  topWriters = is_top_writer(contributorScoresWithEnrollment)

  contributorScoresWithEnrollment.loc[
    (earnedOutUsers) & (topWriters), c.successfulRatingNeededToEarnIn
  ] = contributorScoresWithEnrollment.apply(
    lambda row: c.ratingImpactForEarnIn + max([row[c.ratingImpact], 0]),
    axis=1,
  ).loc[(earnedOutUsers) & (topWriters)]

  contributorScoresWithEnrollment.loc[
    earnedOutUsers, c.enrollmentState
  ] = c.enrollmentStateToThrift[c.earnedOutAcknowledged]

  return contributorScoresWithEnrollment.drop(columns=[c.ratingImpact])


def get_contributor_state(
  scoredNotes: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  userEnrollment: pd.DataFrame,
  log: bool = True,
) -> pd.DataFrame:
  """
  Given scored notes, ratings, note status history, the current user enrollment state, this
  uses the contributor counts over ratings and notes and transitions the user between the
  different enrollment states. If current user enrollment state is removed, do not change.

  Args:
      scoredNotes (pd.DataFrame): scored notes
      ratings (pd.DataFrame): all ratings
      noteStatusHistory (pd.DataFrame): history of note statuses
      userEnrollment (pd.DataFrame): User enrollment for BW participants.
      log (bool): Should we log
  Returns:
      pd.DataFrame: contributorScoresWithEnrollment The contributor scores with enrollments
  """
  with c.time_block("Contributor State: Setup"):
    # for users in state Earned Out Ack, update the timestamp of last earn out; this ensures they are only judged against
    # their rating target until they resume writing notes
    userEnrollment.loc[
      userEnrollment[c.enrollmentState] == c.earnedOutAcknowledged, c.timestampOfLastEarnOut
    ] = c.epochMillis

    # We need to consider only the last 5 notes for enrollment state. The ratings are aggregated historically.
    # For users who have earned out, we should only consider notes written since the earn out event
    scoredNotesWithLastEarnOut = scoredNotes.merge(
      userEnrollment[[c.participantIdKey, c.timestampOfLastEarnOut]],
      left_on=c.noteAuthorParticipantIdKey,
      right_on=c.participantIdKey,
      how="left",
      unsafeAllowed=c.timestampOfLastEarnOut,
    )
    # For users who don't appear in the userEnrollment file, set their timeStampOfLastEarnOut to default
    scoredNotesWithLastEarnOut[c.timestampOfLastEarnOut].fillna(1, inplace=True)

  with c.time_block("Contributor State: Contributor Scores"):
    contributorScores = get_contributor_scores(
      scoredNotesWithLastEarnOut,
      ratings,
      noteStatusHistory,
      lastNNotes=c.maxHistoryEarnOut,
      countNMRNotesLast=True,
      sinceLastEarnOut=True,
    )
    contributorScores.fillna(0, inplace=True)

  contributorScores[c.hasCrnhSinceEarnOut] = contributorScores[c.notesCurrentlyRatedNotHelpful] > 0

  with c.time_block("Contributor State: Top NH Tags Per Author"):
    # We merge in the top not helpful tags
    authorTopNotHelpfulTags = explanation_tags.get_top_nonhelpful_tags_per_author(
      noteStatusHistory, ratings
    )
    contributorScores = contributorScores.merge(
      authorTopNotHelpfulTags,
      left_on=c.raterParticipantIdKey,
      right_on=c.noteAuthorParticipantIdKey,
      how="outer",
      unsafeAllowed=c.hasCrnhSinceEarnOut,
    ).drop(columns=[c.noteAuthorParticipantIdKey])

  with c.time_block("Contributor State: Emerging Writers"):
    # We merge in the emerging writer data.
    emergingWriter = is_emerging_writer(scoredNotes)
    contributorScores = contributorScores.merge(
      emergingWriter,
      left_on=c.raterParticipantIdKey,
      right_on=c.noteAuthorParticipantIdKey,
      how="outer",
      unsafeAllowed=c.isEmergingWriterKey,
    ).drop(columns=[c.noteAuthorParticipantIdKey])

  with c.time_block("Contributor State: Combining"):
    # We merge the current enrollment state
    contributorScoresWithEnrollment = contributorScores.merge(
      userEnrollment,
      left_on=c.raterParticipantIdKey,
      right_on=c.participantIdKey,
      how="outer",
      unsafeAllowed={
        c.successfulRatingNeededToEarnIn,
        c.timestampOfLastStateChange,
        c.numberOfTimesEarnedOutKey,
        "coreBool",
        "expansionBool",
      },
    )

    # We set the new contributor state.
    contributorScoresWithEnrollment.fillna(
      inplace=True,
      value={
        c.successfulRatingNeededToEarnIn: c.ratingImpactForEarnIn,
        c.enrollmentState: c.newUser,
        c.isEmergingWriterKey: False,
      },
    )
    contributorScoresWithEnrollment[c.ratingImpact] = (
      contributorScoresWithEnrollment[c.successfulRatingTotal]
      - contributorScoresWithEnrollment[c.unsuccessfulRatingTotal]
      # 2x penalty for helpful ratings on CRNH notes
      - contributorScoresWithEnrollment[c.unsuccessfulRatingNotHelpfulCount]
    )

    contributorScoresWithEnrollment.loc[
      should_earn_in(contributorScoresWithEnrollment), c.enrollmentState
    ] = c.enrollmentStateToThrift[c.earnedIn]
    contributorScoresWithEnrollment.loc[
      should_earn_in(contributorScoresWithEnrollment), c.timestampOfLastStateChange
    ] = c.epochMillis

    contributorScoresWithEnrollment.loc[
      newly_at_risk(contributorScoresWithEnrollment), c.enrollmentState
    ] = c.enrollmentStateToThrift[c.atRisk]
    contributorScoresWithEnrollment.loc[
      newly_at_risk(contributorScoresWithEnrollment), c.timestampOfLastStateChange
    ] = c.epochMillis

    # for earned out users, first increment the number of times they have earned out,
    # use this to overwrite successful rating needed to earn in,
    # then set new state
    earnedOutUsers = is_earned_out(contributorScoresWithEnrollment)
    contributorScoresWithEnrollment.loc[earnedOutUsers, c.numberOfTimesEarnedOutKey] = (
      contributorScoresWithEnrollment.loc[earnedOutUsers, c.numberOfTimesEarnedOutKey] + 1
    )

    # use earned out no ack internally to identify newly earned out users
    contributorScoresWithEnrollment.loc[
      earnedOutUsers, c.enrollmentState
    ] = c.enrollmentStateToThrift[c.earnedOutNoAcknowledge]

    contributorScoresWithEnrollment.loc[
      earnedOutUsers, c.timestampOfLastStateChange
    ] = c.epochMillis

    # at risk users transitioning back to earned in
    contributorScoresWithEnrollment.loc[
      newly_earned_in(contributorScoresWithEnrollment), c.enrollmentState
    ] = c.enrollmentStateToThrift[c.earnedIn]
    contributorScoresWithEnrollment.loc[
      newly_earned_in(contributorScoresWithEnrollment), c.timestampOfLastStateChange
    ] = c.epochMillis

    contributorScoresWithEnrollment[c.enrollmentState] = contributorScoresWithEnrollment[
      c.enrollmentState
    ].map(_transform_to_thrift_code)

    mappedUserEnrollment = userEnrollment[
      [c.participantIdKey, c.timestampOfLastEarnOut, c.enrollmentState]
    ]
    mappedUserEnrollment[c.enrollmentState] = mappedUserEnrollment[c.enrollmentState].map(
      _transform_to_thrift_code
    )
    mappedUserEnrollment = mappedUserEnrollment.rename(
      columns={c.enrollmentState: c.enrollmentState + "_prev"}
    )

    # This addresses an issue in the TSV dump in HDFS getting corrupted. It removes lines
    # users that do not have an id.
    contributorScoresWithEnrollment.dropna(subset=[c.raterParticipantIdKey], inplace=True)

  if log:
    logger.info("Enrollment State")
    logger.info(
      f"Number of Earned In {len(contributorScoresWithEnrollment[contributorScoresWithEnrollment[c.enrollmentState] == 0])}"
    )
    logger.info(
      f"Number At Risk {len(contributorScoresWithEnrollment[contributorScoresWithEnrollment[c.enrollmentState] == 1])}"
    )
    logger.info(
      f"Number of Earn Out No Ack {len(contributorScoresWithEnrollment[contributorScoresWithEnrollment[c.enrollmentState] == 2])}"
    )
    logger.info(
      f"Number of Earned Out Ack {len(contributorScoresWithEnrollment[contributorScoresWithEnrollment[c.enrollmentState] == 3])}"
    )
    logger.info(
      f"Number of New Users {len(contributorScoresWithEnrollment[contributorScoresWithEnrollment[c.enrollmentState] == 4])}"
    )

  return contributorScoresWithEnrollment, mappedUserEnrollment


def get_contributor_scores(
  scoredNotes: pd.DataFrame,
  ratings: pd.DataFrame,
  statusHistory: pd.DataFrame,
  lastNNotes=-1,
  countNMRNotesLast: bool = False,
  sinceLastEarnOut: bool = False,
  log: bool = True,
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
      sinceLastEarnOut: only count notes since last Earn Out event
      log (bool): Should we log?
  Returns:
      pd.DataFrame: contributorScores - rating + note aggregates per contributor.
  """
  visibleRatingCounts = _get_visible_rating_counts(scoredNotes, ratings, statusHistory)
  visibleNoteCounts = _get_visible_note_counts(
    scoredNotes, lastNNotes, countNMRNotesLast, sinceLastEarnOut
  )
  contributorCounts = (
    visibleRatingCounts.join(
      visibleNoteCounts,
      lsuffix="note",
      rsuffix="rater",
      how="outer",
      unsafeAllowed={
        c.defaultIndexKey,
        c.awaitingMoreRatingsBoolKey + "note",
        c.ratingsAwaitingMoreRatings,
        c.currentlyRatedHelpfulBoolKey,
        c.currentlyRatedNotHelpfulBoolKey,
        c.awaitingMoreRatingsBoolKey + "rater",
        c.notesCurrentlyRatedHelpful,
        c.notesCurrentlyRatedNotHelpful,
        c.notesAwaitingMoreRatings,
        c.numRatingsKey,
        c.aggregateRatingReceivedTotal,
      },
    )
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

  if log:
    logger.info(f"Number Contributor Counts: {len(contributorCounts)}")

  return contributorCounts
