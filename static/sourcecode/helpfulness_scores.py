import numpy as np
import pandas as pd
from constants import *

def compute_general_helpfulness_scores(
  noteParams,
  notes,
  ratings,
  thresholdCRH,
  thresholdCRNH,
  minRatingsNeeded,
  CRNHMultiplier=5.0,
  logging=True,
):
  scoredNotes = (
    notes[[noteIdKey, participantIdKey, classificationKey]]
    .drop_duplicates()
    .rename({participantIdKey: noteAuthorParticipantIdKey}, axis=1)
    .merge(noteParams, on=noteIdKey)
  )

  scoredNotes[currentlyRatedHelpfulBoolKey] = False
  scoredNotes[currentlyRatedNotHelpfulBoolKey] = False
  scoredNotes.loc[(scoredNotes[noteInterceptKey] >= thresholdCRH), currentlyRatedHelpfulBoolKey] = True
  scoredNotes.loc[(scoredNotes[noteInterceptKey] <= thresholdCRNH), currentlyRatedNotHelpfulBoolKey] = True
  scoredNotes[noteCountKey] = 1

  if logging:
    print(
      f"With threshold {thresholdCRH}, {scoredNotes[currentlyRatedHelpfulBoolKey].sum()} notes are CRH " +
       f"({100*scoredNotes[currentlyRatedHelpfulBoolKey].sum()/len(scoredNotes)}%)"
    )
    print(
      f"With threshold {thresholdCRNH}, {scoredNotes[currentlyRatedNotHelpfulBoolKey].sum()} notes are CRNH " +
       f"({100*scoredNotes[currentlyRatedNotHelpfulBoolKey].sum()/len(scoredNotes)}%)"
    )

  # Author Helpfulness
  authorCounts = scoredNotes.groupby(noteAuthorParticipantIdKey).sum()[[currentlyRatedHelpfulBoolKey, currentlyRatedNotHelpfulBoolKey, noteCountKey, noteInterceptKey]]
  authorCounts[crhRatioKey] = authorCounts[currentlyRatedHelpfulBoolKey] / authorCounts[noteCountKey]
  authorCounts[crnhRatioKey] = authorCounts[currentlyRatedNotHelpfulBoolKey] / authorCounts[noteCountKey]
  authorCounts[crhCrnhRatioDifferenceKey] = authorCounts[crhRatioKey] - (authorCounts[crnhRatioKey] * CRNHMultiplier)
  authorCounts[meanNoteScoreKey] = authorCounts[noteInterceptKey] / authorCounts[noteCountKey]

  # Rater Helpfulness
  validRatings = (
    ratings[[raterParticipantIdKey, noteIdKey, helpfulNumKey, createdAtMillisKey]]
    .sort_values(createdAtMillisKey)
    .groupby(noteIdKey)
    .head(minRatingsNeeded)
  )

  ratingsWithScores = validRatings[[raterParticipantIdKey, helpfulNumKey, noteIdKey]].merge(
    scoredNotes[[noteIdKey, noteInterceptKey, currentlyRatedHelpfulBoolKey, currentlyRatedNotHelpfulBoolKey]], on=noteIdKey
  )
  ratingsWithScores[ratingCountKey] = 1

  binaryRatingsOnNotesWithStatusLabels = ratingsWithScores[
    (ratingsWithScores[currentlyRatedHelpfulBoolKey] | ratingsWithScores[currentlyRatedNotHelpfulBoolKey])
    & ((ratingsWithScores[helpfulNumKey] == 1) | (ratingsWithScores[helpfulNumKey] == 0))
  ].copy()

  helpfulRatingOnCrhNote = binaryRatingsOnNotesWithStatusLabels[currentlyRatedHelpfulBoolKey] & binaryRatingsOnNotesWithStatusLabels[helpfulNumKey] == 1
  notHelpfulRatingOnCrnhNote = (binaryRatingsOnNotesWithStatusLabels[currentlyRatedNotHelpfulBoolKey] & binaryRatingsOnNotesWithStatusLabels[helpfulNumKey] == 0)
  binaryRatingsOnNotesWithStatusLabels[ratingAgreesWithNoteStatusKey] = helpfulRatingOnCrhNote | notHelpfulRatingOnCrnhNote
  raterCounts = binaryRatingsOnNotesWithStatusLabels.groupby(raterParticipantIdKey).sum()[
    [ratingAgreesWithNoteStatusKey, ratingCountKey]
  ]
  raterCounts[raterAgreeRatioKey] = raterCounts[ratingAgreesWithNoteStatusKey] / raterCounts[ratingCountKey]

  helpfulnessScores = (
    authorCounts.join(raterCounts, how="outer", lsuffix="_author", rsuffix="_rater")
    .reset_index()
    .rename({"index": raterParticipantIdKey}, axis=1)[
      [raterParticipantIdKey, crhCrnhRatioDifferenceKey, meanNoteScoreKey, raterAgreeRatioKey]
    ]
  )

  return helpfulnessScores


def filter_ratings_by_helpfulness_scores(
  ratingsForTraining,
  helpfulnessScores,
  minMeanNoteScore,
  minCRHVsCRNHRatio,
  minRaterAgreeRatio,
  logging=True,
):
  includedUsers = helpfulnessScores[
    (
      (
        (helpfulnessScores[crhCrnhRatioDifferenceKey] >= minCRHVsCRNHRatio)
        & (helpfulnessScores[meanNoteScoreKey] >= minMeanNoteScore)
      )
      | (pd.isna(helpfulnessScores[crhCrnhRatioDifferenceKey]) & pd.isna(helpfulnessScores[meanNoteScoreKey]))
      | (
        pd.isna(helpfulnessScores[crhCrnhRatioDifferenceKey]) & helpfulnessScores[meanNoteScoreKey]
        >= minMeanNoteScore
      )
    )
    & (helpfulnessScores[raterAgreeRatioKey] >= minRaterAgreeRatio)
  ][[raterParticipantIdKey]]

  ratingsHelpfulnessScoreFiltered = includedUsers.merge(ratingsForTraining, on=raterParticipantIdKey)

  if logging:
    print("Unique Raters: ", len(np.unique(ratingsForTraining[raterParticipantIdKey])))
    print("Raters With Helpfulness Scores: ", len(helpfulnessScores))
    print("Raters Included Based on Helpfulness Scores: ", len(includedUsers))
    print("Number of Ratings Used For 1st Training: ", len(ratingsForTraining))
    print("Number of Ratings for Final Training: ", len(ratingsHelpfulnessScoreFiltered))

  return ratingsHelpfulnessScoreFiltered
