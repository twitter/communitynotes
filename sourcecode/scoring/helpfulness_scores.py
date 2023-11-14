from typing import Optional

from . import constants as c

import numpy as np
import pandas as pd


def author_helpfulness(
  scoredNotes: pd.DataFrame,
  noteInterceptKey: str,
  CRNHMultiplier: float = 5.0,
) -> pd.DataFrame:
  """Computes author helpfulness scores as described in:
  https://twitter.github.io/communitynotes/contributor-scores/#author-helpfulness-scores

  Args:
    scoredNotes (pd.DataFrame): one row per note, containing preliminary note statuses
    noteInterceptKey: str dictating column containing note intercept
    CRNHMultiplier (float): how much more to penalize CRNH notes written vs. reward CRH notes

  Returns:
    pd.DataFrame: one row per author, containing columns for author helpfulness scores
  """

  scoredNotes.loc[:, c.noteCountKey] = 1
  authorCounts = scoredNotes.groupby(c.noteAuthorParticipantIdKey).sum(numeric_only=True)[
    [
      c.currentlyRatedHelpfulBoolKey,
      c.currentlyRatedNotHelpfulBoolKey,
      c.noteCountKey,
      noteInterceptKey,
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
  authorCounts[c.meanNoteScoreKey] = authorCounts[noteInterceptKey] / authorCounts[c.noteCountKey]

  return authorCounts


def _rater_helpfulness(validRatings: pd.DataFrame) -> pd.DataFrame:
  """Computes rater helpfulness scores as described in:
  https://twitter.github.io/communitynotes/contributor-scores/#rater-helpfulness-score

  Args:
      validRatings (pd.DataFrame): ratings to use

  Returns:
      pd.DataFrame: one row per rater, containing rater helpfulness scores in columns
  """

  raterCounts = validRatings.groupby(c.raterParticipantIdKey).sum()[
    [c.ratingAgreesWithNoteStatusKey, c.ratingCountKey]
  ]
  raterCounts[c.raterAgreeRatioKey] = (
    raterCounts[c.ratingAgreesWithNoteStatusKey] / raterCounts[c.ratingCountKey]
  )
  return raterCounts


def compute_general_helpfulness_scores(
  scoredNotes: pd.DataFrame,
  validRatings: pd.DataFrame,
  minMeanNoteScore: float,
  minCRHVsCRNHRatio: float,
  minRaterAgreeRatio: float,
  ratings: Optional[pd.DataFrame] = None,
  tagConsensusHarassmentAbuseNotes: Optional[pd.DataFrame] = None,
  tagConsensusHarassmentHelpfulRatingPenalty=10,
  multiplyPenaltyByHarassmentScore: bool = False,
  minimumHarassmentScoreToPenalize: float = 2.5,
) -> pd.DataFrame:
  """Given notes scored by matrix factorization, compute helpfulness scores.
  Author helpfulness scores are based on the scores of the notes you wrote.
  Rater helpfulness scores are based on how the ratings you made match up with note scores.
  See https://twitter.github.io/communitynotes/contributor-scores/.

  Args:
      scoredNotes: one row per note, containing preliminary note statuses.
      validRatings: ratings to use.
      minMeanNoteScore: minimum average intercept for notes written.
      minCRHVsCRNHRatio: threshold applied to crhCrnhRatioDifference, a weighted measure
        comparing how often an author produces CRH / CRNH notes.  See author_helpfulness.
      minRaterAgreeRatio: minimum standard for how often a rater must predict the
        eventual outcome when rating before a note is assigned status.
      ratings: all ratings (to check if tag-consensus harassment/abuse notes were rated helpful)
  Returns:
      helpfulness_scores pandas.DataFrame: 1 row per user, with helpfulness scores as columns.
  """
  # don't consider any notes which we didn't score
  scoredNotes = scoredNotes[~pd.isna(scoredNotes[c.internalNoteInterceptKey])]
  authorCounts = author_helpfulness(scoredNotes, c.internalNoteInterceptKey)
  raterCounts = _rater_helpfulness(validRatings)

  helpfulnessScores = (
    authorCounts.join(raterCounts, how="outer", lsuffix="_author", rsuffix="_rater")
    .reset_index()
    .rename({"index": c.raterParticipantIdKey}, axis=1)[
      [
        c.raterParticipantIdKey,
        c.crhCrnhRatioDifferenceKey,
        c.meanNoteScoreKey,
        c.raterAgreeRatioKey,
        c.ratingAgreesWithNoteStatusKey,
        c.ratingCountKey,
      ]
    ]
  )

  if (ratings is None) or (tagConsensusHarassmentAbuseNotes is None):
    helpfulnessScores[c.totalHelpfulHarassmentRatingsPenaltyKey] = 0
  else:
    filteredAbuseNotes = tagConsensusHarassmentAbuseNotes[
      tagConsensusHarassmentAbuseNotes[c.harassmentNoteInterceptKey]
      >= minimumHarassmentScoreToPenalize
    ]
    helpfulRatingsOnBadNotes = ratings[ratings[c.helpfulNumKey] == 1].merge(
      filteredAbuseNotes, on=c.noteIdKey
    )

    helpfulRatingsOnBadNotes[
      c.totalHelpfulHarassmentRatingsPenaltyKey
    ] = tagConsensusHarassmentHelpfulRatingPenalty
    if multiplyPenaltyByHarassmentScore:
      helpfulRatingsOnBadNotes[c.totalHelpfulHarassmentRatingsPenaltyKey] *= (
        helpfulRatingsOnBadNotes[c.harassmentNoteInterceptKey] / minimumHarassmentScoreToPenalize
      )

    helpfulRatingsOnBadNotesCount = (
      helpfulRatingsOnBadNotes.groupby(c.raterParticipantIdKey)
      .sum()[[c.totalHelpfulHarassmentRatingsPenaltyKey]]
      .reset_index()
    )
    helpfulnessScores = helpfulnessScores.merge(
      helpfulRatingsOnBadNotesCount, on=c.raterParticipantIdKey, how="left"
    )
    helpfulnessScores[c.totalHelpfulHarassmentRatingsPenaltyKey].fillna(0, inplace=True)

  helpfulnessScores[c.raterAgreeRatioWithHarassmentAbusePenaltyKey] = (
    helpfulnessScores[c.ratingAgreesWithNoteStatusKey]
    - helpfulnessScores[c.totalHelpfulHarassmentRatingsPenaltyKey]
  ) / helpfulnessScores[c.ratingCountKey]

  helpfulnessScores[c.aboveHelpfulnessThresholdKey] = (
    (
      (
        (helpfulnessScores[c.crhCrnhRatioDifferenceKey] >= minCRHVsCRNHRatio)
        & (helpfulnessScores[c.meanNoteScoreKey] >= minMeanNoteScore)
      )
      | (
        pd.isna(helpfulnessScores[c.crhCrnhRatioDifferenceKey])
        & pd.isna(helpfulnessScores[c.meanNoteScoreKey])
      )
      | (
        pd.isna(helpfulnessScores[c.crhCrnhRatioDifferenceKey])
        & helpfulnessScores[c.meanNoteScoreKey]
        >= minMeanNoteScore
      )
    )
    & (helpfulnessScores[c.raterAgreeRatioKey] >= minRaterAgreeRatio)
    & (helpfulnessScores[c.raterAgreeRatioWithHarassmentAbusePenaltyKey] >= minRaterAgreeRatio)
  )

  helpfulnessScores.drop(columns=[c.ratingCountKey, c.ratingAgreesWithNoteStatusKey], inplace=True)
  return helpfulnessScores


def filter_ratings_by_helpfulness_scores(
  ratingsForTraining: pd.DataFrame,
  helpfulnessScores: pd.DataFrame,
  logging: bool = True,
):
  """Filter out ratings from raters whose helpfulness scores are too low.
  See https://twitter.github.io/communitynotes/contributor-scores/#filtering-ratings-based-on-helpfulness-scores.

  Args:
      ratingsForTraining pandas.DataFrame: unfiltered input ratings
      helpfulnessScores pandas.DataFrame: helpfulness scores to use to determine which raters to filter out.
      logging (bool, optional): debug output. Defaults to True.

  Returns:
      filtered_ratings pandas.DataFrame: same schema as input ratings, but filtered.
  """
  includedUsers = helpfulnessScores.loc[
    helpfulnessScores[c.aboveHelpfulnessThresholdKey], [c.raterParticipantIdKey]
  ]
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
