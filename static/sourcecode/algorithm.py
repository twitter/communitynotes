from typing import Tuple

import constants as c

import explanation_tags
import helpfulness_scores
import matrix_factorization
import note_status_history
import pandas as pd
import process_data


def run_algorithm(
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  epochs: int = c.epochs,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Run the entire Birdwatch scoring algorithm, as described in https://twitter.github.io/birdwatch/ranking-notes/
  and https://twitter.github.io/birdwatch/contributor-scores/.

  Args:
      ratings (pd.DataFrame): preprocessed ratings
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
      epochs (int, optional): number of epochs to train matrix factorization for. Defaults to c.epochs.

  Returns:
      Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        scoredNotes pd.DataFrame: one row per note contained note scores and parameters.
        helpfulnessScores pd.DataFrame: one row per user containing a column for each helpfulness score.
        noteStatusHistory pd.DataFrame: one row per note containing when they got their most recent statuses.
  """
  ratingsForTraining = process_data.filter_ratings(ratings)

  noteParamsUnfiltered, raterParamsUnfiltered, globalBias = matrix_factorization.run_mf(
    ratingsForTraining,
    c.l2_lambda,
    c.l2_intercept_multiplier,
    c.numFactors,
    epochs,
    c.useGlobalIntercept,
    runName="unfiltered",
  )

  helpfulnessScores = helpfulness_scores.compute_general_helpfulness_scores(
    noteParamsUnfiltered, ratings, noteStatusHistory
  )

  ratingsHelpfulnessScoreFiltered = helpfulness_scores.filter_ratings_by_helpfulness_scores(
    ratingsForTraining, helpfulnessScores
  )

  noteParams, raterParams, globalBias = matrix_factorization.run_mf(
    ratingsHelpfulnessScoreFiltered,
    c.l2_lambda,
    c.l2_intercept_multiplier,
    c.numFactors,
    epochs,
    c.useGlobalIntercept,
  )

  # take raterParams from most recent MF run, but use the pre-computed helpfulness scores
  helpfulnessScores = raterParams.drop(c.raterIndexKey, axis=1).merge(
    helpfulnessScores[
      [
        c.raterParticipantIdKey,
        c.crhCrnhRatioDifferenceKey,
        c.meanNoteScoreKey,
        c.raterAgreeRatioKey,
      ]
    ],
    on=c.raterParticipantIdKey,
    how="outer",
  )

  scoredNotes = explanation_tags.get_rating_status_and_explanation_tags(
    ratings,
    noteParams,
    c.minRatingsNeeded,
    c.minRatingsToGetTag,
    c.minTagsNeededForStatus,
    c.crhThreshold,
    c.crnhThresholdIntercept,
    c.crnhThresholdNoteFactorMultiplier,
  )

  scoredNotes = scoredNotes.merge(
    noteStatusHistory[[c.noteIdKey, c.createdAtMillisKey, c.noteAuthorParticipantIdKey]],
    on=c.noteIdKey,
    how="inner",
  )

  newNoteStatusHistory = note_status_history.update_note_status_history(
    noteStatusHistory, scoredNotes
  )

  return scoredNotes, helpfulnessScores, newNoteStatusHistory
