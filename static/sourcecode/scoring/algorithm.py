from typing import Optional, Tuple

from . import (
  constants as c,
  contributor_state,
  helpfulness_scores,
  matrix_factorization,
  note_ratings,
  note_status_history,
  process_data,
)

import numpy as np
import pandas as pd
import torch


def note_post_processing(
  ratings: pd.DataFrame,
  noteParams: pd.DataFrame,
  raterParams: pd.DataFrame,
  helpfulnessScores: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  userEnrollment: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Given scored Birdwatch notes and rater helpfulness, calculate contributor scores and update noteStatusHistory, as described in
  and https://twitter.github.io/communitynotes/contributor-scores/.

  Args:
      ratings (pd.DataFrame): preprocessed ratings
      noteParams (pd.DataFrame): notes with scores returned by MF scoring algorithm
      raterParams (pd.DataFrame): raters with scores returned by MF scoring algorithm
      helpfulnessScores (pd.DataFrame): BasicReputation scores for all raters
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
      userEnrollment (pd.DataFrame): The enrollment state for each contributor

  Returns:
      Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        scoredNotes pd.DataFrame: one row per note contained note scores and parameters.
        helpfulnessScores pd.DataFrame: one row per user containing a column for each helpfulness score.
        noteStatusHistory pd.DataFrame: one row per note containing when they got their most recent statuses.
        auxilaryNoteInfo pd.DataFrame: one row per note containing adjusted and ratio tag values
  """
  helpfulnessScores = raterParams.merge(
    helpfulnessScores[
      [
        c.raterParticipantIdKey,
        c.crhCrnhRatioDifferenceKey,
        c.meanNoteScoreKey,
        c.raterAgreeRatioKey,
        c.aboveHelpfulnessThresholdKey,
      ]
    ],
    on=c.raterParticipantIdKey,
    how="outer",
  )

  scoredNotes = note_ratings.compute_scored_notes(
    ratings, noteParams, raterParams, noteStatusHistory, finalRound=True
  )
  contributorScores = contributor_state.get_contributor_scores(
    scoredNotes, ratings, noteStatusHistory
  )
  contributorState = contributor_state.get_contributor_state(
    scoredNotes, ratings, noteStatusHistory, userEnrollment
  )

  contributorScores = contributorScores.merge(
    contributorState[
      [
        c.raterParticipantIdKey,
        c.timestampOfLastStateChange,
        c.enrollmentState,
        c.successfulRatingNeededToEarnIn,
        c.authorTopNotHelpfulTagValues,
        c.isEmergingWriterKey,
      ]
    ],
    on=c.raterParticipantIdKey,
    how="outer",
  )

  helpfulnessScores = helpfulnessScores.merge(
    contributorScores,
    on=c.raterParticipantIdKey,
    how="outer",
  )

  helpfulnessScores = helpfulnessScores.merge(
    userEnrollment[[c.participantIdKey, c.timestampOfLastEarnOut]],
    left_on=c.raterParticipantIdKey,
    right_on=c.participantIdKey,
    how="left",
  ).drop(c.participantIdKey, axis=1)
  helpfulnessScores[c.timestampOfLastEarnOut].fillna(1, inplace=True)

  assert ratings.columns.tolist() == c.ratingTSVColumns + [c.helpfulNumKey]

  scoredNotes = scoredNotes.merge(noteParams[[c.noteIdKey]], on=c.noteIdKey, how="inner")
  castColumns = c.helpfulTagsTSVOrder + c.notHelpfulTagsTSVOrder + [c.numRatingsKey]
  scoredNotes[castColumns] = scoredNotes[castColumns].astype(np.int64)

  newNoteStatusHistory = note_status_history.update_note_status_history(
    noteStatusHistory, scoredNotes
  )

  assert set(scoredNotes.columns) == set(
    c.noteModelOutputTSVColumns + c.auxilaryScoredNotesTSVColumns
  )
  auxilaryNoteInfo = scoredNotes[c.auxilaryScoredNotesTSVColumns]
  scoredNotes = scoredNotes[c.noteModelOutputTSVColumns]
  helpfulnessScores = helpfulnessScores[c.raterModelOutputTSVColumns]
  newNoteStatusHistory = newNoteStatusHistory[c.noteStatusHistoryTSVColumns]

  return scoredNotes, helpfulnessScores, newNoteStatusHistory, auxilaryNoteInfo


def run_algorithm(
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  userEnrollment: pd.DataFrame,
  epochs: int = c.epochs,
  seed: Optional[int] = None,
  pseudoraters: Optional[bool] = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Run the entire Birdwatch scoring algorithm, as described in https://twitter.github.io/communitynotes/ranking-notes/
  and https://twitter.github.io/communitynotes/contributor-scores/.

  Args:
      ratings (pd.DataFrame): preprocessed ratings
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
      userEnrollment (pd.DataFrame): The enrollment state for each contributor
      epochs (int, optional): number of epochs to train matrix factorization for. Defaults to c.epochs.
      mf_seed (int, optional): if not None, base distinct seeds for the first and second MF rounds on this value
      pseudoraters (bool, optional): if True, compute optional pseudorater confidence intervals

  Returns:
      Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        scoredNotes pd.DataFrame: one row per note contained note scores and parameters.
        helpfulnessScores pd.DataFrame: one row per user containing a column for each helpfulness score.
        noteStatusHistory pd.DataFrame: one row per note containing when they got their most recent statuses.
        auxilaryNoteInfo: one row per note containing adjusted and ratio tag values
  """
  if seed is not None:
    torch.manual_seed(seed)

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

  scoredNotes = note_ratings.compute_scored_notes(
    ratings, noteParamsUnfiltered, raterParamsUnfiltered, noteStatusHistory
  )

  validRatings = note_ratings.get_valid_ratings(ratings, noteStatusHistory, scoredNotes)

  helpfulnessScores = helpfulness_scores.compute_general_helpfulness_scores(
    scoredNotes, validRatings
  )

  """
  TODO: try doing this to re-apply 5-rating minimum
  ratingsHelpfulnessScoreFiltered = helpfulness_scores.filter_ratings_by_helpfulness_scores(
    ratingsForTraining, helpfulnessScores
  )

  ratingsHelpfulnessScoreAndMinNumberOfRatingsFiltered = process_data.filter_ratings(ratingsHelpfulnessScoreFiltered)

  noteParams, raterParams, globalBias = matrix_factorization.run_mf(
    ratingsHelpfulnessScoreAndMinNumberOfRatingsFiltered,
    c.l2_lambda,
    c.l2_intercept_multiplier,
    c.numFactors,
    epochs,
    c.useGlobalIntercept,
  )
  """

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
    noteInit=noteParamsUnfiltered,
    userInit=raterParamsUnfiltered,
  )
  raterParams.drop(c.raterIndexKey, axis=1, inplace=True)

  if pseudoraters:
    noteIdMap, raterIdMap, noteRatingIds = matrix_factorization.get_note_and_rater_id_maps(
      ratingsHelpfulnessScoreFiltered
    )

    extremeRaters = matrix_factorization.make_extreme_raters(raterParams, raterIdMap)

    (
      rawRescoredNotesWithEachExtraRater,
      notesWithConfidenceBounds,
    ) = matrix_factorization.fit_note_params_for_each_dataset_with_extreme_ratings(
      extremeRaters,
      noteRatingIds,
      ratingsHelpfulnessScoreFiltered,
      noteParams,
      raterParams,
      globalBias,
    )

    noteParams = noteParams.merge(notesWithConfidenceBounds.reset_index(), on="noteId", how="left")

  else:
    for col in c.noteParameterUncertaintyTSVColumns:
      noteParams[col] = np.nan

  return note_post_processing(
    ratings, noteParams, raterParams, helpfulnessScores, noteStatusHistory, userEnrollment
  )
