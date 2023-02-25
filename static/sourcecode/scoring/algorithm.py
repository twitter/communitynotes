from typing import Optional, Tuple

import constants as c, contributor_state, helpfulness_scores, matrix_factorization, note_ratings, note_status_history, process_data

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
  # Takes raterParams from most recent MF run, but use the pre-computed
  # helpfulness scores.
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

  # Assigns updated CRH / CRNH bits to notes based on volume of prior ratings
  # and ML output.
  scoredNotes = note_ratings.compute_scored_notes(
    ratings, noteParams, raterParams, noteStatusHistory, finalRound=True
  )
  # Return one row per rater with stats including trackrecord identifying note labels.
  contributorScores = contributor_state.get_contributor_scores(
    scoredNotes, ratings, noteStatusHistory
  )
  contributorState = contributor_state.get_contributor_state(
    scoredNotes, ratings, noteStatusHistory, userEnrollment
  )

  # We need to do an outer merge because the contributor can have a state (be a new user)
  # without any notes or ratings.
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

  # Consolidates all information on raters / authors.
  helpfulnessScores = helpfulnessScores.merge(
    contributorScores,
    on=c.raterParticipantIdKey,
    how="outer",
  )

  # Pass timestampOfLastEarnOut through to raterModelOutput
  helpfulnessScores = helpfulnessScores.merge(
    userEnrollment[[c.participantIdKey, c.timestampOfLastEarnOut]],
    left_on=c.raterParticipantIdKey,
    right_on=c.participantIdKey,
    how="left",
  ).drop(c.participantIdKey, axis=1)
  # If field is not set by userEvent or by update script, ok to default to 1
  helpfulnessScores[c.timestampOfLastEarnOut].fillna(1, inplace=True)

  # Computes business results of scoring and update status history.
  assert ratings.columns.tolist() == c.ratingTSVColumns + [c.helpfulNumKey]

  # Prune notes which weren't in second MF round and merge NSH to generate final scoredNotes.
  scoredNotes = scoredNotes.merge(noteParams[[c.noteIdKey]], on=c.noteIdKey, how="inner")
  castColumns = c.helpfulTagsTSVOrder + c.notHelpfulTagsTSVOrder + [c.numRatingsKey]
  scoredNotes[castColumns] = scoredNotes[castColumns].astype(np.int64)

  # Merge scoring results into noteStatusHistory
  newNoteStatusHistory = note_status_history.update_note_status_history(
    noteStatusHistory, scoredNotes
  )

  # Finalize output dataframes with correct columns
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

  # Removes ratings where either (1) the note did not receive enough ratings, or
  # (2) the rater did not rate enough notes.
  ratingsForTraining = process_data.filter_ratings(ratings)

  # TODO: Save parameters from this first run in note_model_output next time we add extra fields to model output TSV.
  noteParamsUnfiltered, raterParamsUnfiltered, globalBias = matrix_factorization.run_mf(
    ratingsForTraining,
    c.l2_lambda,
    c.l2_intercept_multiplier,
    c.numFactors,
    epochs,
    c.useGlobalIntercept,
    runName="unfiltered",
  )

  # Get a dataframe of scored notes based on the algorithm results above
  scoredNotes = note_ratings.compute_scored_notes(
    ratings, noteParamsUnfiltered, raterParamsUnfiltered, noteStatusHistory
  )

  # Determine "valid" ratings
  validRatings = note_ratings.get_valid_ratings(ratings, noteStatusHistory, scoredNotes)

  # Assigns contributor (author & rater) helpfulness bit based on (1) performance
  # authoring and reviewing previous and current notes.
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

  # Filters ratings matrix to include only rows (ratings) where the rater was
  # considered helpful.
  ratingsHelpfulnessScoreFiltered = helpfulness_scores.filter_ratings_by_helpfulness_scores(
    ratingsForTraining, helpfulnessScores
  )

  # Re-runs matrix factorization using only ratings given by helpful raters.
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

  # Add pseudo-raters with the most extreme parameters and re-score notes, to estimate
  #  upper and lower confidence bounds on note parameters.
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
