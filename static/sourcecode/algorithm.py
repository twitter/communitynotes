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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Given scored Birdwatch notes and rater helpfulness, calculate contributor scores and update noteStatusHistory, as described in
  and https://twitter.github.io/birdwatch/contributor-scores/.

  Args:
      ratings (pd.DataFrame): preprocessed ratings
      noteParams (pd.DataFrame): notes with scores returned by MF scoring algorithm
      raterParams (pd.DataFrame): raters with scores returned by MF scoring algorithm
      helpfulnessScores (pd.DataFrame): BasicReputation scores for all raters
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status

  Returns:
      Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        scoredNotes pd.DataFrame: one row per note contained note scores and parameters.
        helpfulnessScores pd.DataFrame: one row per user containing a column for each helpfulness score.
        noteStatusHistory pd.DataFrame: one row per note containing when they got their most recent statuses.
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
  contributorNotes = note_ratings.compute_scored_notes(
    ratings, noteParams, noteStatusHistory, allNotes=True
  )
  # Return one row per rater with stats including trackrecord identifying note labels.
  contributorScores = contributor_state.get_contributor_scores(
    contributorNotes, ratings, noteStatusHistory
  )

  # Consolidates all information on raters / authors.
  helpfulnessScores = helpfulnessScores.merge(
    contributorScores,
    on=c.raterParticipantIdKey,
    how="outer",
  )

  print("computing scoredNotes from contributorNotes")
  scoredNotes = contributorNotes[c.scoredNotesColumns].merge(
    noteParams[[c.noteIdKey]], on=c.noteIdKey, how="inner"
  )
  castColumns = c.helpfulTagsTSVOrder + c.notHelpfulTagsTSVOrder + [c.numRatingsKey]
  scoredNotes[castColumns] = scoredNotes[castColumns].astype(np.int64)

  scoredNotes = scoredNotes.merge(
    noteStatusHistory[[c.noteIdKey, c.createdAtMillisKey, c.noteAuthorParticipantIdKey]],
    on=c.noteIdKey,
    how="inner",
  )

  newNoteStatusHistory = note_status_history.update_note_status_history(
    noteStatusHistory, scoredNotes
  )

  return scoredNotes, helpfulnessScores, newNoteStatusHistory


def run_algorithm(
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  epochs: int = c.epochs,
  seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Run the entire Birdwatch scoring algorithm, as described in https://twitter.github.io/birdwatch/ranking-notes/
  and https://twitter.github.io/birdwatch/contributor-scores/.

  Args:
      ratings (pd.DataFrame): preprocessed ratings
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
      epochs (int, optional): number of epochs to train matrix factorization for. Defaults to c.epochs.
      mf_seed (int, optional): if not None, base distinct seeds for the first and second MF rounds on this value

  Returns:
      Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        scoredNotes pd.DataFrame: one row per note contained note scores and parameters.
        helpfulnessScores pd.DataFrame: one row per user containing a column for each helpfulness score.
        noteStatusHistory pd.DataFrame: one row per note containing when they got their most recent statuses.
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
  scoredNotes = note_ratings.compute_scored_notes(ratings, noteParamsUnfiltered, noteStatusHistory)

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

  scoredNotes, helpfulnessScores, newNoteStatusHistory = note_post_processing(
    ratings, noteParams, raterParams, helpfulnessScores, noteStatusHistory
  )

  return scoredNotes, helpfulnessScores, newNoteStatusHistory
