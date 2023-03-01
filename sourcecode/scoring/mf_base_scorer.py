from typing import List, Optional, Tuple

from . import constants as c, helpfulness_scores, matrix_factorization, note_ratings, process_data
from .scorer import Scorer

import numpy as np
import pandas as pd
import torch


class MFBaseScorer(Scorer):
  """Runs MatrixFactorization to determine raw note scores and ultimately note status."""

  def __init__(
    self,
    seed: Optional[int] = None,
    pseudoraters: Optional[bool] = True,
    minNumRatingsPerRater: int = 10,
    minNumRatersPerNote: int = 5,
    minRatingsNeeded: int = 5,
    minMeanNoteScore: float = 0.05,
    minCRHVsCRNHRatio: float = 0.00,
    minRaterAgreeRatio: float = 0.66,
    crhThreshold: float = 0.40,
    crnhThresholdIntercept: float = -0.05,
    crnhThresholdNoteFactorMultiplier: float = -0.8,
    crnhThresholdNMIntercept: float = -0.15,
    crhSuperThreshold: float = 0.5,
    inertiaDelta: float = 0.01,
  ):
    """Configure MatrixFactorizationScorer object.

    Args:
      seed: if not None, seed value to ensure deterministic execution
      pseudoraters: if True, compute optional pseudorater confidence intervals
      minNumRatingsPerRater: Minimum number of ratings which a rater must produce to be
        included in scoring.  Raters with fewer ratings are removed.
      minNumRatersPerNote: Minimum number of ratings which a note must have to be included
        in scoring.  Notes with fewer ratings are removed.
      minRatingsNeeded: Minimum number of ratings for a note to achieve status.
      minMeanNoteScore: Raters included in the second MF round must achieve this minimum
        average intercept for any notes written.
      minCRHVsCRNHRatio: Minimum crhCrnhRatioDifference for raters included in the second
        MF round. crhCrnhRatioDifference is a weighted measure comparing how often an author
        produces CRH / CRNH notes.  See author_helpfulness for more info.
      minRaterAgreeRatio: Raters in the second MF round must exceed this minimum standard for how
        often a rater must predict the eventual outcome when rating before a note is assigned status.
      crhThreshold: Minimum intercept for most notes to achieve CRH status.
      crnhThresholdIntercept: Minimum intercept for most notes to achieve CRNH status.
      crnhThresholdNoteFactorMultiplier: Scaling factor making controlling the relationship between
        CRNH threshold and note intercept.  Note that this constant is set negative so that notes with
        larger (magnitude) factors must have proportionally lower intercepts to become CRNH.
      crnhThresholdNMIntercept: Minimum intercept for notes which do not claim a tweet is misleading
        to achieve CRNH status.
      crhSuperThreshold: Minimum intercept for notes which have consistent and common patterns of
        repeated reason tags in not-helpful ratings to achieve CRH status.
      inertiaDelta: Minimum amount which a note that has achieve CRH status must drop below the
        applicable threshold to lose CRH status.
    """
    super().__init__(seed)
    self._pseudoraters = pseudoraters
    self._minNumRatingsPerRater = minNumRatingsPerRater
    self._minNumRatersPerNote = minNumRatersPerNote
    self._minRatingsNeeded = minRatingsNeeded
    self._minMeanNoteScore = minMeanNoteScore
    self._minCRHVsCRNHRatio = minCRHVsCRNHRatio
    self._minRaterAgreeRatio = minRaterAgreeRatio
    self._crhThreshold = crhThreshold
    self._crnhThresholdIntercept = crnhThresholdIntercept
    self._crnhThresholdNoteFactorMultiplier = crnhThresholdNoteFactorMultiplier
    self._crnhThresholdNMIntercept = crnhThresholdNMIntercept
    self._crhSuperThreshold = crhSuperThreshold
    self._inertiaDelta = inertiaDelta
    self._mfRanker = matrix_factorization.MatrixFactorization()

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      c.internalNoteInterceptKey,
      c.internalNoteFactor1Key,
      c.internalRatingStatusKey,
      c.internalActiveRulesKey,
      c.activeFilterTagsKey,
    ]

  def get_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the helpfulnessScores output."""
    return [
      c.raterParticipantIdKey,
      c.internalRaterInterceptKey,
      c.internalRaterFactor1Key,
      c.crhCrnhRatioDifferenceKey,
      c.meanNoteScoreKey,
      c.raterAgreeRatioKey,
      c.aboveHelpfulnessThresholdKey,
    ]

  def get_auxiliary_note_info_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the auxiliaryNoteInfo output."""
    return [c.noteIdKey, c.ratingWeightKey,] + (
      c.notHelpfulTagsAdjustedColumns
      + c.notHelpfulTagsAdjustedRatioColumns
      + c.noteParameterUncertaintyTSVColumns
    )

  def _get_dropped_note_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from scoredNotes and auxiliaryNoteInfo."""
    return (
      [
        c.currentlyRatedHelpfulBoolKey,
        c.currentlyRatedNotHelpfulBoolKey,
        c.awaitingMoreRatingsBoolKey,
        c.currentLabelKey,
        c.classificationKey,
        c.numRatingsKey,
        c.noteAuthorParticipantIdKey,
      ]
      + c.helpfulTagsTSVOrder
      + c.notHelpfulTagsTSVOrder
    )

  def _get_dropped_user_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from helpfulnessScores output."""
    return []

  def _score_notes_and_users(
    self, ratings: pd.DataFrame, noteStatusHistory: pd.DataFrame
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the matrix factorization scoring algorithm.

    See links below for more info:
      https://twitter.github.io/communitynotes/ranking-notes/
      https://twitter.github.io/communitynotes/contributor-scores/.

    Args:
      ratings (pd.DataFrame): preprocessed ratings
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame]:
        noteScores pd.DataFrame: one row per note contained note scores and parameters.
        userScores pd.DataFrame: one row per user containing a column for each helpfulness score.
    """
    if self._seed is not None:
      torch.manual_seed(self._seed)

    ratingsForTraining = process_data.filter_ratings(
      ratings, self._minNumRatingsPerRater, self._minNumRatersPerNote
    )

    noteParamsUnfiltered, raterParamsUnfiltered, globalBias = self._mfRanker.run_mf(
      ratingsForTraining,
    )

    scoredNotes = note_ratings.compute_scored_notes(
      ratings,
      noteParamsUnfiltered,
      raterParamsUnfiltered,
      noteStatusHistory,
      minRatingsNeeded=self._minRatingsNeeded,
      crhThreshold=self._crhThreshold,
      crnhThresholdIntercept=self._crnhThresholdIntercept,
      crnhThresholdNoteFactorMultiplier=self._crnhThresholdNoteFactorMultiplier,
      crnhThresholdNMIntercept=self._crnhThresholdNMIntercept,
      crhSuperThreshold=self._crhSuperThreshold,
      inertiaDelta=self._inertiaDelta,
    )

    validRatings = note_ratings.get_valid_ratings(
      ratings,
      noteStatusHistory,
      scoredNotes[
        [
          c.noteIdKey,
          c.currentlyRatedHelpfulBoolKey,
          c.currentlyRatedNotHelpfulBoolKey,
          c.awaitingMoreRatingsBoolKey,
        ]
      ],
    )

    helpfulnessScores = helpfulness_scores.compute_general_helpfulness_scores(
      scoredNotes[
        [
          c.noteAuthorParticipantIdKey,
          c.currentlyRatedHelpfulBoolKey,
          c.currentlyRatedNotHelpfulBoolKey,
          c.internalNoteInterceptKey,
        ]
      ],
      validRatings,
      self._minMeanNoteScore,
      self._minCRHVsCRNHRatio,
      self._minRaterAgreeRatio,
    )

    ratingsHelpfulnessScoreFiltered = helpfulness_scores.filter_ratings_by_helpfulness_scores(
      ratingsForTraining, helpfulnessScores
    )

    noteParams, raterParams, globalBias = self._mfRanker.run_mf(
      ratingsHelpfulnessScoreFiltered,
      noteInit=noteParamsUnfiltered,
      userInit=raterParamsUnfiltered,
    )

    if self._pseudoraters:
      noteIdMap, raterIdMap, noteRatingIds = self._mfRanker.get_note_and_rater_id_maps(
        ratingsHelpfulnessScoreFiltered
      )

      extremeRaters = self._mfRanker.make_extreme_raters(raterParams, raterIdMap)

      (
        rawRescoredNotesWithEachExtraRater,
        notesWithConfidenceBounds,
      ) = self._mfRanker.fit_note_params_for_each_dataset_with_extreme_ratings(
        extremeRaters,
        noteRatingIds,
        ratingsHelpfulnessScoreFiltered,
        noteParams,
        raterParams,
        globalBias,
      )

      noteParams = noteParams.merge(
        notesWithConfidenceBounds.reset_index(), on="noteId", how="left"
      )

    else:
      for col in c.noteParameterUncertaintyTSVColumns:
        noteParams[col] = np.nan

    scoredNotes = note_ratings.compute_scored_notes(
      ratings,
      noteParams,
      raterParams,
      noteStatusHistory,
      minRatingsNeeded=self._minRatingsNeeded,
      crhThreshold=self._crhThreshold,
      crnhThresholdIntercept=self._crnhThresholdIntercept,
      crnhThresholdNoteFactorMultiplier=self._crnhThresholdNoteFactorMultiplier,
      crnhThresholdNMIntercept=self._crnhThresholdNMIntercept,
      crhSuperThreshold=self._crhSuperThreshold,
      inertiaDelta=self._inertiaDelta,
      finalRound=True,
    )
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

    return scoredNotes, helpfulnessScores
