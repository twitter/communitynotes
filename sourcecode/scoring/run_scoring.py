"""This file drives the scoring and user reputation logic for Community Notes.

This file defines "run_scoring" which invokes all Community Notes scoring algorithms,
merges results and computes contribution statistics for users.  run_scoring should be
intergrated into main files for execution in internal and external environments.
"""

from typing import List, Optional, Tuple

from . import constants as c, contributor_state, note_ratings, note_status_history, scoring_rules
from .mf_core_scorer import MFCoreScorer
from .mf_coverage_scorer import MFCoverageScorer
from .mf_expansion_scorer import MFExpansionScorer
from .scorer import Scorer
from .scoring_rules import RuleID

import numpy as np
import pandas as pd


def _get_scorers(seed: Optional[int], pseudoraters: Optional[bool]) -> List[Scorer]:
  """Instantiate all Scorer objects which should be used for note ranking.

  Args:
    seed (int, optional): if not None, base distinct seeds for the first and second MF rounds on this value
    pseudoraters (bool, optional): if True, compute optional pseudorater confidence intervals

  Returns:
    List[Scorer] containing instantiated Scorer objects for note ranking.
  """
  return [MFCoreScorer(seed, pseudoraters), MFExpansionScorer(seed), MFCoverageScorer(seed)]


def _merge_results(
  scoredNotes: pd.DataFrame,
  helpfulnessScores: pd.DataFrame,
  auxiliaryNoteInfo: pd.DataFrame,
  modelScoredNotes: pd.DataFrame,
  modelHelpfulnessScores: Optional[pd.DataFrame],
  modelauxiliaryNoteInfo: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Merges results from a specific model with results from prior models.

  The DFs returned by each model will be (outer) merged and passed through directly to the
  return value of run_scoring.  Column names must be unique in each DF with the exception of
  noteId or raterParticipantId, which are used to conduct the merge.

  Args:
    scoredNotes: pd.DataFrame containing key scoring results
    helpfulnessScores: pd.DataFrame containing contributor specific scoring results
    auxiliaryNoteInfo: pd.DataFrame containing intermediate scoring state
    modelScoredNotes: pd.DataFrame containing scoredNotes result for a particular model
    modelHelpfulnessScores: None or pd.DataFrame containing helpfulnessScores result for a particular model
    modelauxiliaryNoteInfo: None or pd.DataFrame containing auxiliaryNoteInfo result for a particular model

  Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
  """
  assert (set(modelScoredNotes.columns) & set(scoredNotes.columns)) == {
    c.noteIdKey
  }, "column names must be globally unique"
  scoredNotesSize = len(scoredNotes)
  scoredNotes = scoredNotes.merge(modelScoredNotes, on=c.noteIdKey, how="outer")
  assert len(scoredNotes) == scoredNotesSize, "scoredNotes should not expand"
  if modelHelpfulnessScores is not None:
    assert (set(modelHelpfulnessScores.columns) & set(helpfulnessScores.columns)) == {
      c.raterParticipantIdKey
    }, "column names must be globally unique"
    helpfulnessScores = helpfulnessScores.merge(
      modelHelpfulnessScores, on=c.raterParticipantIdKey, how="outer"
    )
  if modelauxiliaryNoteInfo is not None:
    assert (set(modelauxiliaryNoteInfo.columns) & set(auxiliaryNoteInfo.columns)) == {
      c.noteIdKey
    }, "column names must be globally unique"
    auxiliaryNoteInfoSize = len(auxiliaryNoteInfo)
    auxiliaryNoteInfo = auxiliaryNoteInfo.merge(modelauxiliaryNoteInfo, on=c.noteIdKey, how="outer")
    assert len(auxiliaryNoteInfo) == auxiliaryNoteInfoSize, "auxiliaryNoteInfo should not expand"
  return scoredNotes, helpfulnessScores, auxiliaryNoteInfo


def _run_scorers(
  scorers: List[Scorer],
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  userEnrollment: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Applies all Community Notes models to user ratings and returns merged result.

  Each model must return a scoredNotes DF and may return helpfulnessScores and auxiliaryNoteInfo.
  scoredNotes and auxiliaryNoteInfo will be forced to contain one row per note to guarantee
  that all notes can be assigned a status during meta scoring regardless of whether any
  individual scoring algorithm scored the note.

  Args:
    scorers (List[Scorer]): Instantiated Scorer objects for note ranking.
    ratings (pd.DataFrame): Complete DF containing all ratings after preprocessing.
    noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
    userEnrollment (pd.DataFrame): The enrollment state for each contributor

  Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
      scoredNotes pd.DataFrame: one row per note contained note scores and parameters.
      helpfulnessScores pd.DataFrame: one row per user containing a column for each helpfulness score.
      auxiliaryNoteInfo pd.DataFrame: one row per note containing supplemental values used in scoring.
  """
  scoredNotes = noteStatusHistory[[c.noteIdKey]].drop_duplicates()
  auxiliaryNoteInfo = noteStatusHistory[[c.noteIdKey]].drop_duplicates()
  helpfulnessScores = pd.DataFrame({c.raterParticipantIdKey: []})

  for scorer in scorers:
    modelScoredNotes, modelHelpfulnessScores, modelauxiliaryNoteInfo = scorer.score(
      ratings, noteStatusHistory, userEnrollment
    )
    scoredNotes, helpfulnessScores, auxiliaryNoteInfo = _merge_results(
      scoredNotes,
      helpfulnessScores,
      auxiliaryNoteInfo,
      modelScoredNotes,
      modelHelpfulnessScores,
      modelauxiliaryNoteInfo,
    )

  return scoredNotes, helpfulnessScores, auxiliaryNoteInfo


def meta_score(
  scoredNotes: pd.DataFrame, auxiliaryNoteInfo: pd.DataFrame, lockedStatus: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Determine final note status based on individual scoring results.

  This function applies logic merging the results of individual scorers to determine
  the final rating status of each note.  As part of determining the final rating status,
  we also apply scoring logic which exists independent of individual scorers (specifically
  note tag assignment and status locking).

  Args:
    scoredNotes: pd.DataFrame containing all scored note results.
    auxiliaryNoteInfo: pd.DataFrame containing tag aggregates
    lockedStatus: pd.DataFrame containing {noteId, status} pairs for all notes

  Returns:
    Tuple[pd.DataFrame, pd.DataFrame]:
      scoredNotesCols pd.DataFrame: one row per note contained note scores and parameters.
      auxiliaryNoteInfoCols pd.DataFrame: one row per note containing adjusted and ratio tag values
  """
  assert len(scoredNotes) == len(auxiliaryNoteInfo)
  scoredNotes = scoredNotes.merge(
    auxiliaryNoteInfo[[c.noteIdKey] + c.helpfulTagsTSVOrder + c.notHelpfulTagsTSVOrder],
    on=c.noteIdKey,
  )
  assert len(scoredNotes) == len(auxiliaryNoteInfo)
  rules = [
    scoring_rules.DefaultRule(RuleID.META_INITIAL_NMR, set(), c.needsMoreRatings),
    scoring_rules.ApplyModelResult(
      RuleID.CORE_MODEL, {RuleID.META_INITIAL_NMR}, c.coreRatingStatusKey
    ),
    scoring_rules.ApplyAdditiveModelResult(
      RuleID.COVERAGE_MODEL, {RuleID.CORE_MODEL}, c.coverageRatingStatusKey, 0.39
    ),
    scoring_rules.ScoringDriftGuard(RuleID.SCORING_DRIFT_GUARD, {RuleID.CORE_MODEL}, lockedStatus),
    scoring_rules.InsufficientExplanation(
      RuleID.INSUFFICIENT_EXPLANATION,
      {RuleID.CORE_MODEL},
      c.needsMoreRatings,
      c.minRatingsToGetTag,
      c.minTagsNeededForStatus,
    ),
  ]
  scoredNotes[c.firstTagKey] = np.nan
  scoredNotes[c.secondTagKey] = np.nan
  scoringResult = scoring_rules.apply_scoring_rules(
    scoredNotes,
    rules,
    c.finalRatingStatusKey,
    c.metaScorerActiveRulesKey,
    decidedByColumn=c.decidedByKey,
  )
  scoredNotesCols = scoringResult[
    [
      c.noteIdKey,
      c.finalRatingStatusKey,
      c.metaScorerActiveRulesKey,
      c.firstTagKey,
      c.secondTagKey,
      c.decidedByKey,
    ]
  ]
  auxiliaryNoteInfoCols = scoringResult[
    [
      c.noteIdKey,
      c.currentlyRatedHelpfulBoolKey,
      c.currentlyRatedNotHelpfulBoolKey,
      c.awaitingMoreRatingsBoolKey,
      c.unlockedRatingStatusKey,
    ]
  ]
  return scoredNotesCols, auxiliaryNoteInfoCols


def _compute_note_stats(
  ratings: pd.DataFrame, noteStatusHistory: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Generates DFs containing aggregate / global properties for each note.

  This funciton computes note aggregates over ratings and merges in selected fields
  from noteStatusHistory.  This function runs independent of individual scorers and
  augments the results of individual scorers because individual scorer may elect to
  consider only a subset of notes or ratings.  Computing on all available data after
  scorers have run guarantees completeness over all Community Notes data.

  Args:
    ratings: pd.DataFrame continaing *all* ratings on *all* notes from *all* users.
    noteStatusHistory: pd.DataFrame containing complete noteStatusHistory for all notes.

  Returns:
    Tuple[pd.DataFrame, pd.DataFrame]:
      scoredNotesCols pd.DataFrame: one row per note contained note scores and parameters.
      auxiliaryNoteInfoCols pd.DataFrame: one row per note containing adjusted and ratio tag values
  """
  noteStats = note_ratings.compute_note_stats(ratings, noteStatusHistory)
  scoredNotesCols = noteStats[[c.noteIdKey, c.classificationKey, c.createdAtMillisKey]]
  auxiliaryNoteInfoCols = noteStats[
    [
      c.noteIdKey,
      c.noteAuthorParticipantIdKey,
      c.numRatingsKey,
      c.createdAtMillisKey,
      c.numRatingsLast28DaysKey,
      c.currentLabelKey,
    ]
    + (c.helpfulTagsTSVOrder + c.notHelpfulTagsTSVOrder)
  ]
  return scoredNotesCols, auxiliaryNoteInfoCols


def _compute_helpfulness_scores(
  ratings: pd.DataFrame,
  scoredNotes: pd.DataFrame,
  auxiliaryNoteInfo: pd.DataFrame,
  helpfulnessScores: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  userEnrollment: pd.DataFrame,
) -> pd.DataFrame:
  """Computes usage statistics for Community Notes contributors.

  This function takes as input scoredNotes and auxiliaryNoteInfo (which represent the scoring
  results and associated statistics), helpfulnessScores (which contains any contributor metrics
  exported as a side effect of note scoring), ratings (which contains raw data about direct user
  contributions) along with noteStatusHistory (used to determine when notes were first assigned
  status) and userEnrollment (used to determine writing ability for users).

  See documentation below for additional information about contributor scores:
  https://twitter.github.io/communitynotes/contributor-scores/.

  Args:
      ratings (pd.DataFrame): preprocessed ratings
      scoredNotes (pd.DataFrame): notes with scores returned by MF scoring algorithm
      auxiliaryNoteInfo (pd.DataFrame): additional fields generated during note scoring
      helpfulnessScores (pd.DataFrame): BasicReputation scores for all raters
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
      userEnrollment (pd.DataFrame): The enrollment state for each contributor

  Returns:
      helpfulnessScores pd.DataFrame: one row per user containing a column for each helpfulness score.
  """
  assert len(scoredNotes) == len(auxiliaryNoteInfo), "notes in both note inputs must match"
  scoredNotesWithStats = scoredNotes.merge(
    auxiliaryNoteInfo,
    on=[c.noteIdKey, c.createdAtMillisKey],
    how="inner",
  )[
    [
      c.noteIdKey,
      c.finalRatingStatusKey,
      c.coreNoteInterceptKey,
      c.currentlyRatedHelpfulBoolKey,
      c.currentlyRatedNotHelpfulBoolKey,
      c.awaitingMoreRatingsBoolKey,
      c.createdAtMillisKey,
      c.noteAuthorParticipantIdKey,
      c.numRatingsKey,
      c.numRatingsLast28DaysKey,
    ]
  ]
  assert len(scoredNotesWithStats) == len(scoredNotes)

  contributorScores = contributor_state.get_contributor_scores(
    scoredNotesWithStats,
    ratings,
    noteStatusHistory,
  )
  contributorState = contributor_state.get_contributor_state(
    scoredNotesWithStats,
    ratings,
    noteStatusHistory,
    userEnrollment,
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

  return helpfulnessScores


def _validate(
  scoredNotes: pd.DataFrame,
  helpfulnessScores: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  auxiliaryNoteInfo: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Guarantee that each dataframe has the expected columns in the correct order.

  Args:
    scoredNotes (pd.DataFrame): notes with scores returned by MF scoring algorithm
    helpfulnessScores (pd.DataFrame): BasicReputation scores for all raters
    noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
    auxiliaryNoteInfo (pd.DataFrame): additional fields generated during note scoring

  Returns:
    Input arguments with columns potentially re-ordered.
  """
  assert set(scoredNotes.columns) == set(c.noteModelOutputTSVColumns)
  scoredNotes = scoredNotes[c.noteModelOutputTSVColumns]
  assert set(helpfulnessScores.columns) == set(c.raterModelOutputTSVColumns)
  helpfulnessScores = helpfulnessScores[c.raterModelOutputTSVColumns]
  assert set(noteStatusHistory.columns) == set(c.noteStatusHistoryTSVColumns)
  noteStatusHistory = noteStatusHistory[c.noteStatusHistoryTSVColumns]
  assert set(auxiliaryNoteInfo.columns) == set(c.auxiliaryScoredNotesTSVColumns)
  auxiliaryNoteInfo = auxiliaryNoteInfo[c.auxiliaryScoredNotesTSVColumns]
  return (scoredNotes, helpfulnessScores, noteStatusHistory, auxiliaryNoteInfo)


def run_scoring(
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  userEnrollment: pd.DataFrame,
  seed: Optional[int] = None,
  pseudoraters: Optional[bool] = True,
):
  """Invokes note scoring algorithms, merges results and computes user stats.

  Args:
    ratings (pd.DataFrame): preprocessed ratings
    noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
    userEnrollment (pd.DataFrame): The enrollment state for each contributor
    seed (int, optional): if not None, base distinct seeds for the first and second MF rounds on this value
    pseudoraters (bool, optional): if True, compute optional pseudorater confidence intervals

  Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
      scoredNotes pd.DataFrame: one row per note contained note scores and parameters.
      helpfulnessScores pd.DataFrame: one row per user containing a column for each helpfulness score.
      noteStatusHistory pd.DataFrame: one row per note containing when they got their most recent statuses.
      auxiliaryNoteInfo: one row per note containing adjusted and ratio tag values
  """
  scorers = _get_scorers(seed, pseudoraters)
  scoredNotes, helpfulnessScores, auxiliaryNoteInfo = _run_scorers(
    scorers, ratings, noteStatusHistory, userEnrollment
  )

  scoredNotesCols, auxiliaryNoteInfoCols = _compute_note_stats(ratings, noteStatusHistory)
  scoredNotes = scoredNotes.merge(scoredNotesCols, on=c.noteIdKey)
  auxiliaryNoteInfo = auxiliaryNoteInfo.merge(auxiliaryNoteInfoCols, on=c.noteIdKey)

  scoredNotesCols, auxiliaryNoteInfoCols = meta_score(
    scoredNotes, auxiliaryNoteInfo, noteStatusHistory[[c.noteIdKey, c.lockedStatusKey]]
  )
  scoredNotes = scoredNotes.merge(scoredNotesCols, on=c.noteIdKey)
  auxiliaryNoteInfo = auxiliaryNoteInfo.merge(auxiliaryNoteInfoCols, on=c.noteIdKey)

  assert len(scoredNotes) == len(
    noteStatusHistory
  ), "noteStatusHistory should be complete, and all notes should be scored."
  assert len(auxiliaryNoteInfo) == len(
    noteStatusHistory
  ), "noteStatusHistory should be complete, and all notes should be scored."

  helpfulnessScores = _compute_helpfulness_scores(
    ratings, scoredNotes, auxiliaryNoteInfo, helpfulnessScores, noteStatusHistory, userEnrollment
  )

  newNoteStatusHistory = note_status_history.update_note_status_history(
    noteStatusHistory, scoredNotes
  )
  assert len(newNoteStatusHistory) == len(
    noteStatusHistory
  ), "noteStatusHistory should contain all notes after preprocessing"

  return _validate(scoredNotes, helpfulnessScores, newNoteStatusHistory, auxiliaryNoteInfo)
