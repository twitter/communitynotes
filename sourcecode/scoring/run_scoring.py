"""This file drives the scoring and user reputation logic for Community Notes.

This file defines "run_scoring" which invokes all Community Notes scoring algorithms,
merges results and computes contribution statistics for users.  run_scoring should be
intergrated into main files for execution in internal and external environments.
"""

from collections import namedtuple
import concurrent.futures
from itertools import chain
import multiprocessing
import time
from typing import Dict, List, Optional, Set, Tuple

from . import constants as c, contributor_state, note_ratings, note_status_history, scoring_rules
from .enums import Scorers
from .mf_base_scorer import MFBaseScorer
from .mf_core_scorer import MFCoreScorer
from .mf_expansion_scorer import MFExpansionScorer
from .mf_group_scorer import coalesce_group_models, groupScorerCount, MFGroupScorer
from .scorer import Scorer
from .scoring_rules import RuleID

import numpy as np
import pandas as pd


ModelResult = namedtuple("ModelResult", ["scoredNotes", "helpfulnessScores", "auxiliaryNoteInfo"])


def _check_flips(
  scoredNotes: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  userEnrollment: pd.DataFrame,
  modelingPopulations: List[str],
  pctFlips: float,
  resultCol: str,
) -> bool:
  """
  Args:
    scoredNotes (pd.DataFrame): notes with scores returned by MF scoring algorithm to evaluate against NSH for flips
    noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
    userEnrollment (pd.DataFrame): The enrollment state for each contributor
    modelingPopulation (str): name of modelingPopulation(s) of authors for which to evaluate CRH flips
    pctFlips (float): percent of flips of unlocked notes that triggers a rerun
    resultCol (str): name of column with noteStatus for model

  Returns:
    bool whether unlocked note flips exceed acceptable level, triggering a rerun
  """
  # combine noteStatusHistory with userEnrollment modelingPopulation
  popNsh = noteStatusHistory.merge(
    userEnrollment[[c.participantIdKey, c.modelingPopulationKey]],
    left_on=c.noteAuthorParticipantIdKey,
    right_on=c.participantIdKey,
    how="inner",
  )
  unlockedPopNsh = popNsh.loc[
    (popNsh[c.modelingPopulationKey].isin(modelingPopulations))
    & (popNsh[c.timestampMillisOfStatusLockKey].isna())
  ]

  # check how many unlocked notes that appeared in previous run have now flipped status
  unlockedPopNsh = unlockedPopNsh.merge(
    scoredNotes[[c.noteIdKey, resultCol]], on=c.noteIdKey, how="inner"
  )
  statusFlips = len(
    unlockedPopNsh.loc[
      (unlockedPopNsh[c.currentLabelKey] != unlockedPopNsh[resultCol])
      & (
        (unlockedPopNsh[c.currentLabelKey] == c.currentlyRatedHelpful)
        | (unlockedPopNsh[resultCol] == c.currentlyRatedHelpful)
      )
    ]
  )
  # take max w/1 to avoid divide by zero error when no prevCRH
  prevCrh = max(
    1, len(unlockedPopNsh.loc[unlockedPopNsh[c.currentLabelKey] == c.currentlyRatedHelpful])
  )
  print(f"Percentage of previous CRH flipping status: {(statusFlips / prevCrh)}")

  return (statusFlips / prevCrh) > pctFlips


def _get_scorers(
  seed: Optional[int], pseudoraters: Optional[bool], enabledScorers: Optional[Set[Scorers]]
) -> Dict[Scorers, List[Scorer]]:
  """Instantiate all Scorer objects which should be used for note ranking.

  Args:
    seed (int, optional): if not None, base distinct seeds for the first and second MF rounds on this value
    pseudoraters (bool, optional): if True, compute optional pseudorater confidence intervals
    enabledScorers: if not None, set of which scorers should be instantiated and enabled

  Returns:
    Dict[Scorers, List[Scorer]] containing instantiated Scorer objects for note ranking.
  """
  scorers: Dict[Scorers, List[Scorer]] = dict()

  if enabledScorers is None or Scorers.MFCoreScorer in enabledScorers:
    scorers[Scorers.MFCoreScorer] = [MFCoreScorer(seed, pseudoraters)]
  if enabledScorers is None or Scorers.MFExpansionScorer in enabledScorers:
    scorers[Scorers.MFExpansionScorer] = [MFExpansionScorer(seed)]
  if enabledScorers is None or Scorers.MFGroupScorer in enabledScorers:
    # Note that index 0 is reserved, corresponding to no group assigned, so scoring group
    # numbers begin with index 1.
    scorers[Scorers.MFGroupScorer] = [
      MFGroupScorer(groupNumber=i) for i in range(1, groupScorerCount + 1)
    ]

  return scorers


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
  # Merge scoredNotes
  assert (set(modelScoredNotes.columns) & set(scoredNotes.columns)) == {
    c.noteIdKey
  }, "column names must be globally unique"
  scoredNotesSize = len(scoredNotes)
  scoredNotes = scoredNotes.merge(modelScoredNotes, on=c.noteIdKey, how="outer")
  assert len(scoredNotes) == scoredNotesSize, "scoredNotes should not expand"
  # Merge helpfulnessScores
  if modelHelpfulnessScores is not None:
    assert (set(modelHelpfulnessScores.columns) & set(helpfulnessScores.columns)) == {
      c.raterParticipantIdKey
    }, "column names must be globally unique"
    helpfulnessScores = helpfulnessScores.merge(
      modelHelpfulnessScores, on=c.raterParticipantIdKey, how="outer"
    )
  # Merge auxiliaryNoteInfo
  if modelauxiliaryNoteInfo is not None:
    assert (set(modelauxiliaryNoteInfo.columns) & set(auxiliaryNoteInfo.columns)) == {
      c.noteIdKey
    }, "column names must be globally unique"
    auxiliaryNoteInfoSize = len(auxiliaryNoteInfo)
    auxiliaryNoteInfo = auxiliaryNoteInfo.merge(modelauxiliaryNoteInfo, on=c.noteIdKey, how="outer")
    assert len(auxiliaryNoteInfo) == auxiliaryNoteInfoSize, "auxiliaryNoteInfo should not expand"
  return scoredNotes, helpfulnessScores, auxiliaryNoteInfo


def _run_scorer_parallelizable(scorer, ratings, noteStatusHistory, userEnrollment):
  runCounter = 0
  scorerStartTime = time.perf_counter()
  result = None
  flips = False

  # TODO: encapsulate this at a lower level so we're not accessing private state and can deal
  # with the case where there are no ratings for a scorer to run on after filtering.
  if isinstance(scorer, (MFBaseScorer, MFCoreScorer, MFExpansionScorer)):
    while (
      (len(scorer._mfRanker.train_errors) == 0)
      or (scorer._mfRanker.train_errors[-1] > c.maxTrainError)
      or (flips and runCounter < c.maxReruns)
    ):
      runCounter += 1
      result = ModelResult(*scorer.score(ratings, noteStatusHistory, userEnrollment))
      if runCounter > c.maxReruns:
        break
      # check for notes createdat > lock time, not more than x% have changed status from nsh
      if isinstance(scorer, MFExpansionScorer):
        flips = _check_flips(
          result.scoredNotes,
          noteStatusHistory,
          userEnrollment,
          [c.expansion],
          c.expansionFlipPct,
          c.expansionRatingStatusKey,
        )
      elif isinstance(scorer, MFCoreScorer):
        flips = _check_flips(
          result.scoredNotes,
          noteStatusHistory,
          userEnrollment,
          [c.core],
          c.coreFlipPct,
          c.coreRatingStatusKey,
        )
      if flips:
        if scorer._seed is not None:
          scorer._seed = scorer._seed + 1
  else:
    result = ModelResult(*scorer.score(ratings, noteStatusHistory, userEnrollment))
    runCounter += 1

  scorerEndTime = time.perf_counter()
  return result, (scorerEndTime - scorerStartTime), runCounter


def _run_scorers(
  scorers: List[Scorer],
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  userEnrollment: pd.DataFrame,
  runParallel: bool = False,
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
  # Apply scoring algorithms
  overallStartTime = time.perf_counter()
  if runParallel:
    with concurrent.futures.ProcessPoolExecutor(
      mp_context=multiprocessing.get_context("spawn")
    ) as executor:
      print(f"Starting parallel scorer execution with {len(scorers)} scorers.")
      futures = [
        executor.submit(_run_scorer_parallelizable, s, ratings, noteStatusHistory, userEnrollment)
        for s in scorers
      ]
      modelResultsAndTimes = [f.result() for f in futures]
  else:
    modelResultsAndTimes = [
      _run_scorer_parallelizable(s, ratings, noteStatusHistory, userEnrollment) for s in scorers
    ]

  modelResults, scorerTimes, runCounts = zip(*modelResultsAndTimes)

  overallTime = time.perf_counter() - overallStartTime
  print(
    f"""----
  Completed individual scorers. Ran in parallel: {runParallel}
  Succeeded in {overallTime:.2f} seconds. Individual scorers: {['{:.2f}'.format(t) for t in scorerTimes]} seconds.
  Run Counters (>1 if loss was too high, forcing a re-train): {dict(zip([type(scorer) for scorer in scorers], runCounts))}
  ----"""
  )

  # Initialize return data frames.
  scoredNotes = noteStatusHistory[[c.noteIdKey]].drop_duplicates()
  auxiliaryNoteInfo = noteStatusHistory[[c.noteIdKey]].drop_duplicates()
  helpfulnessScores = pd.DataFrame({c.raterParticipantIdKey: []})

  # Merge the results
  for modelResult in modelResults:
    scoredNotes, helpfulnessScores, auxiliaryNoteInfo = _merge_results(
      scoredNotes,
      helpfulnessScores,
      auxiliaryNoteInfo,
      modelResult.scoredNotes,
      modelResult.helpfulnessScores,
      modelResult.auxiliaryNoteInfo,
    )
  scoredNotes, helpfulnessScores = coalesce_group_models(scoredNotes, helpfulnessScores)

  return scoredNotes, helpfulnessScores, auxiliaryNoteInfo


def meta_score(
  scorers: Dict[Scorers, List[Scorer]],
  scoredNotes: pd.DataFrame,
  auxiliaryNoteInfo: pd.DataFrame,
  lockedStatus: pd.DataFrame,
  enabledScorers: Optional[Set[Scorers]],
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
    enabledScorers: if not None, set of which scorers should be instantiated and enabled

  Returns:
    Tuple[pd.DataFrame, pd.DataFrame]:
      scoredNotesCols pd.DataFrame: one row per note contained note scores and parameters.
      auxiliaryNoteInfoCols pd.DataFrame: one row per note containing adjusted and ratio tag values
  """
  # Temporarily merge helpfulness tag aggregates into scoredNotes so we can run InsufficientExplanation
  assert len(scoredNotes) == len(auxiliaryNoteInfo)
  scoredNotes = scoredNotes.merge(
    auxiliaryNoteInfo[[c.noteIdKey] + c.helpfulTagsTSVOrder + c.notHelpfulTagsTSVOrder],
    on=c.noteIdKey,
  )
  assert len(scoredNotes) == len(auxiliaryNoteInfo)
  rules: List[scoring_rules.ScoringRule] = [
    scoring_rules.DefaultRule(RuleID.META_INITIAL_NMR, set(), c.needsMoreRatings)
  ]
  # Only attach meta-scoring rules for models which actually run.
  if enabledScorers is None or Scorers.MFExpansionScorer in enabledScorers:
    rules.append(
      scoring_rules.ApplyModelResult(
        RuleID.EXPANSION_MODEL, {RuleID.META_INITIAL_NMR}, c.expansionRatingStatusKey
      )
    )
  if enabledScorers is None or Scorers.MFCoreScorer in enabledScorers:
    rules.append(
      scoring_rules.ApplyModelResult(
        RuleID.CORE_MODEL, {RuleID.META_INITIAL_NMR}, c.coreRatingStatusKey
      )
    )
  if enabledScorers is None or Scorers.MFGroupScorer in enabledScorers:
    # TODO: modify this code to work when MFExpansionScorer is disabled by the system test
    assert len(scorers[Scorers.MFCoreScorer]) == 1
    assert len(scorers[Scorers.MFExpansionScorer]) == 1
    coreScorer = scorers[Scorers.MFCoreScorer][0]
    assert isinstance(coreScorer, MFCoreScorer)
    expnasionScorer = scorers[Scorers.MFExpansionScorer][0]
    assert isinstance(expnasionScorer, MFExpansionScorer)
    coreCrhThreshold = coreScorer.get_crh_threshold()
    expansionCrhThreshold = expnasionScorer.get_crh_threshold()
    for i in range(1, groupScorerCount + 1):
      rules.append(
        scoring_rules.ApplyGroupModelResult(
          RuleID[f"GROUP_MODEL_{i}"],
          {RuleID.EXPANSION_MODEL, RuleID.CORE_MODEL},
          i,
          coreCrhThreshold,
          expansionCrhThreshold,
        )
      )
  rules.extend(
    [
      scoring_rules.ScoringDriftGuard(
        RuleID.SCORING_DRIFT_GUARD, {RuleID.CORE_MODEL}, lockedStatus
      ),
      # TODO: The rule below both sets tags for notes which are CRH / CRNH and unsets status for
      # any notes which are CRH / CRNH but don't have enough ratings to assign two tags.  The later
      # behavior can lead to unsetting locked status.  We should refactor this code to (1) remove
      # the behavior which unsets status (instead tags will be assigned on a best effort basis) and
      # (2) set tags in logic which is not run as a ScoringRule (since ScoringRules function to
      # update note status).
      scoring_rules.InsufficientExplanation(
        RuleID.INSUFFICIENT_EXPLANATION,
        {RuleID.CORE_MODEL},
        c.needsMoreRatings,
        c.minRatingsToGetTag,
        c.minTagsNeededForStatus,
      ),
    ]
  )
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
  # Generate a uunified view of note scoring information for computing contributor stats
  assert len(scoredNotes) == len(auxiliaryNoteInfo), "notes in both note inputs must match"
  scoredNotesWithStats = scoredNotes.merge(
    # noteId and timestamp are the only common fields, and should always be equal.
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

  # Return one row per rater with stats including trackrecord identifying note labels.
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

  # Pass timestampOfLastEarnOut through to raterModelOutput.
  helpfulnessScores = helpfulnessScores.merge(
    userEnrollment[[c.participantIdKey, c.timestampOfLastEarnOut]],
    left_on=c.raterParticipantIdKey,
    right_on=c.participantIdKey,
    how="left",
  ).drop(c.participantIdKey, axis=1)
  # If field is not set by userEvent or by update script, ok to default to 1
  helpfulnessScores[c.timestampOfLastEarnOut].fillna(1, inplace=True)

  return helpfulnessScores


def _add_deprecated_columns(scoredNotes: pd.DataFrame) -> pd.DataFrame:
  """Impute columns which are no longer used but must be maintained in output.

  Args:
    scoredNotes: DataFrame containing note scoring output

  Returns:
    scoredNotes augmented to include deprecated columns filled with dummy values
  """
  for column, columnType in c.deprecatedNoteModelOutputTSVColumnsAndTypes:
    assert column not in scoredNotes.columns
    if columnType == np.double:
      scoredNotes[column] = np.nan
    elif columnType == np.str:
      scoredNotes[column] = ""
    else:
      assert False, f"column type {columnType} unsupported"
  return scoredNotes


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
  assert set(scoredNotes.columns) == set(
    c.noteModelOutputTSVColumns
  ), f"Got {sorted(scoredNotes.columns)}, expected {sorted(c.noteModelOutputTSVColumns)}"
  scoredNotes = scoredNotes[c.noteModelOutputTSVColumns]
  assert set(helpfulnessScores.columns) == set(
    c.raterModelOutputTSVColumns
  ), f"Got {sorted(helpfulnessScores.columns)}, expected {sorted(c.raterModelOutputTSVColumns)}"
  helpfulnessScores = helpfulnessScores[c.raterModelOutputTSVColumns]
  assert set(noteStatusHistory.columns) == set(
    c.noteStatusHistoryTSVColumns
  ), f"Got {sorted(noteStatusHistory.columns)}, expected {sorted(c.noteStatusHistoryTSVColumns)}"
  noteStatusHistory = noteStatusHistory[c.noteStatusHistoryTSVColumns]
  assert set(auxiliaryNoteInfo.columns) == set(
    c.auxiliaryScoredNotesTSVColumns
  ), f"Got {sorted(auxiliaryNoteInfo.columns)}, expected {sorted(c.auxiliaryScoredNotesTSVColumns)}"
  auxiliaryNoteInfo = auxiliaryNoteInfo[c.auxiliaryScoredNotesTSVColumns]
  return (scoredNotes, helpfulnessScores, noteStatusHistory, auxiliaryNoteInfo)


def run_scoring(
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  userEnrollment: pd.DataFrame,
  seed: Optional[int] = None,
  pseudoraters: Optional[bool] = True,
  enabledScorers: Optional[Set[Scorers]] = None,
  strictColumns: bool = True,
):
  """Invokes note scoring algorithms, merges results and computes user stats.

  Args:
    ratings (pd.DataFrame): preprocessed ratings
    noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
    userEnrollment (pd.DataFrame): The enrollment state for each contributor
    seed (int, optional): if not None, base distinct seeds for the first and second MF rounds on this value
    pseudoraters (bool, optional): if True, compute optional pseudorater confidence intervals
    enabledScorers (Set[Scorers], optional): Scorers which should be instantiated
    strictColumns (bool, optional): if True, validate which columns are present

  Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
      scoredNotes pd.DataFrame: one row per note contained note scores and parameters.
      helpfulnessScores pd.DataFrame: one row per user containing a column for each helpfulness score.
      noteStatusHistory pd.DataFrame: one row per note containing when they got their most recent statuses.
      auxiliaryNoteInfo: one row per note containing adjusted and ratio tag values
  """
  # Apply individual scoring models and obtained merged result.
  scorers = _get_scorers(seed, pseudoraters, enabledScorers)
  scoredNotes, helpfulnessScores, auxiliaryNoteInfo = _run_scorers(
    list(chain(*scorers.values())), ratings, noteStatusHistory, userEnrollment
  )

  # Augment scoredNotes and auxiliaryNoteInfo with additional attributes for each note
  # which are computed over the corpus of notes / ratings as a whole and are independent
  # of any particular model.
  scoredNotesCols, auxiliaryNoteInfoCols = _compute_note_stats(ratings, noteStatusHistory)
  scoredNotes = scoredNotes.merge(scoredNotesCols, on=c.noteIdKey)
  auxiliaryNoteInfo = auxiliaryNoteInfo.merge(auxiliaryNoteInfoCols, on=c.noteIdKey)

  # Assign final status to notes based on individual model scores and note attributes.
  scoredNotesCols, auxiliaryNoteInfoCols = meta_score(
    scorers,
    scoredNotes,
    auxiliaryNoteInfo,
    noteStatusHistory[[c.noteIdKey, c.lockedStatusKey]],
    enabledScorers,
  )
  scoredNotes = scoredNotes.merge(scoredNotesCols, on=c.noteIdKey)
  auxiliaryNoteInfo = auxiliaryNoteInfo.merge(auxiliaryNoteInfoCols, on=c.noteIdKey)

  # Validate that no notes were dropped or duplicated.
  assert len(scoredNotes) == len(
    noteStatusHistory
  ), "noteStatusHistory should be complete, and all notes should be scored."
  assert len(auxiliaryNoteInfo) == len(
    noteStatusHistory
  ), "noteStatusHistory should be complete, and all notes should be scored."

  # Compute contribution statistics and enrollment state for users.
  helpfulnessScores = _compute_helpfulness_scores(
    ratings, scoredNotes, auxiliaryNoteInfo, helpfulnessScores, noteStatusHistory, userEnrollment
  )

  # Merge scoring results into noteStatusHistory.
  newNoteStatusHistory = note_status_history.update_note_status_history(
    noteStatusHistory, scoredNotes
  )
  assert len(newNoteStatusHistory) == len(
    noteStatusHistory
  ), "noteStatusHistory should contain all notes after preprocessing"

  # Skip validation and selection out output columns if the set of scorers is overridden.
  scoredNotes = _add_deprecated_columns(scoredNotes)
  if strictColumns:
    scoredNotes, helpfulnessScores, newNoteStatusHistory, auxiliaryNoteInfo = _validate(
      scoredNotes, helpfulnessScores, newNoteStatusHistory, auxiliaryNoteInfo
    )
  return scoredNotes, helpfulnessScores, newNoteStatusHistory, auxiliaryNoteInfo
