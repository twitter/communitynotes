"""This file drives the scoring and user reputation logic for Community Notes.

This file defines "run_scoring" which invokes all Community Notes scoring algorithms,
merges results and computes contribution statistics for users.  run_scoring should be
intergrated into main files for execution in internal and external environments.
"""
import concurrent.futures
import copy
from itertools import chain
import multiprocessing
from multiprocessing import shared_memory  # type: ignore
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

from . import constants as c, contributor_state, note_ratings, note_status_history, scoring_rules
from .constants import FinalScoringArgs, ModelResult, PrescoringArgs, ScoringArgs
from .enums import Scorers, Topics
from .matrix_factorization.normalized_loss import NormalizedLossHyperparameters
from .mf_core_scorer import MFCoreScorer
from .mf_expansion_plus_scorer import MFExpansionPlusScorer
from .mf_expansion_scorer import MFExpansionScorer
from .mf_group_scorer import (
  MFGroupScorer,
  coalesce_group_model_helpfulness_scores,
  coalesce_group_model_scored_notes,
  groupScorerCount,
  trialScoringGroup,
)
from .mf_topic_scorer import MFTopicScorer, coalesce_topic_models
from .post_selection_similarity import (
  PostSelectionSimilarity,
  filter_ratings_by_post_selection_similarity,
)
from .process_data import CommunityNotesDataLoader, filter_input_data_for_testing, preprocess_data
from .reputation_scorer import ReputationScorer
from .scorer import Scorer
from .scoring_rules import RuleID
from .topic_model import TopicModel

import numpy as np
import pandas as pd
import sklearn


def _get_scorers(
  seed: Optional[int],
  pseudoraters: Optional[bool],
  enabledScorers: Optional[Set[Scorers]],
  useStableInitialization: bool = True,
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
    scorers[Scorers.MFCoreScorer] = [
      MFCoreScorer(seed, pseudoraters, useStableInitialization=useStableInitialization, threads=12)
    ]
  if enabledScorers is None or Scorers.MFExpansionScorer in enabledScorers:
    scorers[Scorers.MFExpansionScorer] = [
      MFExpansionScorer(seed, useStableInitialization=useStableInitialization, threads=12)
    ]
  if enabledScorers is None or Scorers.MFExpansionPlusScorer in enabledScorers:
    scorers[Scorers.MFExpansionPlusScorer] = [
      MFExpansionPlusScorer(seed, useStableInitialization=useStableInitialization, threads=12)
    ]
  if enabledScorers is None or Scorers.ReputationScorer in enabledScorers:
    scorers[Scorers.ReputationScorer] = [
      ReputationScorer(seed, useStableInitialization=useStableInitialization, threads=12)
    ]
  if enabledScorers is None or Scorers.MFGroupScorer in enabledScorers:
    # Note that index 0 is reserved, corresponding to no group assigned, so scoring group
    # numbers begin with index 1.
    scorers[Scorers.MFGroupScorer] = [
      # Scoring Group 13 is currently the largest by far, so total runtime benefits from
      # adding the group scorers in descending order so we start work on Group 13 first.
      MFGroupScorer(groupNumber=i, seed=seed)
      for i in range(groupScorerCount, 0, -1)
      if i != trialScoringGroup
    ]
    scorers[Scorers.MFGroupScorer].append(
      MFGroupScorer(
        groupNumber=trialScoringGroup,
        seed=seed,
        noteInterceptLambda=0.03 * 30,
        userInterceptLambda=0.03 * 5,
        globalInterceptLambda=0.03 * 5,
        noteFactorLambda=0.03 / 3,
        userFactorLambda=0.03 / 4,
        diamondLambda=0.03 * 25,
        normalizedLossHyperparameters=NormalizedLossHyperparameters(
          globalSignNorm=True, noteSignAlpha=None, noteNormExp=0, raterNormExp=-0.25
        ),
        maxFinalMFTrainError=0.16,
        groupThreshold=0.4,
        minMeanNoteScore=-0.01,
        crhThreshold=0.15,
        crhSuperThreshold=None,
        crnhThresholdIntercept=-0.01,
        crnhThresholdNoteFactorMultiplier=0,
        crnhThresholdNMIntercept=-0.02,
        lowDiligenceThreshold=1000,
        factorThreshold=0.4,
        multiplyPenaltyByHarassmentScore=False,
        minimumHarassmentScoreToPenalize=2.5,
        tagConsensusHarassmentHelpfulRatingPenalty=10,
        tagFilterPercentile=90,
        incorrectFilterThreshold=1.5,
      )
    )
  if enabledScorers is None or Scorers.MFTopicScorer in enabledScorers:
    scorers[Scorers.MFTopicScorer] = [
      MFTopicScorer(topicName=topic.name, seed=seed) for topic in Topics
    ]

  return scorers


def _merge_results(
  scoredNotes: pd.DataFrame,
  auxiliaryNoteInfo: pd.DataFrame,
  modelScoredNotes: pd.DataFrame,
  modelauxiliaryNoteInfo: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Merges results from a specific model with results from prior models.

  The DFs returned by each model will be (outer) merged and passed through directly to the
  return value of run_scoring.  Column names must be unique in each DF with the exception of
  noteId or raterParticipantId, which are used to conduct the merge.

  Args:
    scoredNotes: pd.DataFrame containing key scoring results
    auxiliaryNoteInfo: pd.DataFrame containing intermediate scoring state
    modelScoredNotes: pd.DataFrame containing scoredNotes result for a particular model
    modelauxiliaryNoteInfo: None or pd.DataFrame containing auxiliaryNoteInfo result for a particular model

  Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
  """
  # Merge scoredNotes
  assert (set(modelScoredNotes.columns) & set(scoredNotes.columns)) == {
    c.noteIdKey
  }, "column names must be globally unique"
  scoredNotesSize = len(scoredNotes)
  unsafeAllowed = set(
    [
      c.noteIdKey,
      c.defaultIndexKey,
    ]
    + [f"{c.modelingGroupKey}_{group}" for group in range(groupScorerCount, 0, -1)]
    + [f"{c.topicNoteConfidentKey}_{topic.name}" for topic in Topics]
    + [f"{c.groupNumFinalRoundRatingsKey}_{group}" for group in range(groupScorerCount, 0, -1)]
  )
  scoredNotes = scoredNotes.merge(
    modelScoredNotes,
    on=c.noteIdKey,
    how="outer",
    unsafeAllowed=unsafeAllowed,
  )
  assert len(scoredNotes) == scoredNotesSize, "scoredNotes should not expand"

  # Merge auxiliaryNoteInfo
  if modelauxiliaryNoteInfo is not None:
    assert (set(modelauxiliaryNoteInfo.columns) & set(auxiliaryNoteInfo.columns)) == {
      c.noteIdKey
    }, "column names must be globally unique"
    auxiliaryNoteInfoSize = len(auxiliaryNoteInfo)
    auxiliaryNoteInfo = auxiliaryNoteInfo.merge(modelauxiliaryNoteInfo, on=c.noteIdKey, how="outer")
    assert len(auxiliaryNoteInfo) == auxiliaryNoteInfoSize, "auxiliaryNoteInfo should not expand"

  return scoredNotes, auxiliaryNoteInfo


def _load_data_with_data_loader_parallelizable(
  dataLoader: CommunityNotesDataLoader, scoringArgs: ScoringArgs
) -> ScoringArgs:
  """
  Load data from the dataLoader into the scoringArgs object. This function is designed to be run
  in a multiprocessing pool,

  Deprecated: prefer _load_data_from_shared_memory_parallelizable.
  """
  _, ratings, noteStatusHistory, userEnrollment = dataLoader.get_data()

  scoringArgs.ratings = ratings
  scoringArgs.noteStatusHistory = noteStatusHistory
  scoringArgs.userEnrollment = userEnrollment
  if type(scoringArgs) == FinalScoringArgs:
    prescoringNoteModelOutput, prescoringRaterParams = dataLoader.get_prescoring_model_output()
    scoringArgs.prescoringNoteModelOutput = prescoringNoteModelOutput
    scoringArgs.prescoringRaterModelOutput = prescoringRaterParams
  return scoringArgs


def _load_data_from_shared_memory_parallelizable(
  scoringArgsSharedMemory: c.ScoringArgsSharedMemory, scoringArgs: ScoringArgs
) -> ScoringArgs:
  """
  Load data from shared memory into the scoringArgs object. This function is designed to be run
  in a multiprocessing pool.
  """
  scoringArgs.noteTopics = get_df_from_shared_memory(scoringArgsSharedMemory.noteTopics)
  scoringArgs.ratings = get_df_from_shared_memory(scoringArgsSharedMemory.ratings)
  scoringArgs.noteStatusHistory = get_df_from_shared_memory(
    scoringArgsSharedMemory.noteStatusHistory
  )
  scoringArgs.userEnrollment = get_df_from_shared_memory(scoringArgsSharedMemory.userEnrollment)

  if type(scoringArgs) == FinalScoringArgs:
    assert type(scoringArgsSharedMemory) == c.FinalScoringArgsSharedMemory
    scoringArgs.prescoringNoteModelOutput = get_df_from_shared_memory(
      scoringArgsSharedMemory.prescoringNoteModelOutput
    )
    scoringArgs.prescoringRaterModelOutput = get_df_from_shared_memory(
      scoringArgsSharedMemory.prescoringRaterModelOutput
    )
  return scoringArgs


def _run_scorer_parallelizable(
  scorer: Scorer,
  runParallel: bool,
  scoringArgs: ScoringArgs,
  dataLoader: Optional[CommunityNotesDataLoader] = None,
  scoringArgsSharedMemory=None,
) -> Tuple[ModelResult, float]:
  """
  Run scoring (either prescoring or final scoring) for a single scorer.
  This function is designed to be run in a multiprocessing pool, so you can run this function
  for each scorer in parallel.

  We determine whether to run prescoring or final scoring based on the type of scoringArgs
    (PrescoringArgs or FinalScoringArgs).

  If runParallel is False, then we read input dataframes from scoringArgs.

  If runParallel is True, then we ignore the dataframe attributes of scoringArgs, and read
  the input dataframes from shared memory if scoringArgsSharedMemory is not None (preferred),
  or from the dataLoader if scoringArgsSharedMemory is None. However, using the dataLoader to
  re-read the dataframes from disk is much slower than using shared memory and is deprecated.
  """
  scorerStartTime = time.perf_counter()

  # Load data if multiprocessing
  if runParallel:
    with c.time_block(f"{scorer.get_name()} run_scorer_parallelizable: Loading data"):
      scoringArgs.remove_large_args_for_multiprocessing()  # Should be redundant
      scoringArgs = copy.deepcopy(scoringArgs)

      if scoringArgsSharedMemory is not None:
        print(
          f"{scorer.get_name()} run_scorer_parallelizable just started in parallel: loading data from shared memory."
        )
        scoringArgs = _load_data_from_shared_memory_parallelizable(
          scoringArgsSharedMemory, scoringArgs
        )
        print(
          f"{scorer.get_name()} run_scorer_parallelizable just finished loading data from shared memory."
        )
      elif dataLoader is not None:
        print(
          f"{scorer.get_name()} run_scorer_parallelizable just started in parallel: loading data with dataLoader."
        )
        scoringArgs = _load_data_with_data_loader_parallelizable(dataLoader, scoringArgs)
      else:
        raise ValueError(
          "Must provide either scoringArgsSharedMemory or dataLoader to run parallel"
        )

  # Run scoring
  scorerStartTime = time.perf_counter()
  if type(scoringArgs) == PrescoringArgs:
    scoringResults = scorer.prescore(scoringArgs)
  elif type(scoringArgs) == FinalScoringArgs:
    scoringResults = scorer.score_final(scoringArgs)
  else:
    raise ValueError(f"Unknown scoringArgs type: {type(scoringArgs)}")
  scorerEndTime = time.perf_counter()

  return scoringResults, (scorerEndTime - scorerStartTime)


def save_df_to_shared_memory(df: pd.DataFrame, shms: List) -> c.SharedMemoryDataframeInfo:
  """
  Intended to be called before beginning multiprocessing: saves the df to shared memory
  and returns the info needed to access it, as well as appends it to the list of shared memory objects
  so it's not garbage collected and can be closed later.
  """
  cols = df.columns
  data = df.to_numpy()
  df_dtypes_dict = dict(list(zip(df.columns, df.dtypes)))
  shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
  np_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
  np_array[:] = data[:]
  shms.append(shm)  # save the shared memory object so we can close it later
  return c.SharedMemoryDataframeInfo(
    sharedMemoryName=shm.name,
    columns=cols,
    dataShape=data.shape,
    dtypesDict=df_dtypes_dict,
    npDtype=np_array.dtype,
  )


def get_df_from_shared_memory(sharedMemoryDfInfo: c.SharedMemoryDataframeInfo) -> pd.DataFrame:
  """
  Intended to be called from a process within a multiprocessing pool in parallel.
  Read a dataframe from shared memory and return it.
  """
  existing_shm = shared_memory.SharedMemory(name=sharedMemoryDfInfo.sharedMemoryName)
  np_array = np.ndarray(
    sharedMemoryDfInfo.dataShape, buffer=existing_shm.buf, dtype=sharedMemoryDfInfo.npDtype
  )
  df = pd.DataFrame(np_array, columns=sharedMemoryDfInfo.columns)
  df = df.astype(sharedMemoryDfInfo.dtypesDict)
  return df


def _save_dfs_to_shared_memory(
  scoringArgs: ScoringArgs,
) -> Tuple[List[shared_memory.SharedMemory], c.ScoringArgsSharedMemory]:
  """
  Save large dfs to shared memory. Called before beginning multiprocessing.
  """
  shms: List[shared_memory.SharedMemory] = []
  noteTopics = save_df_to_shared_memory(scoringArgs.noteTopics, shms)
  ratings = save_df_to_shared_memory(scoringArgs.ratings, shms)
  noteStatusHistory = save_df_to_shared_memory(scoringArgs.noteStatusHistory, shms)
  userEnrollment = save_df_to_shared_memory(scoringArgs.userEnrollment, shms)

  if type(scoringArgs) == FinalScoringArgs:
    prescoringNoteModelOutput = save_df_to_shared_memory(
      scoringArgs.prescoringNoteModelOutput, shms
    )
    prescoringRaterModelOutput = save_df_to_shared_memory(
      scoringArgs.prescoringRaterModelOutput, shms
    )
    return shms, c.FinalScoringArgsSharedMemory(
      noteTopics,
      ratings,
      noteStatusHistory,
      userEnrollment,
      prescoringNoteModelOutput,
      prescoringRaterModelOutput,
    )
  else:
    return shms, c.PrescoringArgsSharedMemory(
      noteTopics, ratings, noteStatusHistory, userEnrollment
    )


def _run_scorers(
  scorers: List[Scorer],
  scoringArgs: ScoringArgs,
  runParallel: bool = True,
  maxWorkers: Optional[int] = None,
  dataLoader: Optional[CommunityNotesDataLoader] = None,
) -> List[ModelResult]:
  """Applies all Community Notes models to user ratings and returns merged result.

  Each model must return a scoredNotes DF and may return helpfulnessScores and auxiliaryNoteInfo.
  scoredNotes and auxiliaryNoteInfo will be forced to contain one row per note to guarantee
  that all notes can be assigned a status during meta scoring regardless of whether any
  individual scoring algorithm scored the note.

  Args:
    scorers (List[Scorer]): Instantiated Scorer objects for note ranking.
    noteTopics: DF pairing notes with topics
    ratings (pd.DataFrame): Complete DF containing all ratings after preprocessing.
    noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
    userEnrollment (pd.DataFrame): The enrollment state for each contributor

  Returns:
    List[ModelResult]
  """
  # Apply scoring algorithms
  overallStartTime = time.perf_counter()
  if runParallel:
    shms, scoringArgsSharedMemory = _save_dfs_to_shared_memory(scoringArgs)

    with concurrent.futures.ProcessPoolExecutor(
      mp_context=multiprocessing.get_context("fork"),
      max_workers=maxWorkers,
    ) as executor:
      print(f"Starting parallel scorer execution with {len(scorers)} scorers.")
      # Pass mostly-empty scoringArgs: the data is too large to be copied in-memory to
      # each process, so must be re-loaded from disk by every scorer's dataLoader.
      scoringArgs.remove_large_args_for_multiprocessing()
      futures = [
        executor.submit(
          _run_scorer_parallelizable,
          scorer=scorer,
          runParallel=True,
          scoringArgs=copy.deepcopy(scoringArgs),
          dataLoader=dataLoader,
          scoringArgsSharedMemory=copy.deepcopy(scoringArgsSharedMemory),
        )
        for scorer in scorers
      ]
      modelResultsAndTimes = [f.result() for f in futures]

      for shm in shms:
        shm.close()
        shm.unlink()  # free the shared memory
  else:
    modelResultsAndTimes = [
      _run_scorer_parallelizable(
        scorer=scorer,
        runParallel=False,
        scoringArgs=scoringArgs,
      )
      for scorer in scorers
    ]

  modelResultsTuple, scorerTimesTuple = zip(*modelResultsAndTimes)

  overallTime = time.perf_counter() - overallStartTime
  print(
    f"""----
    Completed individual scorers. Ran in parallel: {runParallel}.  Succeeded in {overallTime:.2f} seconds. 
    Individual scorers: (name, runtime): {list(zip(
      [scorer.get_name() for scorer in scorers],
      ['{:.2f}'.format(t/60.0) + " mins" for t in scorerTimesTuple]
    ))}
    ----"""
  )
  return list(modelResultsTuple)


def combine_prescorer_scorer_results(
  modelResults: List[ModelResult],
) -> Tuple[pd.DataFrame, pd.DataFrame, c.PrescoringMetaOutput]:
  """
  Returns dfs with original columns plus an extra scorer name column.
  """
  assert isinstance(modelResults[0], ModelResult)

  prescoringNoteModelOutputList = []
  raterParamsUnfilteredMultiScorersList = []
  prescoringMetaOutput = c.PrescoringMetaOutput(metaScorerOutput={})

  for modelResult in modelResults:
    if modelResult.scoredNotes is not None:
      modelResult.scoredNotes[c.scorerNameKey] = modelResult.scorerName
      prescoringNoteModelOutputList.append(modelResult.scoredNotes)

    if modelResult.helpfulnessScores is not None:
      modelResult.helpfulnessScores[c.scorerNameKey] = modelResult.scorerName
      raterParamsUnfilteredMultiScorersList.append(modelResult.helpfulnessScores)

    if modelResult.metaScores is not None and modelResult.scorerName is not None:
      prescoringMetaOutput.metaScorerOutput[modelResult.scorerName] = modelResult.metaScores

  prescoringNoteModelOutput = pd.concat(
    prescoringNoteModelOutputList,
    unsafeAllowed={
      c.defaultIndexKey,
      c.noteIdKey,
      c.internalNoteInterceptKey,
      c.internalNoteFactor1Key,
      c.lowDiligenceNoteInterceptKey,
      c.lowDiligenceNoteFactor1Key,
    },
  )
  raterParamsUnfilteredMultiScorers = pd.concat(
    raterParamsUnfilteredMultiScorersList,
    unsafeAllowed={
      c.defaultIndexKey,
      c.internalRaterInterceptKey,
      c.internalRaterFactor1Key,
      c.crhCrnhRatioDifferenceKey,
      c.meanNoteScoreKey,
      c.raterAgreeRatioKey,
      c.aboveHelpfulnessThresholdKey,
      c.internalRaterReputationKey,
      c.lowDiligenceRaterInterceptKey,
      c.lowDiligenceRaterFactor1Key,
      c.lowDiligenceRaterReputationKey,
    },
  )
  return (
    prescoringNoteModelOutput[c.prescoringNoteModelOutputTSVColumns],
    raterParamsUnfilteredMultiScorers[c.prescoringRaterModelOutputTSVColumns],
    prescoringMetaOutput,
  )


def combine_final_scorer_results(
  modelResultsFromEachScorer: List[ModelResult],
  noteStatusHistory: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """
  Returns:
    Tuple[pd.DataFrame, pd.DataFrame]:
      scoredNotes pd.DataFrame: one row per note contained note scores and parameters.
      auxiliaryNoteInfo pd.DataFrame: one row per note containing supplemental values used in scoring.
  """
  # Initialize return data frames.
  scoredNotes = noteStatusHistory[[c.noteIdKey]].drop_duplicates()
  auxiliaryNoteInfo = noteStatusHistory[[c.noteIdKey]].drop_duplicates()

  # Merge the results
  for modelResult in modelResultsFromEachScorer:
    scoredNotes, auxiliaryNoteInfo = _merge_results(
      scoredNotes,
      auxiliaryNoteInfo,
      modelResult.scoredNotes,
      modelResult.auxiliaryNoteInfo,
    )
  scoredNotes = coalesce_group_model_scored_notes(scoredNotes)
  scoredNotes = coalesce_topic_models(scoredNotes)
  return scoredNotes, auxiliaryNoteInfo


def convert_prescoring_rater_model_output_to_coalesced_helpfulness_scores(
  prescoringRaterModelOutput: pd.DataFrame,
  userEnrollment: pd.DataFrame,
):
  # Join modeling groups from enrollment
  prescoringRaterModelOutput = prescoringRaterModelOutput.merge(
    userEnrollment[[c.participantIdKey, c.modelingGroupKey]],
    left_on=c.raterParticipantIdKey,
    right_on=c.participantIdKey,
    how="left",
  )

  helpfulnessScores = prescoringRaterModelOutput[
    [
      c.raterParticipantIdKey,
    ]
  ].drop_duplicates()

  scorersEnumDict = _get_scorers(seed=None, pseudoraters=None, enabledScorers=None)
  scorers = chain(*scorersEnumDict.values())
  uniqueScorerNames = prescoringRaterModelOutput[c.scorerNameKey].unique()
  for scorer in scorers:
    scorerName = scorer.get_name()
    if scorerName not in uniqueScorerNames:
      continue

    scorerOutputInternalNames = prescoringRaterModelOutput[
      (prescoringRaterModelOutput[c.scorerNameKey] == scorerName)
    ]
    scorerOutputExternalNames = scorerOutputInternalNames.rename(
      columns=scorer._get_user_col_mapping()
    )
    if isinstance(scorer, MFGroupScorer):
      scorerOutputExternalNames[scorer._modelingGroupKey] = scorer._groupNumber
      # Raters may appear in multiple groups due to authorship -- filter out rows not from this group
      scorerOutputExternalNames = scorerOutputExternalNames[
        scorerOutputExternalNames[c.modelingGroupKey] == scorer._groupNumber
      ]

    finalCols = scorer.get_helpfulness_scores_cols()
    if c.raterParticipantIdKey not in finalCols:
      finalCols.append(c.raterParticipantIdKey)
    scorerOutputExternalNames = scorerOutputExternalNames[finalCols]

    if isinstance(scorer, MFGroupScorer):
      helpfulnessScores = helpfulnessScores.merge(
        scorerOutputExternalNames,
        on=c.raterParticipantIdKey,
        how="outer",
        unsafeAllowed=scorer._modelingGroupKey,
      )
    else:
      helpfulnessScores = helpfulnessScores.merge(
        scorerOutputExternalNames, on=c.raterParticipantIdKey, how="outer"
      )

  return helpfulnessScores


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
  with c.time_block("Post-scorers: Meta Score: Setup"):
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
    if enabledScorers is None or Scorers.MFExpansionPlusScorer in enabledScorers:
      # The MFExpansionPlusScorer should score a disjoint set of notes from MFExpansionScorer
      # and MFCoreScorer because it should score notes by EXPANSION_PLUS writers and should be
      # the only model to score notes by EXPANSION_PLUS writers.  This ordering is safe, where if
      # there is any bug and a note is scored by MFExpansionPlusScorer and another scorer, then
      # MFExpansionPlusScorer will have the lowest priority.
      rules.append(
        scoring_rules.ApplyModelResult(
          RuleID.EXPANSION_PLUS_MODEL,
          {RuleID.META_INITIAL_NMR},
          c.expansionPlusRatingStatusKey,
        )
      )
    if enabledScorers is None or Scorers.MFExpansionScorer in enabledScorers:
      rules.append(
        scoring_rules.ApplyModelResult(
          RuleID.EXPANSION_MODEL,
          {RuleID.META_INITIAL_NMR},
          c.expansionRatingStatusKey,
        )
      )
    if enabledScorers is None or Scorers.MFCoreScorer in enabledScorers:
      rules.append(
        scoring_rules.ApplyModelResult(
          RuleID.CORE_MODEL,
          {RuleID.META_INITIAL_NMR},
          c.coreRatingStatusKey,
        )
      )
    if enabledScorers is None or Scorers.MFGroupScorer in enabledScorers:
      # TODO: modify this code to work when MFExpansionScorer is disabled by the system test
      assert len(scorers[Scorers.MFCoreScorer]) == 1
      assert len(scorers[Scorers.MFExpansionScorer]) == 1
      coreScorer = scorers[Scorers.MFCoreScorer][0]
      assert isinstance(coreScorer, MFCoreScorer)
      expansionScorer = scorers[Scorers.MFExpansionScorer][0]
      assert isinstance(expansionScorer, MFExpansionScorer)
      coreCrhThreshold = coreScorer.get_crh_threshold()
      expansionCrhThreshold = expansionScorer.get_crh_threshold()
      for i in range(1, groupScorerCount + 1):
        if i != trialScoringGroup:
          rules.append(
            scoring_rules.ApplyGroupModelResult(
              RuleID[f"GROUP_MODEL_{i}"],
              {RuleID.EXPANSION_MODEL, RuleID.CORE_MODEL},
              i,
              coreCrhThreshold,
              expansionCrhThreshold,
            )
          )
        else:
          rules.append(
            scoring_rules.ApplyGroupModelResult(
              RuleID[f"GROUP_MODEL_{i}"],
              {RuleID.EXPANSION_MODEL, RuleID.CORE_MODEL},
              i,
              None,
              None,
              minSafeguardThreshold=0.25,
            )
          )
    if enabledScorers is None or Scorers.MFTopicScorer in enabledScorers:
      for topic in Topics:
        if topic == Topics.Unassigned:
          continue
        rules.append(
          scoring_rules.ApplyTopicModelResult(
            RuleID[f"TOPIC_MODEL_{topic.value}"],
            {RuleID.EXPANSION_PLUS_MODEL, RuleID.EXPANSION_MODEL, RuleID.CORE_MODEL},
            topic,
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

  with c.time_block("Post-scorers: Meta Score: Apply Scoring Rules"):
    scoringResult = scoring_rules.apply_scoring_rules(
      scoredNotes,
      rules,
      c.finalRatingStatusKey,
      c.metaScorerActiveRulesKey,
      decidedByColumn=c.decidedByKey,
    )

  with c.time_block("Post-scorers: Meta Score: Preparing Return Values"):
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

  This function computes note aggregates over ratings and merges in selected fields
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
  scoredNotesCols = noteStats[
    [c.noteIdKey, c.classificationKey, c.createdAtMillisKey, c.numRatingsKey]
  ]
  auxiliaryNoteInfoCols = noteStats[
    [
      c.noteIdKey,
      c.noteAuthorParticipantIdKey,
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
  with c.time_block("Meta Helpfulness Scorers: Setup"):
    # Generate a unified view of note scoring information for computing contributor stats
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

  with c.time_block("Meta Helpfulness Scores: Contributor Scores"):
    # Return one row per rater with stats including trackrecord identifying note labels.
    contributorScores = contributor_state.get_contributor_scores(
      scoredNotesWithStats,
      ratings,
      noteStatusHistory,
    )
  with c.time_block("Meta Helpfulness Scorers: Contributor State"):
    contributorState, prevState = contributor_state.get_contributor_state(
      scoredNotesWithStats,
      ratings,
      noteStatusHistory,
      userEnrollment,
    )

  with c.time_block("Meta Helpfulness Scorers: Combining"):
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
          c.numberOfTimesEarnedOutKey,
          c.ratingImpact,
          c.hasCrnhSinceEarnOut,
        ]
      ],
      on=c.raterParticipantIdKey,
      how="outer",
      unsafeAllowed={c.enrollmentState, c.isEmergingWriterKey},
    )
    contributorScores = contributor_state.single_trigger_earn_out(contributorScores)
    contributorScores = contributor_state.calculate_ri_to_earn_in(contributorScores)

    # Consolidates all information on raters / authors.
    helpfulnessScores = helpfulnessScores.merge(
      contributorScores,
      on=c.raterParticipantIdKey,
      how="outer",
    )
    # Pass timestampOfLastEarnOut through to raterModelOutput.
    helpfulnessScores = helpfulnessScores.merge(
      prevState,
      left_on=c.raterParticipantIdKey,
      right_on=c.participantIdKey,
      how="left",
      unsafeAllowed=(c.enrollmentState + "_prev"),
    ).drop(c.participantIdKey, axis=1)

    # For users who did not earn a new enrollmentState, carry over the previous one
    helpfulnessScores[c.enrollmentState] = helpfulnessScores[c.enrollmentState].fillna(
      helpfulnessScores[c.enrollmentState + "_prev"]
    )
    helpfulnessScores.drop(columns=[c.enrollmentState + "_prev"], inplace=True)

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
    elif columnType == str:
      scoredNotes[column] = ""
    else:
      assert False, f"column type {columnType} unsupported"
  return scoredNotes


def _validate_note_scoring_output(
  scoredNotes: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  auxiliaryNoteInfo: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Guarantee that each dataframe has the expected columns in the correct order.

  Args:
    scoredNotes (pd.DataFrame): notes with scores returned by MF scoring algorithm
    noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
    auxiliaryNoteInfo (pd.DataFrame): additional fields generated during note scoring

  Returns:
    Input arguments with columns potentially re-ordered.
  """
  assert set(scoredNotes.columns) == set(
    c.noteModelOutputTSVColumns
  ), f"Got {sorted(scoredNotes.columns)}, expected {sorted(c.noteModelOutputTSVColumns)}"
  scoredNotes = scoredNotes[c.noteModelOutputTSVColumns]
  assert set(noteStatusHistory.columns) == set(
    c.noteStatusHistoryTSVColumns
  ), f"Got {sorted(noteStatusHistory.columns)}, expected {sorted(c.noteStatusHistoryTSVColumns)}"
  noteStatusHistory = noteStatusHistory[c.noteStatusHistoryTSVColumns]
  assert set(auxiliaryNoteInfo.columns) == set(
    c.auxiliaryScoredNotesTSVColumns
  ), f"Got {sorted(auxiliaryNoteInfo.columns)}, expected {sorted(c.auxiliaryScoredNotesTSVColumns)}"
  auxiliaryNoteInfo = auxiliaryNoteInfo[c.auxiliaryScoredNotesTSVColumns]
  return (scoredNotes, noteStatusHistory, auxiliaryNoteInfo)


def _validate_contributor_scoring_output(helpfulnessScores: pd.DataFrame) -> pd.DataFrame:
  assert set(helpfulnessScores.columns) == set(
    c.raterModelOutputTSVColumns
  ), f"Got {sorted(helpfulnessScores.columns)}, expected {sorted(c.raterModelOutputTSVColumns)}"
  helpfulnessScores = helpfulnessScores[c.raterModelOutputTSVColumns]
  return helpfulnessScores


def _log_df_info(
  notes: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  userEnrollment: pd.DataFrame,
  prescoringNoteModelOutput: Optional[pd.DataFrame] = None,
  prescoringRaterModelOutput: Optional[pd.DataFrame] = None,
) -> None:
  """Log dtype and RAM usage stats for each input DataFrame."""
  pairs = [
    ("notes", notes),
    ("ratings", ratings),
    ("noteStatusHistory", noteStatusHistory),
    ("userEnrollment", userEnrollment),
    ("prescoringNoteModelOutput", prescoringNoteModelOutput),
    ("prescoringRaterModelOutput", prescoringRaterModelOutput),
  ]
  for name, df in pairs:
    if df is None:
      continue
    stats = (
      df.dtypes.to_frame().reset_index(drop=False).rename(columns={"index": "column", 0: "dtype"})
    ).merge(
      # deep=True shows memory usage for the entire contained object (e.g. if the type
      # of a column is "object", then deep=True shows the size of the objects instead
      # of the size of the pointers.
      df.memory_usage(index=True, deep=True)
      .to_frame()
      .reset_index(drop=False)
      .rename(columns={"index": "column", 0: "RAM"})
    )
    print(f"""{name} total RAM: {stats["RAM"].sum()}""")
    print(stats)
    print()


def run_prescoring(
  notes: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  userEnrollment: pd.DataFrame,
  seed: Optional[int] = None,
  enabledScorers: Optional[Set[Scorers]] = None,
  runParallel: bool = True,
  dataLoader: Optional[CommunityNotesDataLoader] = None,
  useStableInitialization: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, sklearn.pipeline.Pipeline, c.PrescoringMetaOutput]:
  with c.time_block("Logging RAM usage"):
    _log_df_info(notes, ratings, noteStatusHistory, userEnrollment)
  with c.time_block("Note Topic Assignment"):
    topicModel = TopicModel()
    noteTopicClassifierPipe, seedLabels, conflictedTexts = topicModel.train_note_topic_classifier(
      notes
    )
    noteTopics = topicModel.get_note_topics(
      notes, noteTopicClassifierPipe, seedLabels, conflictedTextsForAccuracyEval=conflictedTexts
    )

  with c.time_block("Compute Post Selection Similarity"):
    pss = PostSelectionSimilarity(notes, ratings)
    postSelectionSimilarityValues = pss.get_post_selection_similarity_values()
    print(f"Post Selection Similarity Prescoring: begin with {len(ratings)} ratings.")
    ratings = filter_ratings_by_post_selection_similarity(
      notes, ratings, postSelectionSimilarityValues
    )
    print(f"Post Selection Similarity Prescoring: {len(ratings)} ratings remaining.")
    del pss

  scorers = _get_scorers(
    seed=seed,
    pseudoraters=False,
    enabledScorers=enabledScorers,
    useStableInitialization=useStableInitialization,
  )

  prescoringModelResultsFromAllScorers = _run_scorers(
    scorers=list(chain(*scorers.values())),
    scoringArgs=PrescoringArgs(
      noteTopics=noteTopics,
      ratings=ratings,
      noteStatusHistory=noteStatusHistory,
      userEnrollment=userEnrollment,
    ),
    runParallel=runParallel,
    dataLoader=dataLoader,
    # Restrict parallelism to 6 processes.  Memory usage scales linearly with the number of
    # processes and 6 is enough that the limiting factor continues to be the longest running
    # scorer (i.e. we would not finish faster with >6 worker processes.)
    maxWorkers=6,
  )

  (
    prescoringNoteModelOutput,
    prescoringRaterModelOutput,
    prescoringMetaOutput,
  ) = combine_prescorer_scorer_results(prescoringModelResultsFromAllScorers)
  prescoringRaterModelOutput = pd.concat(
    [prescoringRaterModelOutput, postSelectionSimilarityValues],
    unsafeAllowed={
      c.postSelectionValueKey,
    },
  )

  return (
    prescoringNoteModelOutput,
    prescoringRaterModelOutput,
    noteTopicClassifierPipe,
    prescoringMetaOutput,
  )


def run_contributor_scoring(
  ratings: pd.DataFrame,
  scoredNotes: pd.DataFrame,
  auxiliaryNoteInfo: pd.DataFrame,
  prescoringRaterModelOutput: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  userEnrollment: pd.DataFrame,
  strictColumns: bool = True,
) -> pd.DataFrame:
  helpfulnessScores = convert_prescoring_rater_model_output_to_coalesced_helpfulness_scores(
    prescoringRaterModelOutput, userEnrollment
  )
  helpfulnessScores = coalesce_group_model_helpfulness_scores(helpfulnessScores)

  # Compute contribution statistics and enrollment state for users.
  with c.time_block("Post-scorers: Compute helpfulness scores"):
    helpfulnessScores = _compute_helpfulness_scores(
      ratings,
      scoredNotes,
      auxiliaryNoteInfo,
      helpfulnessScores,
      noteStatusHistory,
      userEnrollment,
    )
    if strictColumns:
      helpfulnessScores = _validate_contributor_scoring_output(helpfulnessScores)
  return helpfulnessScores


def determine_which_notes_to_rescore(
  notes: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  previousRatingCutoffTimestampMillis: Optional[int] = None,
  scoreRecentNotesMinimumFrequencyMillis: Optional[int] = 1000 * 60 * 60 * 24,  # 1 day
  recentNotesAgeCutoffMillis: Optional[int] = 1000 * 60 * 60 * 24 * 14,  # 14 days
) -> Tuple[Optional[List[c.NoteSubset]], set]:
  # 1. Rescore all notes with a new rating since last scoring run.
  if previousRatingCutoffTimestampMillis is not None:
    notesWithNewRatings = set(
      ratings.loc[ratings[c.createdAtMillisKey] > previousRatingCutoffTimestampMillis, c.noteIdKey]
    )
  else:
    notesWithNewRatings = set()

  currentMillis = int(time.time() * 1000)

  # 2. Rescore all recently created notes if not rescored at the minimum frequency.
  if recentNotesAgeCutoffMillis is not None and scoreRecentNotesMinimumFrequencyMillis is not None:
    noteCreatedRecently = (
      noteStatusHistory[c.createdAtMillisKey] > currentMillis - recentNotesAgeCutoffMillis
    )
    noteNotRescoredRecently = (
      noteStatusHistory[c.timestampMillisOfNoteCurrentLabelKey]
      < currentMillis - scoreRecentNotesMinimumFrequencyMillis
    )
    newNotesNotRescoredRecentlyEnough = set(
      noteStatusHistory.loc[noteCreatedRecently & noteNotRescoredRecently, c.noteIdKey]
    )
    # Remove notes with new ratings from this set.
    newNotesNotRescoredRecentlyEnough = newNotesNotRescoredRecentlyEnough.difference(
      notesWithNewRatings
    )
  else:
    newNotesNotRescoredRecentlyEnough = set()

  # TODO: 3. Recently-flipped notes.

  noteSubsets = [
    c.NoteSubset(
      noteSet=notesWithNewRatings,
      maxCrhChurnRate=c.finalNotesWithNewRatingsMaxCrhChurn,
      description="notesWithNewRatings",
    ),
    c.NoteSubset(
      noteSet=newNotesNotRescoredRecentlyEnough,
      maxCrhChurnRate=c.finalUnlockedNotesWithNoNewRatingsMaxCrhChurn,
      description="newNotesNotRescoredRecentlyEnough",
    ),
  ]

  notesToRescoreSet = set()
  for noteSubset in noteSubsets:
    if noteSubset.noteSet is not None:
      notesToRescoreSet.update(noteSubset.noteSet)

  print(
    f"""Notes to rescore:
        {len(notesWithNewRatings)} notes with new ratings since last scoring run.
        {len(newNotesNotRescoredRecentlyEnough)} notes created recently and not rescored recently enough.
        Total: {len(notesToRescoreSet)} notes to rescore, out of {len(notes)} total."""
  )

  return noteSubsets, notesToRescoreSet


def run_final_note_scoring(
  notes: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  userEnrollment: pd.DataFrame,
  prescoringNoteModelOutput: pd.DataFrame,
  prescoringRaterModelOutput: pd.DataFrame,
  noteTopicClassifier: sklearn.pipeline.Pipeline,
  prescoringMetaOutput: c.PrescoringMetaOutput,
  seed: Optional[int] = None,
  pseudoraters: Optional[bool] = True,
  enabledScorers: Optional[Set[Scorers]] = None,
  strictColumns: bool = True,
  runParallel: bool = True,
  dataLoader: Optional[CommunityNotesDataLoader] = None,
  useStableInitialization: bool = True,
  checkFlips: bool = True,
  previousScoredNotes: Optional[pd.DataFrame] = None,
  previousAuxiliaryNoteInfo: Optional[pd.DataFrame] = None,
  previousRatingCutoffTimestampMillis: Optional[int] = 0,
):
  with c.time_block("Logging RAM usage"):
    _log_df_info(
      notes,
      ratings,
      noteStatusHistory,
      userEnrollment,
      prescoringNoteModelOutput,
      prescoringRaterModelOutput,
    )

  with c.time_block("Determine which notes to score."):
    if previousScoredNotes is None:
      print("No previous scored notes passed; scoring all notes.")
      notesToRescoreSet: Set[int] = set()
      scoredNotesPassthrough = None
      noteSubsets: Optional[List[c.NoteSubset]] = [
        c.NoteSubset(
          noteSet=None,
          maxCrhChurnRate=c.prescoringAllUnlockedNotesMaxCrhChurn,
          description="allNotes",
        )
      ]
    else:
      assert previousAuxiliaryNoteInfo is not None
      assert previousRatingCutoffTimestampMillis is not None
      print("Previous scored notes passed; determining which notes to rescore.")
      # Filter all datasets to smaller versions which only contain notes which need to be scored.
      noteSubsets, notesToRescoreSet = determine_which_notes_to_rescore(
        notes, ratings, noteStatusHistory, previousRatingCutoffTimestampMillis
      )

      for item in notesToRescoreSet:
        print(f"notesToRescoreSet: {item}, {type(item)}, len: {len(notesToRescoreSet)}")
        break
      scoredNotesPassthrough = previousScoredNotes[
        ~previousScoredNotes[c.noteIdKey].isin(notesToRescoreSet)
      ]
      auxiliaryNoteInfoPassthrough = previousAuxiliaryNoteInfo[
        ~previousAuxiliaryNoteInfo[c.noteIdKey].isin(notesToRescoreSet)
      ]
      noteStatusHistoryPassthrough = noteStatusHistory[
        ~noteStatusHistory[c.noteIdKey].isin(notesToRescoreSet)
      ]

      print(
        f"Rescoring {len(notesToRescoreSet)} notes, out of {len(notes)} total. Original number of ratings: {len(ratings)}"
      )

      # Filter all datasets to only contain notes which need to be scored.
      notes = notes[notes[c.noteIdKey].isin(notesToRescoreSet)]
      ratings = ratings[ratings[c.noteIdKey].isin(notesToRescoreSet)]
      noteStatusHistory = noteStatusHistory[noteStatusHistory[c.noteIdKey].isin(notesToRescoreSet)]
      prescoringNoteModelOutput = prescoringNoteModelOutput[
        prescoringNoteModelOutput[c.noteIdKey].isin(notesToRescoreSet)
      ]

      print(f"Ratings on notes to rescore: {len(ratings)}")

  with c.time_block("Preprocess smaller dataset since we skipped preprocessing at read time"):
    notes, ratings, noteStatusHistory = preprocess_data(notes, ratings, noteStatusHistory)

  with c.time_block("Note Topic Assignment"):
    topicModel = TopicModel()
    noteTopics = topicModel.get_note_topics(notes, noteTopicClassifier)

  with c.time_block("Post Selection Similarity: Final Scoring"):
    print(f"Post Selection Similarity Final Scoring: begin with {len(ratings)} ratings.")
    ratings = filter_ratings_by_post_selection_similarity(
      notes,
      ratings,
      prescoringRaterModelOutput[prescoringRaterModelOutput[c.postSelectionValueKey] >= 1][
        [c.raterParticipantIdKey, c.postSelectionValueKey]
      ],
    )
    print(f"Post Selection Similarity Final Scoring: {len(ratings)} ratings remaining.")

  scorers = _get_scorers(
    seed, pseudoraters, enabledScorers, useStableInitialization=useStableInitialization
  )

  modelResults = _run_scorers(
    scorers=list(chain(*scorers.values())),
    scoringArgs=FinalScoringArgs(
      noteTopics,
      ratings,
      noteStatusHistory,
      userEnrollment,
      prescoringNoteModelOutput=prescoringNoteModelOutput,
      prescoringRaterModelOutput=prescoringRaterModelOutput,
      prescoringMetaOutput=prescoringMetaOutput,
    ),
    runParallel=runParallel,
    dataLoader=dataLoader,
    # Restrict parallelism to 6 processes.  Memory usage scales linearly with the number of
    # processes and 6 is enough that the limiting factor continues to be the longest running
    # scorer (i.e. we would not finish faster with >6 worker processes.)
    maxWorkers=6,
  )

  scoredNotes, auxiliaryNoteInfo = combine_final_scorer_results(modelResults, noteStatusHistory)

  if not checkFlips:
    noteSubsets = None

  scoredNotes, newNoteStatusHistory, auxiliaryNoteInfo = post_note_scoring(
    scorers,
    scoredNotes,
    auxiliaryNoteInfo,
    ratings,
    noteStatusHistory,
    enabledScorers,
    strictColumns,
    noteSubsets,
  )

  # Concat final scoring results for newly-scored notes with the results for old notes not scores.
  if scoredNotesPassthrough is not None:
    # Convert scoredNotes dtypes to match scoredNotesPassthrough
    for column, targetDtype in c.noteModelOutputTSVTypeMapping.items():
      if column in scoredNotes.columns:
        if targetDtype == pd.BooleanDtype():
          # Due to current Python version in prod, we cannot interpret pd.BooleanDtype() as a datatype yet.
          continue
        if scoredNotes[column].dtype != targetDtype:
          scoredNotes[column] = scoredNotes[column].astype(targetDtype)
    scoredNotes = pd.concat(
      [scoredNotes, scoredNotesPassthrough],
    )

    # Convert auxiliaryNoteInfo dtypes to match auxiliaryNoteInfoPassthrough
    for column, targetDtype in c.auxiliaryScoredNotesTSVTypeMapping.items():
      if column in auxiliaryNoteInfo.columns:
        if auxiliaryNoteInfo[column].dtype != targetDtype:
          auxiliaryNoteInfo[column] = auxiliaryNoteInfo[column].astype(targetDtype)
    auxiliaryNoteInfo = pd.concat(
      [auxiliaryNoteInfo, auxiliaryNoteInfoPassthrough],
    )

    newNoteStatusHistory = pd.concat([newNoteStatusHistory, noteStatusHistoryPassthrough])

  return scoredNotes, newNoteStatusHistory, auxiliaryNoteInfo


def post_note_scoring(
  scorers: Dict[Scorers, List[Scorer]],
  scoredNotes: pd.DataFrame,
  auxiliaryNoteInfo: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  enabledScorers: Optional[Set[Scorers]] = None,
  strictColumns: bool = True,
  noteSubsetsAndMaxFlipRates: Optional[List[c.NoteSubset]] = None,
):
  """
  Apply individual scoring models and obtained merged result.
  """
  postScoringStartTime = time.time()
  # Augment scoredNotes and auxiliaryNoteInfo with additional attributes for each note
  # which are computed over the corpus of notes / ratings as a whole and are independent
  # of any particular model.

  with c.time_block("Post-scorers: Compute note stats"):
    scoredNotesCols, auxiliaryNoteInfoCols = _compute_note_stats(ratings, noteStatusHistory)
    scoredNotes = scoredNotes.merge(scoredNotesCols, on=c.noteIdKey)
    auxiliaryNoteInfo = auxiliaryNoteInfo.merge(auxiliaryNoteInfoCols, on=c.noteIdKey)

  # Assign final status to notes based on individual model scores and note attributes.
  with c.time_block("Post-scorers: Meta score"):
    scoredNotesCols, auxiliaryNoteInfoCols = meta_score(
      scorers,
      scoredNotes,
      auxiliaryNoteInfo,
      noteStatusHistory[[c.noteIdKey, c.lockedStatusKey]],
      enabledScorers,
    )

  with c.time_block("Post-scorers: Join scored notes"):
    scoredNotes = scoredNotes.merge(scoredNotesCols, on=c.noteIdKey)
    scoredNotes[c.timestampMillisOfNoteCurrentLabelKey] = c.epochMillis
    auxiliaryNoteInfo = auxiliaryNoteInfo.merge(auxiliaryNoteInfoCols, on=c.noteIdKey)

    # Validate that no notes were dropped or duplicated.
    assert len(scoredNotes) == len(
      noteStatusHistory
    ), "noteStatusHistory should be complete, and all notes should be scored."
    assert len(auxiliaryNoteInfo) == len(
      noteStatusHistory
    ), "noteStatusHistory should be complete, and all notes should be scored."

  # Merge scoring results into noteStatusHistory.
  with c.time_block("Post-scorers: Update note status history"):
    mergedNoteStatuses = note_status_history.merge_old_and_new_note_statuses(
      noteStatusHistory, scoredNotes
    )
    if noteSubsetsAndMaxFlipRates is not None:
      for noteSubset in noteSubsetsAndMaxFlipRates:
        note_status_history.check_flips(mergedNoteStatuses, noteSubset=noteSubset)

    newNoteStatusHistory = note_status_history.update_note_status_history(mergedNoteStatuses)
    assert len(newNoteStatusHistory) == len(
      noteStatusHistory
    ), "noteStatusHistory should contain all notes after preprocessing"

  # Skip validation and selection out output columns if the set of scorers is overridden.
  with c.time_block("Post-scorers: finalize output columns"):
    scoredNotes = _add_deprecated_columns(scoredNotes)
    if strictColumns:
      (scoredNotes, newNoteStatusHistory, auxiliaryNoteInfo) = _validate_note_scoring_output(
        scoredNotes, newNoteStatusHistory, auxiliaryNoteInfo
      )

  print(f"Meta scoring elapsed time: {((time.time() - postScoringStartTime)/60.0):.2f} minutes.")
  return scoredNotes, newNoteStatusHistory, auxiliaryNoteInfo


def run_scoring(
  notes: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  userEnrollment: pd.DataFrame,
  seed: Optional[int] = None,
  pseudoraters: Optional[bool] = True,
  enabledScorers: Optional[Set[Scorers]] = None,
  strictColumns: bool = True,
  runParallel: bool = True,
  dataLoader: Optional[CommunityNotesDataLoader] = None,
  useStableInitialization: bool = True,
  writePrescoringScoringOutputCallback: Optional[
    Callable[[pd.DataFrame, pd.DataFrame, sklearn.pipeline.Pipeline, c.PrescoringMetaOutput], None]
  ] = None,
  cutoffTimestampMillis: Optional[int] = None,
  excludeRatingsAfterANoteGotFirstStatusPlusNHours: Optional[int] = None,
  daysInPastToApplyPostFirstStatusFiltering: Optional[int] = 14,
  filterPrescoringInputToSimulateDelayInHours: Optional[int] = None,
  checkFlips: bool = True,
  previousScoredNotes: Optional[pd.DataFrame] = None,
  previousAuxiliaryNoteInfo: Optional[pd.DataFrame] = None,
  previousRatingCutoffTimestampMillis: Optional[int] = 0,
):
  """Runs both phases of scoring consecutively. Only for adhoc/testing use.
  In prod, we run each phase as a separate binary.

  Wrapper around run_prescoring, run_final_note_scoring, and run_contributor_scoring.

  Invokes note scoring algorithms, merges results and computes user stats.

  Args:
    ratings (pd.DataFrame): preprocessed ratings
    noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
    userEnrollment (pd.DataFrame): The enrollment state for each contributor
    seed (int, optional): if not None, base distinct seeds for the first and second MF rounds on this value
    pseudoraters (bool, optional): if True, compute optional pseudorater confidence intervals
    enabledScorers (Set[Scorers], optional): Scorers which should be instantiated
    strictColumns (bool, optional): if True, validate which columns are present
    runParallel (bool, optional): if True, run algorithms in parallel
    dataLoader (CommunityNotesDataLoader, optional): dataLoader provided to parallel execution
    useStableInitialization
    writePrescoringScoringOutputCallback
    filterPrescoringInputToSimulateDelayInHours

  Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
      scoredNotes pd.DataFrame: one row per note contained note scores and parameters.
      helpfulnessScores pd.DataFrame: one row per user containing a column for each helpfulness score.
      noteStatusHistory pd.DataFrame: one row per note containing when they got their most recent statuses.
      auxiliaryNoteInfo: one row per note containing adjusted and ratio tag values
  """
  # Filter input data for testing if optional args present. Else, do nothing.
  (
    notes,
    ratings,
    prescoringNotesInput,
    prescoringRatingsInput,
  ) = filter_input_data_for_testing(
    notes,
    ratings,
    noteStatusHistory,
    cutoffTimestampMillis,
    excludeRatingsAfterANoteGotFirstStatusPlusNHours,
    daysInPastToApplyPostFirstStatusFiltering,
    filterPrescoringInputToSimulateDelayInHours,
  )

  (
    prescoringNoteModelOutput,
    prescoringRaterModelOutput,
    prescoringNoteTopicClassifier,
    prescoringMetaOutput,
  ) = run_prescoring(
    notes=prescoringNotesInput,
    ratings=prescoringRatingsInput,
    noteStatusHistory=noteStatusHistory,
    userEnrollment=userEnrollment,
    seed=seed,
    enabledScorers=enabledScorers,
    runParallel=runParallel,
    dataLoader=dataLoader,
    useStableInitialization=useStableInitialization,
  )

  print("We invoked run_scoring and are now in between prescoring and scoring.")
  if writePrescoringScoringOutputCallback is not None:
    with c.time_block("Writing prescoring output."):
      writePrescoringScoringOutputCallback(
        prescoringNoteModelOutput,
        prescoringRaterModelOutput,
        prescoringNoteTopicClassifier,
        prescoringMetaOutput,
      )
  print("Starting final scoring")

  scoredNotes, newNoteStatusHistory, auxiliaryNoteInfo = run_final_note_scoring(
    notes=notes,
    ratings=ratings,
    noteStatusHistory=noteStatusHistory,
    userEnrollment=userEnrollment,
    seed=seed,
    pseudoraters=pseudoraters,
    enabledScorers=enabledScorers,
    strictColumns=strictColumns,
    runParallel=runParallel,
    dataLoader=dataLoader,
    useStableInitialization=useStableInitialization,
    prescoringNoteModelOutput=prescoringNoteModelOutput,
    prescoringRaterModelOutput=prescoringRaterModelOutput,
    noteTopicClassifier=prescoringNoteTopicClassifier,
    prescoringMetaOutput=prescoringMetaOutput,
    checkFlips=checkFlips,
    previousScoredNotes=previousScoredNotes,
    previousAuxiliaryNoteInfo=previousAuxiliaryNoteInfo,
    previousRatingCutoffTimestampMillis=previousRatingCutoffTimestampMillis,
  )

  print("Starting contributor scoring")

  helpfulnessScores = run_contributor_scoring(
    ratings=ratings,
    scoredNotes=scoredNotes,
    auxiliaryNoteInfo=auxiliaryNoteInfo,
    prescoringRaterModelOutput=prescoringRaterModelOutput,
    noteStatusHistory=newNoteStatusHistory,
    userEnrollment=userEnrollment,
    strictColumns=strictColumns,
  )

  return scoredNotes, helpfulnessScores, newNoteStatusHistory, auxiliaryNoteInfo
