from typing import Dict, List, Optional, Tuple

from . import constants as c
from .mf_base_scorer import MFBaseScorer

import numpy as np
import pandas as pd


# Number of MFGroupScorer objects we expect to instantiate
groupScorerCount = 14

# Group ID assigned to trial scoring algorithm
trialScoringGroup = 14

# Mapping of how many threads to assign to each group scorer
_groupScorerParalleism = {
  # Group model 13 is larger and benefits from more threads.
  # Others can default to 4.
  13: 8
}


def _coalesce_columns(df: pd.DataFrame, columnPrefix: str) -> pd.DataFrame:
  """Condense all columns beginning with columnPrefix into a single column.

  With each row there must be at most one column with a non-NaN value in the set of
  columns beginning with columnPrefix.  If a non-NaN value is present that will
  become the value in the condensed column, otherwise the value will be NaN.  After
  column values are condensed the original (prefixed) columns will be dropped.

  Args:
    df: DataFrame containing columns to condense
    collumnPrefix: Prefix used to detect columns to coalesce, and the name for
      the output column.

  Returns:
    DataFrame with all columns prefixed by columnPrefix dropped and replaced by
    a single column named columnPrefix

  Raises:
    AssertionError if multiple columns prefixed by columnPrefix have non-NaN values
    for any row.
  """
  # Identify columns to coalesce
  columns = [col for col in df.columns if col.startswith(f"{columnPrefix}_")]
  if not columns:
    return df
  # Validate that at most one column is set, and store which rows have a column set
  rowResults = np.invert(df[columns].isna()).sum(axis=1)
  assert all(rowResults <= 1), "each row should only be in one modeling group"

  # Coalesce results
  def _get_value(row):
    idx = row.first_valid_index()
    return row[idx] if idx is not None else np.nan

  coalesced = df[columns].apply(_get_value, axis=1)
  # Drop old columns and replace with new
  df = df.drop(columns=columns)
  df[columnPrefix] = coalesced
  return df


def coalesce_group_models(
  scoredNotes: pd.DataFrame, helpfulnessScores: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Coalesce all group modeling columns across note and user scoring.

  Since each Scorer must have distinct output columns, we use coalescing to run
  multiple instances of MFGroupScorer objects and then condense the results into
  a single set of columns.  This approach works because each note will be scored
  by at most one MFGroupScorer instance.

  Args:
    scoredNotes: scoring output for notes.
    helpfulnessScores: scoring output for users.

  Returns:
    tuple containing coalesced scoring results for notes and users.
  """
  for col in [
    c.groupNoteInterceptKey,
    c.groupNoteFactor1Key,
    c.groupRatingStatusKey,
    c.groupNoteInterceptMaxKey,
    c.groupNoteInterceptMinKey,
    c.modelingGroupKey,
  ]:
    scoredNotes = _coalesce_columns(scoredNotes, col)

  for col in [c.groupRaterInterceptKey, c.groupRaterFactor1Key, c.modelingGroupKey]:
    helpfulnessScores = _coalesce_columns(helpfulnessScores, col)

  return scoredNotes, helpfulnessScores


class MFGroupScorer(MFBaseScorer):
  def __init__(
    self,
    groupNumber: int,
    seed: Optional[int] = None,
    pseudoraters: Optional[bool] = False,
    groupThreshold: float = 0.8,
    saveIntermediateState: bool = False,
    userFactorLambda=None,
    noteFactorLambda=None,
    userInterceptLambda=None,
    noteInterceptLambda=None,
    globalInterceptLambda=None,
    diamondLambda=None,
    normalizedLossHyperparameters=None,
    maxFirstMFTrainError: float = 0.16,
    maxFinalMFTrainError: float = 0.09,
    requireInternalAuthor: bool = True,
    minMeanNoteScore: float = 0.05,
    crhThreshold: float = 0.40,
    crnhThresholdIntercept: float = -0.05,
    crnhThresholdNoteFactorMultiplier: float = -0.8,
    crnhThresholdNMIntercept: float = -0.15,
    crhSuperThreshold: float = 0.5,
    lowDiligenceThreshold: float = 0.217,
    factorThreshold: float = 0.5,
  ) -> None:
    """Configure MFGroupScorer object.

    Notice that each MFGroupScorer defines column names by appending the groupNumber to
    column prefixes which are constant.  Dynamically defining the column names allows the
    group scorer to be instantiated multiple times while maintaining the property that
    the columns attached by each scorer remain unique.  Once all scorers have ran, we
    (will) validate that each note was scored by at most one group scorer and then coalesce
    all of the group scoring columns and remove the groupNumber suffix.

    Args:
      groupNumber: int indicating which group this scorer instance should filter for.
      seed: if not None, seed value to ensure deterministic execution
      pseudoraters: if True, compute optional pseudorater confidence intervals
      groupThreshold: float indicating what fraction of ratings must be from within a group
        for the model to be active
    """
    super().__init__(
      seed,
      pseudoraters,
      useStableInitialization=False,
      saveIntermediateState=saveIntermediateState,
      threads=_groupScorerParalleism.get(groupNumber, 4),
      userFactorLambda=userFactorLambda,
      noteFactorLambda=noteFactorLambda,
      userInterceptLambda=userInterceptLambda,
      noteInterceptLambda=noteInterceptLambda,
      globalInterceptLambda=globalInterceptLambda,
      diamondLambda=diamondLambda,
      normalizedLossHyperparameters=normalizedLossHyperparameters,
      maxFirstMFTrainError=maxFirstMFTrainError,
      maxFinalMFTrainError=maxFinalMFTrainError,
      minMeanNoteScore=minMeanNoteScore,
      crhThreshold=crhThreshold,
      crnhThresholdIntercept=crnhThresholdIntercept,
      crnhThresholdNoteFactorMultiplier=crnhThresholdNoteFactorMultiplier,
      crnhThresholdNMIntercept=crnhThresholdNMIntercept,
      crhSuperThreshold=crhSuperThreshold,
      lowDiligenceThreshold=lowDiligenceThreshold,
      factorThreshold=factorThreshold,
    )
    assert groupNumber > 0, "groupNumber must be positive.  0 is reserved for unassigned."
    assert groupNumber <= groupScorerCount, "groupNumber exceeds maximum expected groups."
    self._groupNumber = groupNumber
    self._groupThreshold = groupThreshold
    self._groupNoteInterceptKey = f"{c.groupNoteInterceptKey}_{self._groupNumber}"
    self._groupNoteFactor1Key = f"{c.groupNoteFactor1Key}_{self._groupNumber}"
    self._groupRatingStatusKey = f"{c.groupRatingStatusKey}_{self._groupNumber}"
    self._groupNoteInterceptMaxKey = f"{c.groupNoteInterceptMaxKey}_{self._groupNumber}"
    self._groupNoteInterceptMinKey = f"{c.groupNoteInterceptMinKey}_{self._groupNumber}"
    self._groupRaterInterceptKey = f"{c.groupRaterInterceptKey}_{self._groupNumber}"
    self._groupRaterFactor1Key = f"{c.groupRaterFactor1Key}_{self._groupNumber}"
    self._modelingGroupKey = f"{c.modelingGroupKey}_{self._groupNumber}"
    self._requireInternalAuthor = requireInternalAuthor

  def get_name(self):
    return f"MFGroupScorer_{self._groupNumber}"

  def _get_note_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default note column names to custom names for a specific model."""
    return {
      c.internalNoteInterceptKey: self._groupNoteInterceptKey,
      c.internalNoteFactor1Key: self._groupNoteFactor1Key,
      c.internalRatingStatusKey: self._groupRatingStatusKey,
      c.noteInterceptMinKey: self._groupNoteInterceptMinKey,
      c.noteInterceptMaxKey: self._groupNoteInterceptMaxKey,
    }

  def _get_user_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default user column names to custom names for a specific model."""
    return {
      c.internalRaterInterceptKey: self._groupRaterInterceptKey,
      c.internalRaterFactor1Key: self._groupRaterFactor1Key,
    }

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      self._groupNoteInterceptKey,
      self._groupNoteFactor1Key,
      self._groupRatingStatusKey,
      self._groupNoteInterceptMaxKey,
      self._groupNoteInterceptMinKey,
      self._modelingGroupKey,
    ]

  def get_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the helpfulnessScores output."""
    return [
      c.raterParticipantIdKey,
      self._groupRaterInterceptKey,
      self._groupRaterFactor1Key,
      self._modelingGroupKey,
    ]

  def get_auxiliary_note_info_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the auxiliaryNoteInfo output."""
    return []

  def _get_dropped_note_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from scoredNotes and auxiliaryNoteInfo."""
    return super()._get_dropped_note_cols() + (
      [
        c.internalActiveRulesKey,
        c.activeFilterTagsKey,
        c.ratingWeightKey,
      ]
      + c.notHelpfulTagsAdjustedColumns
      + c.notHelpfulTagsAdjustedRatioColumns
      + c.incorrectFilterColumns
      + c.noteParameterUncertaintyTSVAuxColumns
    )

  def _get_dropped_user_cols(self) -> List[str]:
    """Returns a list of columns which should be excluded from helpfulnessScores output."""
    return super()._get_dropped_user_cols() + [
      c.crhCrnhRatioDifferenceKey,
      c.meanNoteScoreKey,
      c.raterAgreeRatioKey,
      c.aboveHelpfulnessThresholdKey,
    ]

  def _filter_input(
    self, ratings: pd.DataFrame, noteStatusHistory: pd.DataFrame, userEnrollment: pd.DataFrame
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prune the contents of ratings to only include ratings from users in the modeling group.

    This function identifies the subset of ratings to include in group model scoring.
    To improve modeling within the group, we only include ratings from users in the modeling
    group.  However, we place no restriction on which notes to include in the model and instead
    include ratings on any note.  Including ratings on any note increases the amount of data
    available during training about each user, in effect also increasing the number of users
    and notes we are able to include in the model.

    Including notes by users outside of the modeling group means that the model will issue
    scores for notes which do not meet group modeling criteria (i.e. >80% of ratings are
    from users in the modeling group, and the author is also from the modeling group). We
    enforce these criteria *after* scoring in _postprocess_output so that the maximum amount
    of ratings are available during scoring.

    Args:
      ratings (pd.DataFrame): preprocessed ratings
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
      userEnrollment (pd.DataFrame): one row per user specifying enrollment properties

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame]:
        ratings: ratings filtered to only contain rows of interest
        noteStatusHistory: noteStatusHistory filtered to only contain rows of interest
    """
    userEnrollment = userEnrollment.rename(columns={c.participantIdKey: c.raterParticipantIdKey})
    userEnrollment = userEnrollment[userEnrollment[c.modelingGroupKey] == self._groupNumber]
    ratings = ratings.merge(userEnrollment[[c.raterParticipantIdKey]].drop_duplicates())
    return ratings, noteStatusHistory

  def _postprocess_output(
    self,
    noteScores: pd.DataFrame,
    userScores: pd.DataFrame,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    userEnrollment: pd.DataFrame,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter noteScores and userScores and add column containing modeling group.

    Enforce the requirements that:
      - Notes scored by this group model have >=80% of ratings from the modeling group.
      - Notes scored by this group model were authored by a member of the modeling group.
      - Users assigned a factor / intercept are in the modeling group.

    Args:
      noteScores: note outputs from scoring
      userScores: user outputs from scoring
      ratings (pd.DataFrame): preprocessed ratings
      noteStatusHistory (pd.DataFrame): one row per note; history of when note had each status
      userEnrollment (pd.DataFrame): one row per user specifying enrollment properties

    Returns:
      Tuple[pd.DataFrame, pd.DataFrame]:
        noteScores: filtered and updated note scoring output
        userScores: filtered and updated user scoring output
    """
    # Prune notes according to authorship filter.
    if self._requireInternalAuthor:
      noteScores = noteScores.merge(
        userEnrollment[[c.participantIdKey, c.modelingGroupKey]].rename(
          columns={c.participantIdKey: c.noteAuthorParticipantIdKey}
        ),
        how="left",
      )
      noteScores = noteScores[noteScores[c.modelingGroupKey] == self._groupNumber]
      noteScores = noteScores.drop(columns=c.modelingGroupKey)
    # Identify notes with enough ratings from within the modeling group.
    ratings = ratings.merge(
      userEnrollment[[c.participantIdKey, c.modelingGroupKey]].rename(
        columns={c.participantIdKey: c.raterParticipantIdKey}
      ),
      how="left",
    )
    ratings["inGroup"] = ratings[c.modelingGroupKey] == self._groupNumber
    ratios = ratings[[c.noteIdKey, "inGroup"]].groupby(c.noteIdKey).mean().reset_index()
    notesAboveThreshold = ratios[ratios["inGroup"] >= self._groupThreshold][[c.noteIdKey]]
    noteScores = noteScores.merge(notesAboveThreshold)
    # Note that even though ratings were restricted to the modeling group, users outside of
    # the modeling group may still have authored a note which was rated and may consequently
    # appear in the userScores.  Accordingly, we drop any user which was outside of the
    # modeling group from the user scores.
    userScores = userScores.merge(
      userEnrollment[[c.participantIdKey, c.modelingGroupKey]].rename(
        columns={c.participantIdKey: c.raterParticipantIdKey}
      ),
      how="left",
    )
    userScores = userScores[userScores[c.modelingGroupKey] == self._groupNumber]
    userScores = userScores.drop(columns=c.modelingGroupKey)
    # Set the modelingGroupKey column in each output
    noteScores[self._modelingGroupKey] = self._groupNumber
    userScores[self._modelingGroupKey] = self._groupNumber
    return noteScores, userScores
