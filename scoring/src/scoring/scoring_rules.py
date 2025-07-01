from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from . import constants as c
from .enums import Topics
from .explanation_tags import get_top_two_tags_for_note
from .pflip_plus_model import CRH, LABEL as PFLIP_LABEL

import numpy as np
import pandas as pd


logger = logging.getLogger("birdwatch.scoring_rules")
logger.setLevel(logging.INFO)

RuleAndVersion = namedtuple("RuleAndVersion", ["ruleName", "ruleVersion", "lockingEnabled"])
"""namedtuple identifying ScoringRule with a name and tracking revisions with a version."""


class RuleID(Enum):
  """Each RuleID must have a unique ruleName and can be assigned to at most one ScoringRule."""

  # Rules used by matrix_factorization_scorer.
  INITIAL_NMR = RuleAndVersion("InitialNMR", "1.0", False)
  GENERAL_CRH = RuleAndVersion("GeneralCRH", "1.0", False)
  GENERAL_CRNH = RuleAndVersion("GeneralCRNH", "1.0", False)
  UCB_CRNH = RuleAndVersion("UcbCRNH", "1.0", False)
  RATIO_CRNH = RuleAndVersion("RatioCRNH", "1.0", False)
  TAG_OUTLIER = RuleAndVersion("TagFilter", "1.0", False)
  ELEVATED_CRH = RuleAndVersion("CRHSuperThreshold", "1.0", False)
  NM_CRNH = RuleAndVersion("NmCRNH", "1.0", False)
  GENERAL_CRH_INERTIA = RuleAndVersion("GeneralCRHInertia", "1.0", False)
  ELEVATED_CRH_INERTIA = RuleAndVersion("ElevatedCRHInertia", "1.0", False)
  INCORRECT_OUTLIER = RuleAndVersion("FilterIncorrect", "1.0", False)
  LOW_DILIGENCE = RuleAndVersion("FilterLowDiligence", "1.0", False)
  LARGE_FACTOR = RuleAndVersion("FilterLargeFactor", "1.0", False)
  LOW_INTERCEPT = RuleAndVersion("RejectLowIntercept", "1.0", False)
  MIN_MINORITY_RATERS = RuleAndVersion("MinMinorityRaters", "1.0", False)
  RATER_BALANCE = RuleAndVersion("RaterBalance", "1.0", False)
  NO_HIGH_VOL_INTERCEPT = RuleAndVersion("NoHighVolIntercept", "1.0", False)
  NO_CORRELATED_INTERCEPT = RuleAndVersion("NoCorrelatedIntercept", "1.0", False)

  # Rules used in _meta_score.
  META_INITIAL_NMR = RuleAndVersion("MetaInitialNMR", "1.0", False)
  EXPANSION_MODEL = RuleAndVersion("ExpansionModel", "1.1", True)
  EXPANSION_PLUS_MODEL = RuleAndVersion("ExpansionPlusModel", "1.1", False)
  CORE_MODEL = RuleAndVersion("CoreModel", "1.1", True)
  CORE_WITH_TOPICS_MODEL = RuleAndVersion("CoreWithTopicsModel", "1.1", True)
  COVERAGE_MODEL = RuleAndVersion("CoverageModel", "1.1", False)
  GROUP_MODEL_1 = RuleAndVersion("GroupModel01", "1.1", True)
  GROUP_MODEL_2 = RuleAndVersion("GroupModel02", "1.1", True)
  GROUP_MODEL_3 = RuleAndVersion("GroupModel03", "1.1", True)
  GROUP_MODEL_4 = RuleAndVersion("GroupModel04", "1.1", True)
  GROUP_MODEL_5 = RuleAndVersion("GroupModel05", "1.1", True)
  GROUP_MODEL_6 = RuleAndVersion("GroupModel06", "1.1", True)
  GROUP_MODEL_7 = RuleAndVersion("GroupModel07", "1.1", True)
  GROUP_MODEL_8 = RuleAndVersion("GroupModel08", "1.1", True)
  GROUP_MODEL_9 = RuleAndVersion("GroupModel09", "1.1", True)
  GROUP_MODEL_10 = RuleAndVersion("GroupModel10", "1.1", True)
  GROUP_MODEL_11 = RuleAndVersion("GroupModel11", "1.1", True)
  GROUP_MODEL_12 = RuleAndVersion("GroupModel12", "1.1", True)
  GROUP_MODEL_13 = RuleAndVersion("GroupModel13", "1.1", True)
  GROUP_MODEL_14 = RuleAndVersion("GroupModel14", "1.1", True)
  GROUP_MODEL_33 = RuleAndVersion("GroupModel33", "1.1", True)
  GROUP_MODEL_33_NMR = RuleAndVersion("GroupModel33NMR", "1.1", True)
  TOPIC_MODEL_1 = RuleAndVersion("TopicModel01", "1.0", False)
  TOPIC_MODEL_2 = RuleAndVersion("TopicModel02", "1.0", False)
  TOPIC_MODEL_3 = RuleAndVersion("TopicModel03", "1.0", False)
  TOPIC_MODEL_4 = RuleAndVersion("TopicModel04", "1.0", False)
  MULTI_GROUP_MODEL_1 = RuleAndVersion("MultiGroupModel01", "1.0", True)
  INSUFFICIENT_EXPLANATION = RuleAndVersion("InsufficientExplanation", "1.0", True)
  SCORING_DRIFT_GUARD = RuleAndVersion("ScoringDriftGuard", "1.0", False)
  NMR_DUE_TO_MIN_STABLE_CRH_TIME = RuleAndVersion("NmrDueToMinStableCrhTime", "1.0", False)

  def get_name(self) -> str:
    """Returns a string combining the name and version to uniquely name the logic of the ScoringRule."""
    return f"{self.value.ruleName} (v{self.value.ruleVersion})"


class ScoringRule(ABC):
  """Scoring logic describing how to assign a ratingStatus given raw scoring signals and note attributes.

  Each ScoringRule must have a name and version. Each ScoringRule must implement a score_notes function,
  which accepts as input the raw attributes of notes and currently assigned lables and returns (1) a
  DataFrame specifying the noteIDs and associated status which the rule will assign, and (2) a DF
  containing any new columns which should be added to the output for those noteIDs.
  """

  def __init__(self, ruleID: RuleID, dependencies: Set[RuleID]):
    """Create a ScoringRule.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
    """
    self._ruleID = ruleID
    self._dependencies = dependencies

  def get_rule_id(self) -> RuleID:
    """Returns the RuleID uniquely identifying this ScoringRule."""
    return self._ruleID

  def get_name(self) -> str:
    """Returns a string combining the name and version to uniquely name the logic of the ScoringRule."""
    return self._ruleID.get_name()

  def check_dependencies(self, priorRules: Set[RuleID]) -> None:
    """Raise an AssertionError if rule dependencies have not been satisfied."""
    assert not (self._dependencies - priorRules)

  @abstractmethod
  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Identify which notes the ScoringRule should be active for, and any new columns to add for those notes.

    Args:
      noteStats: Raw note attributes, scoring signals and attirbutes for notes.
      currentLabels: the ratingStatus assigned to each note from prior ScoringRules.
      statusColumn: str indicating column where status should be assigned.

    Returns:
      Tuple[0]: DF specifying note IDs and associated status which this rule will update.
      Tuple[1]: DF containing noteIDs and any new columns to add to output
    """


class DefaultRule(ScoringRule):
  def __init__(self, ruleID: RuleID, dependencies: Set[RuleID], status: str):
    """Creates a ScoringRule which sets all note statuses to a default value.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
    """
    super().__init__(ruleID, dependencies)
    self._status = status

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Returns all noteIDs to initialize all note ratings to a default status (e.g. NMR)."""
    noteStatusUpdates = pd.DataFrame(noteStats[[c.noteIdKey]])
    noteStatusUpdates[statusColumn] = self._status
    return (noteStatusUpdates, None)


class RuleFromFunction(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    status: str,
    function: Callable[[pd.DataFrame], pd.Series],
    onlyApplyToNotesThatSayTweetIsMisleading: bool = True,
  ):
    """Creates a ScoringRule which wraps a boolean function.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
      function: accepts noteStats as input and returns a boolean pd.Series corresponding to
        rows matched by the function.  For example, a valid function would be:
        "lambda noteStats: noteStats[c.internalNoteInterceptKey] > 0.4"
      onlyApplyToNotesThatSayTweetIsMisleading: if True, only apply the rule to that subset of notes.
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._function = function
    self._onlyApplyToNotesThatSayTweetIsMisleading = onlyApplyToNotesThatSayTweetIsMisleading

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Returns noteIDs for notes matched by the boolean function."""
    mask = self._function(noteStats)
    if self._onlyApplyToNotesThatSayTweetIsMisleading:
      # Check for inequality with "not misleading" to include notes whose classificaiton
      # is nan (i.e. deleted notes).
      mask = mask & (noteStats[c.classificationKey] != c.noteSaysTweetIsNotMisleadingKey)

    noteStatusUpdates = noteStats.loc[mask][[c.noteIdKey]]
    noteStatusUpdates[statusColumn] = self._status
    return (noteStatusUpdates, None)


class ApplyModelResult(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    sourceColumn: str,
    checkFirmReject: bool = False,
    filterColumnPairs: List[Tuple[str, Any]] = [],
  ):
    """Propagate the note status from sourceColumn when the status is not NaN.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      sourceColumn: column containing note status (CRH, CRNH, NMR) to propagate to output,
    """
    super().__init__(ruleID, dependencies)
    self._sourceColumn = sourceColumn
    self._checkFirmReject = checkFirmReject
    self._filterColumnPairs = filterColumnPairs

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Propagates any status set in sourceColumn when it is non-NaN."""
    # If necessary, prune noteStats according to prior firm rejects
    if self._checkFirmReject:
      coreRejects = noteStats[c.coreRatingStatusKey].isin(
        {c.firmReject, c.currentlyRatedNotHelpful}
      )
      expansionRejects = noteStats[c.expansionRatingStatusKey].isin(
        {c.firmReject, c.currentlyRatedNotHelpful}
      )
      crhBlocked = coreRejects | (noteStats[c.coreRatingStatusKey].isna() & expansionRejects)
      # FIRM_REJECT from upstream models should block both CRH and NYH status
      crhNotes = noteStats[self._sourceColumn].isin({c.currentlyRatedHelpful, c.needsYourHelp})
      noteStats = noteStats[~(crhBlocked & crhNotes)]
    # If necessary, prune noteStatus based on filter column pairs
    if self._filterColumnPairs:
      for col, value in self._filterColumnPairs:
        noteStats = noteStats[noteStats[col] == value]
    # Generate the set of note status updates
    statusUpdateRows = ~noteStats[self._sourceColumn].isna()
    noteStatusUpdates = noteStats[statusUpdateRows][[c.noteIdKey, self._sourceColumn]].rename(
      columns={self._sourceColumn: statusColumn}
    )
    # Rename FIRM_REJECT to NEEDS_MORE_RATINGS since the status will be exported as the final status
    noteStatusUpdates.loc[
      noteStatusUpdates[statusColumn] == c.firmReject, statusColumn
    ] = c.needsMoreRatings
    assert (
      noteStatusUpdates[statusColumn]
      .isin(
        {c.currentlyRatedHelpful, c.currentlyRatedNotHelpful, c.needsMoreRatings, c.needsYourHelp}
      )
      .all()
    ), "status must be set to CRH, CRNH, NMR or NYH"
    return (noteStatusUpdates, None)


class FilterTagOutliers(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    status: str,
    tagFilterThresholds: Dict[str, float],
    minAdjustedTotal: float = 2.5,
  ):
    """Filter CRH notes for outliers with high levels of any particular tag.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
      minAdjustedTotal: For a filter to trigger, the adjusted total of a tag must
        exceed the minAdjustedTotal.
      tagFilterThresholds: For a filter to trigger, the adjusted ratio value for a
        tag must exceed the given value (computed in prescoring as the Nth percentile
        for notes currently rated as CRH).
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._minAdjustedTotal = minAdjustedTotal
    self._tagFilterThresholds = tagFilterThresholds

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns notes on track for CRH with high levels of any tag to receive NMR status."""
    # Prune noteStats to exclude CRNH notes.  CRNH will have stronger downstream effects, so
    # we don't want to over-write that status.
    candidateNotes = currentLabels[currentLabels[statusColumn] != c.currentlyRatedNotHelpful][
      [c.noteIdKey]
    ]
    noteStats = noteStats.merge(candidateNotes, on=c.noteIdKey, how="inner")
    logger.info(f"Candidate notes prior to tag filtering: {len(noteStats)}")

    # Identify impacted notes.
    impactedNotes = pd.DataFrame.from_dict(
      {
        c.noteIdKey: pd.Series([], dtype=np.int64),
        c.activeFilterTagsKey: pd.Series([], dtype=object),
      }
    )
    logger.info("Checking note tags:")
    for tag in c.notHelpfulTagsTSVOrder:
      adjustedColumn = f"{tag}{c.adjustedSuffix}"
      adjustedRatioColumn = f"{adjustedColumn}{c.ratioSuffix}"
      logger.info(tag)
      if tag == c.notHelpfulHardToUnderstandKey:
        logger.info(f"outlier filtering disabled for tag: {tag}")
        continue
      tagFilteredNotes = noteStats[
        # Adjusted total must pass minimum threhsold set across all tags.
        (noteStats[adjustedColumn] > self._minAdjustedTotal)
        # Adjusted ratio must exceed percentile based total for this specific tag.
        & (noteStats[adjustedRatioColumn] > self._tagFilterThresholds[adjustedRatioColumn])
      ][c.noteIdKey]
      impactedNotes = pd.concat(
        [impactedNotes, pd.DataFrame({c.noteIdKey: tagFilteredNotes, c.activeFilterTagsKey: tag})],
        unsafeAllowed=[c.defaultIndexKey, c.activeFilterTagsKey],
      )
    # log and consolidate imapcted notes
    logger.info(f"Total {{note, tag}} pairs where tag filter logic triggered: {len(impactedNotes)}")
    impactedNotes = impactedNotes.groupby(c.noteIdKey).aggregate(list).reset_index()
    impactedNotes[c.activeFilterTagsKey] = [
      ",".join(tags) for tags in impactedNotes[c.activeFilterTagsKey]
    ]
    logger.info(f"Total unique notes impacted by tag filtering: {len(impactedNotes)}")
    noteStatusUpdates = impactedNotes[[c.noteIdKey]].drop_duplicates()
    noteStatusUpdates[statusColumn] = self._status
    return (noteStatusUpdates, impactedNotes)


class FilterIncorrect(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    status: str,
    tagThreshold: int,
    voteThreshold: int,
    weightedTotalVotes: float,
  ):
    """Filter CRH notes for outliers with high levels of incorrect tag from similar factor raters.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
      tagThreshold: threshold for number of included raters to issue a tag
      voteThreshold: threshold for number of included raters (raters must have issued a NH tag to be inclueed)
      weightedTotalVotes: For the filter to trigger, the sum of weighted incorrect votes must
        exceed the minAdjustedTotal.
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._tagThreshold = tagThreshold
    self._voteThreshold = voteThreshold
    self._weightedTotalVotes = weightedTotalVotes

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns notes on track for CRH with high levels of any tag to receive NMR status."""
    # Prune noteStats to exclude CRNH notes.  CRNH will have stronger downstream effects, so
    # we don't want to over-write that status.
    candidateNotes = currentLabels[currentLabels[statusColumn] != c.currentlyRatedNotHelpful][
      [c.noteIdKey]
    ]
    noteStats = noteStats.merge(candidateNotes, on=c.noteIdKey, how="inner")

    # Identify impacted notes.
    noteStatusUpdates = noteStats.loc[
      (noteStats["notHelpfulIncorrect_interval"] >= self._tagThreshold)
      & (noteStats["num_voters_interval"] >= self._voteThreshold)
      & (noteStats["tf_idf_incorrect_interval"] >= self._weightedTotalVotes)
    ][[c.noteIdKey]]

    pd.testing.assert_frame_equal(noteStatusUpdates, noteStatusUpdates.drop_duplicates())

    logger.info(f"Total notes impacted by incorrect filtering: {len(noteStatusUpdates)}")
    noteStatusUpdates[statusColumn] = self._status

    return (noteStatusUpdates, None)


class FilterLowDiligence(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    status: str,
    interceptThreshold: float,
  ):
    """Filter CRH notes which have a high low diligence intercept.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
      interceptThreshold: threshold for low diligence intercept
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._interceptThreshold = interceptThreshold

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns notes on track for CRH with a high low diligence intercept."""
    # Prune noteStats to exclude CRNH notes.  CRNH will have stronger downstream effects, so
    # we don't want to over-write that status.
    candidateNotes = currentLabels[currentLabels[statusColumn] != c.currentlyRatedNotHelpful][
      [c.noteIdKey]
    ]
    noteStats = noteStats.merge(candidateNotes, on=c.noteIdKey, how="inner")

    # Identify impacted notes.
    noteStatusUpdates = noteStats.loc[
      noteStats[c.lowDiligenceNoteInterceptKey] > self._interceptThreshold
    ][[c.noteIdKey]]

    pd.testing.assert_frame_equal(noteStatusUpdates, noteStatusUpdates.drop_duplicates())

    logger.info(f"Total notes impacted by low diligence filtering: {len(noteStatusUpdates)}")
    noteStatusUpdates[statusColumn] = self._status

    return (noteStatusUpdates, None)


class FilterLargeFactor(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    status: str,
    factorThreshold: float,
  ):
    """Filter CRH notes which have especially large factors (whether positive or negative).

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
      factorThreshold: threshold for filtering large factors
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._factorThreshold = factorThreshold

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns notes on track for CRH with a high low diligence intercept."""
    # Prune noteStats to exclude CRNH notes.  CRNH will have stronger downstream effects, so
    # we don't want to over-write that status.
    candidateNotes = currentLabels[currentLabels[statusColumn] == c.currentlyRatedHelpful][
      [c.noteIdKey]
    ]
    noteStats = noteStats.merge(candidateNotes, on=c.noteIdKey, how="inner")

    # Identify impacted notes.
    noteStatusUpdates = noteStats.loc[
      noteStats[c.internalNoteFactor1Key].abs() > self._factorThreshold
    ][[c.noteIdKey]]

    pd.testing.assert_frame_equal(noteStatusUpdates, noteStatusUpdates.drop_duplicates())

    logger.info(f"Total notes impacted by large factor filtering: {len(noteStatusUpdates)}")
    noteStatusUpdates[statusColumn] = self._status

    return (noteStatusUpdates, None)


class RequireMinMinorityRaters(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    status: str,
    minMinorityNetHelpfulRatings: int,
  ):
    """Set notes to NYH that haven't reached the minimum required number of minority raters.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
      minMinorityNetHelpfulRatings: minimum number of net helpful minority ratings
        required for CRH status.
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._minMinorityNetHelpfulRatings = minMinorityNetHelpfulRatings

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns notes on track for CRH with a high low diligence intercept."""
    # Prune noteStats to only include notes that are on track to be CRH
    candidateNotes = currentLabels[currentLabels[statusColumn] == c.currentlyRatedHelpful][
      [c.noteIdKey]
    ]
    noteStats = noteStats.merge(candidateNotes, on=c.noteIdKey, how="inner")

    # Identify impacted notes.
    noteStatusUpdates = noteStats.loc[
      noteStats[c.netMinHelpfulKey] < self._minMinorityNetHelpfulRatings
    ][[c.noteIdKey]]

    pd.testing.assert_frame_equal(noteStatusUpdates, noteStatusUpdates.drop_duplicates())

    logger.info(
      f"Total notes impacted by minimum minority rater requirement: {len(noteStatusUpdates)}"
    )
    noteStatusUpdates[statusColumn] = self._status

    return (noteStatusUpdates, None)


class NoHighVolIntercept(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    status: str,
    crhThresholdNoHighVol: float,
  ):
    """Set notes to NYH when the intercept drops too low if high volume raters are excluded.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
      crhThresholdNoHighVol: minimum note intercept when omitting high volume raters.
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._crhThresholdNoHighVol = crhThresholdNoHighVol

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns notes on track for CRH with a low intercept when omitting high volume raters."""
    # Prune noteStats to only include notes that are on track to be CRH
    candidateNotes = currentLabels[currentLabels[statusColumn] == c.currentlyRatedHelpful][
      [c.noteIdKey]
    ]
    noteStats = noteStats.merge(candidateNotes, on=c.noteIdKey, how="inner")

    # Identify impacted notes.
    noteStatusUpdates = noteStats.loc[
      noteStats[c.internalNoteInterceptNoHighVolKey] < self._crhThresholdNoHighVol
    ][[c.noteIdKey]]

    pd.testing.assert_frame_equal(noteStatusUpdates, noteStatusUpdates.drop_duplicates())

    logger.info(
      f"Total notes impacted by NoHighVol intercept requirement: {len(noteStatusUpdates)}"
    )
    noteStatusUpdates[statusColumn] = self._status

    return (noteStatusUpdates, None)


class NoCorrelatedIntercept(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    status: str,
    crhThresholdNoCorrelated: float,
  ):
    """Set notes to NYH when the intercept drops too low if correlated raters are excluded.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
      crhThresholdNoCorrelated: minimum note intercept when omitting correlated raters.
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._crhThresholdNoCorrelated = crhThresholdNoCorrelated

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns notes on track for CRH with a low intercept when omitting correlated raters."""
    # Prune noteStats to only include notes that are on track to be CRH
    candidateNotes = currentLabels[currentLabels[statusColumn] == c.currentlyRatedHelpful][
      [c.noteIdKey]
    ]
    noteStats = noteStats.merge(candidateNotes, on=c.noteIdKey, how="inner")

    # Identify impacted notes.
    noteStatusUpdates = noteStats.loc[
      noteStats[c.internalNoteInterceptNoCorrelatedKey] < self._crhThresholdNoCorrelated
    ][[c.noteIdKey]]

    pd.testing.assert_frame_equal(noteStatusUpdates, noteStatusUpdates.drop_duplicates())

    logger.info(
      f"Total notes impacted by NoCorrelated intercept requirement: {len(noteStatusUpdates)}"
    )
    noteStatusUpdates[statusColumn] = self._status

    return (noteStatusUpdates, None)


class RequireRaterBalance(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    status: str,
    minNetBalance: float,
    minMinorityNetHelpfulRatings: int = 4,
    maxMinorityNetHelpfulRatings: int = 10,
  ):
    """Set notes to NYH that haven't reached the minimum required number of minority raters.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
      minNetBalance: minimum level of net balance
      minMinorityNetHelpfulRatings: rule only applies when there are at least this many minority
        net helpful ratings.
      maxMinorityNetHelpfulRatings: rule only applies when there are less than this many minority
        net helpful ratings.
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._minNetBalance = minNetBalance
    self._minMinorityNetHelpfulRatings = minMinorityNetHelpfulRatings
    self._maxMinorityNetHelpfulRatings = maxMinorityNetHelpfulRatings

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns notes on track for CRH with a high low diligence intercept."""
    # Prune noteStats to only include notes that are on track to be CRH
    candidateNotes = currentLabels[currentLabels[statusColumn] == c.currentlyRatedHelpful][
      [c.noteIdKey]
    ]
    noteStats = noteStats.merge(candidateNotes, on=c.noteIdKey, how="inner")

    # Identify impacted notes.
    noteStatusUpdates = noteStats.loc[
      (noteStats[c.netMinHelpfulKey] >= self._minMinorityNetHelpfulRatings)
      & (noteStats[c.netMinHelpfulKey] < self._maxMinorityNetHelpfulRatings)
      & (noteStats[c.netMinHelpfulRatioKey] < self._minNetBalance)
    ][[c.noteIdKey]]

    pd.testing.assert_frame_equal(noteStatusUpdates, noteStatusUpdates.drop_duplicates())

    logger.info(
      f"Total notes impacted by minimum rater balance requirement: {len(noteStatusUpdates)}"
    )
    noteStatusUpdates[statusColumn] = self._status

    return (noteStatusUpdates, None)


class NmrDueToMinStableCrhTime(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    requiredStableCrhMinutesThreshold: int = 30,
    maxStableCrhMinutesThreshold: int = 180,
    maxNyhMinutesThreshold: int = 360,
  ):
    """
    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the
      ScoringRule.
      dependencies: Rules which must run before this rule can run.
      requiredStableCrhMinutesThreshold: threshold for min required stable CRH time, in minutes.
      maxStableCrhMinutesThreshold: threshold for max stable CRH time, in minutes.
      maxNyhMinutesThreshold: threshold for NYH time, in minutes
    """
    super().__init__(ruleID, dependencies)
    self.requiredStableCrhMinutesThreshold = requiredStableCrhMinutesThreshold
    self.maxStableCrhMinutesThreshold = maxStableCrhMinutesThreshold
    self.maxNyhMinutesThreshold = maxNyhMinutesThreshold

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # For any notes that were CRH in the previous scoring run, the only possible status update
    # from this scoring rule would be to switch a NYH status to CRH.  Notes that are CRH, NMR
    # or CRNH in the current scoring round require no action.

    noteStats = noteStats.merge(currentLabels, on=c.noteIdKey, how="inner")
    nyhToCrhUpdates = noteStats[
      (noteStats[c.currentLabelKey] == c.currentlyRatedHelpful)
      & (noteStats[statusColumn] == c.needsYourHelp)
    ][[c.noteIdKey]]
    nyhToCrhUpdates[statusColumn] = c.currentlyRatedHelpful

    # Since we have identified the notes that require a NYH->CRH status change, we now drop all
    # notes that were CRH before the current scoring run.
    noteStats = noteStats[noteStats[c.currentLabelKey] != c.currentlyRatedHelpful]

    # Identify impacted notes:
    # (1) CRH or NYH from current run
    #     (A) If timestampMillisOfNmrDueToMinStableCrhTime is NaN:
    #         Set status to NMR, set timestampMillisOfNmrDueToMinStableCrhTime to now.
    #     (B) If timestampMillisOfNmrDueToMinStableCrhTime is -1:
    #         (a) If note was CRH in current run, set status to NMR and set
    #             timestampMillisOfNmrDueToMinStableCrhTime to now.
    #         (b) If note was NYH in current run, set status to NMR and leave
    #             timestampMillisOfNmrDueToMinStableCrhTime as -1.
    #     (C) If timestampMillisOfNmrDueToMinStableCrhTime t is >0:
    #         -- Start with cases where we know we're leaving stabilization because the time
    #         -- limit has been exceeded.
    #         (a) If (now - t) is larger than maxStableCrhMinutesThreshold and current run is CRH,
    #             set CRH and clear timestampMillisOfNmrDueToMinStableCrhTime.
    #         (b) If (now - t) is larger than maxNyhMinutesThreshold and current run is NYH,
    #             set NMR and clear timestampMillisOfNmrDueToMinStableCrhTime.
    #         -- Next, handle cases where we know we're staying in stabilization because the
    #         -- criteria to leave to CRH has not been met.
    #         (c) Else if (now - t) is smaller than requiredStableCrhMinutesThreshold, NMR.
    #         (d) Else if the note is being scored by a TopicModel but TopicModel is not confident,
    #             NMR.
    #         (e) Else if pflip model predicts that the note status will flip or note has
    #             flipped in the past, NMR.
    #         -- Lastly, if the time limit hasn't been met and there is no criteria that requires
    #         -- the note to remain in stabilization, the status depends on the scoring status.
    #         (f) Else if current status is NYH, set NMR.
    #         (g) Else, current status must be CRH, so set CRH and clear
    #             timestampMillisOfNmrDueToMinStableCrhTime.
    #
    # (2) NMR or CRNH from current run and timestampMillisOfNmrDueToMinStableCrhTime is >0.
    #     Clear timestampMillisOfNmrDueToMinStableCrhTime.
    noteStatusUpdates = noteStats.loc[
      # Notice that this rule is scoped to notes that were not CRH in the last scoring run
      # (see above), and are either trying to switch to CRH/NYH now OR are already in a stabilization
      # period.
      (noteStats[statusColumn].isin({c.currentlyRatedHelpful, c.needsYourHelp}))
      | (
        # Note that timestampMillisOfNmrDueToMinStableCrhTimeKey is NaN if a note has
        # never entered a stabilization period, >0 if a note is in a stabilization period, and -1
        # if a note has exited a stabilization period for any reason (whether set to CRH or
        # reverted to NMR before the period ended).
        noteStats[c.timestampMillisOfNmrDueToMinStableCrhTimeKey].notna()
        & (noteStats[c.timestampMillisOfNmrDueToMinStableCrhTimeKey] > 0)
      )
    ][
      [
        c.noteIdKey,
        c.timestampMillisOfNmrDueToMinStableCrhTimeKey,
        c.firstNonNMRLabelKey,
        c.noteTopicKey,
        c.topicNoteConfidentKey,
        statusColumn,
        PFLIP_LABEL,
      ]
    ]

    pd.testing.assert_frame_equal(noteStatusUpdates, noteStatusUpdates.drop_duplicates())

    newStatusColumn = statusColumn + "_new"
    noteStatusUpdates[newStatusColumn] = np.nan
    noteStatusUpdates[c.updatedTimestampMillisOfNmrDueToMinStableCrhTimeKey] = noteStatusUpdates[
      c.timestampMillisOfNmrDueToMinStableCrhTimeKey
    ]

    # (1)-(A): Never in a stabilization period. Enter stabilization period if note is trying
    # to go CRH or NYH.
    notesGoingCrh = noteStatusUpdates[statusColumn] == c.currentlyRatedHelpful
    notesGoingNyh = noteStatusUpdates[statusColumn] == c.needsYourHelp
    notesNeverInStabilization = noteStatusUpdates[
      c.timestampMillisOfNmrDueToMinStableCrhTimeKey
    ].isna()
    noteStatusUpdates.loc[
      (notesGoingCrh | notesGoingNyh) & notesNeverInStabilization,
      [newStatusColumn, c.updatedTimestampMillisOfNmrDueToMinStableCrhTimeKey],
    ] = [c.needsMoreRatings, c.epochMillis]

    # (1)-(B): Notes that have previously exited a stabilization period.
    notesPreviouslyInStabilization = (
      noteStatusUpdates[c.timestampMillisOfNmrDueToMinStableCrhTimeKey] < 0
    )
    # (1)-(B)-(a): If the note was CRH in the current run, allow the note to enter stabilization
    noteStatusUpdates.loc[
      notesGoingCrh & notesPreviouslyInStabilization,
      [newStatusColumn, c.updatedTimestampMillisOfNmrDueToMinStableCrhTimeKey],
    ] = [c.needsMoreRatings, c.epochMillis]

    # (1)-(B)-(b): If the note was NYH in the current run, do not allow the note to enter stabilization.
    noteStatusUpdates.loc[
      notesGoingNyh & notesPreviouslyInStabilization, newStatusColumn
    ] = c.needsMoreRatings

    # (1)-(C): Currently in stabilization period.
    notesAlreadyInStabilization = (
      noteStatusUpdates[c.timestampMillisOfNmrDueToMinStableCrhTimeKey] > 0
    )
    # (1)-(C)-(a): Exit stabilization period to CRH if the note has been in the period for
    # longer than maxStableCrhMinutesThreshold and the note is currently scored CRH.
    inStabilizationLongerThanCrhMax = (
      c.epochMillis - noteStatusUpdates[c.timestampMillisOfNmrDueToMinStableCrhTimeKey]
      > self.maxStableCrhMinutesThreshold * 60 * 1000
    )

    noteStatusUpdates.loc[
      notesGoingCrh & notesAlreadyInStabilization & inStabilizationLongerThanCrhMax,
      [newStatusColumn, c.updatedTimestampMillisOfNmrDueToMinStableCrhTimeKey],
    ] = [c.currentlyRatedHelpful, -1]

    # (1)-(C)-(b): Exit stabilization period to NMR if the note has been in the period for
    # longer than maxNyhMinutesThreshold and the note is currently scored NYH.
    inStabilizationLongerThanNyhMax = (
      c.epochMillis - noteStatusUpdates[c.timestampMillisOfNmrDueToMinStableCrhTimeKey]
      > self.maxNyhMinutesThreshold * 60 * 1000
    )

    noteStatusUpdates.loc[
      notesGoingNyh & notesAlreadyInStabilization & inStabilizationLongerThanNyhMax,
      [newStatusColumn, c.updatedTimestampMillisOfNmrDueToMinStableCrhTimeKey],
    ] = [c.needsMoreRatings, -1]

    # Set inStabilizationLongerThanMax such that it reflect any note  that has been in stabilization
    # longer than allowed given the limit for the status of that particular note.  This vector will be
    # used below to avoid overwriting status updates for these notes.
    inStabilizationLongerThanMax = (
      notesGoingCrh & notesAlreadyInStabilization & inStabilizationLongerThanCrhMax
    ) | (notesGoingNyh & notesAlreadyInStabilization & inStabilizationLongerThanNyhMax)

    # (1)-(C)-(c): Remain in stabilization period if the note has been in the period for shorter
    # than requiredStableCrhMinutesThreshold
    inStabilizationShorterThanMin = (
      c.epochMillis - noteStatusUpdates[c.timestampMillisOfNmrDueToMinStableCrhTimeKey]
      < self.requiredStableCrhMinutesThreshold * 60 * 1000
    )

    noteStatusUpdates.loc[
      (notesGoingCrh | notesGoingNyh) & notesAlreadyInStabilization & inStabilizationShorterThanMin,
      newStatusColumn,
    ] = c.needsMoreRatings

    # (1)-(C)-(d): Remain in stabilization period if the note has been in the period for between
    # requiredStableCrhMinutesThreshold and maxStableCrhMinutesThreshold, and the note is being
    # scored by a TopicModel but TopicModel is not confident
    notesScoredByTopicButNotConfident = (
      (~noteStatusUpdates[c.noteTopicKey].isna())
      & (~noteStatusUpdates[c.topicNoteConfidentKey].isna())
      & (~(noteStatusUpdates[c.topicNoteConfidentKey].astype(pd.BooleanDtype())))
    )
    noteStatusUpdates.loc[
      (notesGoingCrh | notesGoingNyh)
      & notesAlreadyInStabilization
      & notesScoredByTopicButNotConfident
      & ~inStabilizationShorterThanMin
      & ~inStabilizationLongerThanMax,
      newStatusColumn,
    ] = c.needsMoreRatings

    # (1)-(C)-(e): Remain in stabilization period if the note has been in the period for between
    # requiredStableCrhMinutesThreshold and maxStableCrhMinutesThreshold, and the note has a history
    # of flipping or is predicted to flip
    notesThatAlreadyFlipped = noteStatusUpdates[c.firstNonNMRLabelKey] == c.currentlyRatedHelpful
    notesPredictedToFlip = noteStatusUpdates[PFLIP_LABEL] != CRH
    noteStatusUpdates.loc[
      (notesGoingCrh | notesGoingNyh)
      & notesAlreadyInStabilization
      & ~notesScoredByTopicButNotConfident
      & (notesThatAlreadyFlipped | notesPredictedToFlip)
      & ~inStabilizationShorterThanMin
      & ~inStabilizationLongerThanMax,
      newStatusColumn,
    ] = c.needsMoreRatings
    pflipCounts = noteStatusUpdates[
      (notesGoingCrh | notesGoingNyh)
      & notesAlreadyInStabilization
      & (~notesThatAlreadyFlipped)
      & ~inStabilizationShorterThanMin
      & ~inStabilizationLongerThanMax
    ][PFLIP_LABEL].value_counts(dropna=False)
    logger.info(f"pflip predictions for notes where pflip has an impact: {pflipCounts}")

    # (1)-(C)-(f): Remain in stabilization if the note was scored as NYH and hasn't already met a
    # criteria that required it to either leave or remain in stabilization.
    noteStatusUpdates.loc[
      notesGoingNyh
      & notesAlreadyInStabilization
      & ~notesScoredByTopicButNotConfident
      & ~notesThatAlreadyFlipped
      & ~notesPredictedToFlip
      & ~inStabilizationShorterThanMin
      & ~inStabilizationLongerThanMax,
      newStatusColumn,
    ] = c.needsMoreRatings

    # (1)-(C)-(g): Exit stabilization period to CRH if the note has been in the period for between
    # requiredStableCrhMinutesThreshold and maxStableCrhMinutesThreshold, and the note doesn't have
    # a history of flipping or is not predicted to flip, and the note is currently scored CRH
    noteStatusUpdates.loc[
      notesGoingCrh
      & notesAlreadyInStabilization
      & ~notesScoredByTopicButNotConfident
      & ~notesThatAlreadyFlipped
      & ~notesPredictedToFlip
      & ~inStabilizationShorterThanMin
      & ~inStabilizationLongerThanMax,
      [newStatusColumn, c.updatedTimestampMillisOfNmrDueToMinStableCrhTimeKey],
    ] = [c.currentlyRatedHelpful, -1]

    # (2): Exit a stabilization period if the note stops trying to go to CRH or NMR and we're in
    # a stabilization period.  Set updated timestamp to -1.
    noteStatusUpdates.loc[
      (~notesGoingCrh) & (~notesGoingNyh) & notesAlreadyInStabilization,
      c.updatedTimestampMillisOfNmrDueToMinStableCrhTimeKey,
    ] = -1

    # Identify rows within noteStatusUpdates that actually have a status change.
    noteStatusUpdatesWithStatusChange = noteStatusUpdates.loc[
      (noteStatusUpdates[newStatusColumn].notna())
      & (noteStatusUpdates[statusColumn] != noteStatusUpdates[newStatusColumn])
    ][[c.noteIdKey, newStatusColumn]]
    noteStatusUpdatesWithStatusChange.rename(columns={newStatusColumn: statusColumn}, inplace=True)

    # Augment noteStatusUpdatesWithStatusChange to include notes that were previously CRH but were
    # NYH in this scoring run.
    noteStatusUpdatesWithStatusChange = pd.concat(
      [noteStatusUpdatesWithStatusChange, nyhToCrhUpdates[[c.noteIdKey, statusColumn]]]
    )

    logger.info(
      f"Total notes impacted by NmrDueToMinStableCrhTime: "
      f"{len(noteStatusUpdatesWithStatusChange)}"
    )

    # return pre-stabilization statuses (also before insufficient explanation and scoring drift guard)
    noteStatusUpdates = noteStatusUpdates.merge(
      currentLabels[[c.noteIdKey, statusColumn]].rename(
        columns={statusColumn: c.preStabilizationRatingStatusKey}
      ),
      on=c.noteIdKey,
      how="outer",
    )
    return (
      noteStatusUpdatesWithStatusChange,
      noteStatusUpdates[
        [
          c.noteIdKey,
          c.updatedTimestampMillisOfNmrDueToMinStableCrhTimeKey,
          c.preStabilizationRatingStatusKey,
        ]
      ],
    )


class RejectLowIntercept(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    status: str,
    firmRejectThreshold: float,
  ):
    """Set notes with an intercept below firmRejectThreshold to firmReject, preventing downstream CRH.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
      firmRejectThreshold: firmReject notes with an intercept below this threshold
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._firmRejectThreshold = firmRejectThreshold

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns notes on track for NMR with an intercept below firmRejectThreshold."""
    # Require that notes are currently NMR.  If the note is already on track for firmReject, no need
    # to update the status since a more specific rule has already acted on the note.  If the note is
    # on track for CRNH, leave status unchanged so the finalRatingStatus is CRNH.
    candidateNotes = currentLabels[currentLabels[statusColumn] != c.currentlyRatedNotHelpful][
      [c.noteIdKey]
    ]
    noteStats = noteStats.merge(candidateNotes, on=c.noteIdKey, how="inner")
    noteStatusUpdates = noteStats.loc[
      (noteStats[c.internalNoteInterceptKey] < self._firmRejectThreshold)
    ][[c.noteIdKey]]
    noteStatusUpdates[statusColumn] = self._status
    return (noteStatusUpdates, None)


class ApplyGroupModelResult(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    groupNumber: int,
    coreCrhThreshold: Optional[float],
    expansionCrhThreshold: Optional[float],
    minSafeguardThreshold: float = 0.3,
  ):
    """Set CRH status based on a modeling group result.

    This rule sets CRH note status based on group models subject to several criteria:
      * The note must have CRH status from the group model.
      * The note must currently be scored as NMR.  This criteria guarantees that (1) group
        models strictly expand coverage and (2) notes which were rated CRH by the core
        model never have the decidedBy field overwritten by a less confident model.
      * The note must have an intercept from either the core or expansion models, and the intercept
        of the most confident model must fall within a defined range.  We construct the range
        to guarantee we can avoid CRHing notes which substantially lacked broad appeal, and
        to guarantee that we will not CRH a note which was blocked by tag or inaccuracy filtering
        from either the core or expansion models, as applicable.

    Args:
      ruleID: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      groupNumber: modeling group index which this instance of ApplyGroupModelResult should act on.
      coreCrhThreshold: maximum intercept allowed on core model for group model CRH notes.
      expansionCrhThreshold: maximum intercept allowed on expansion model for group model CRH notes.
      minSafeguardThreshold: minimum intercept for core or expansion model.
    """
    super().__init__(ruleID, dependencies)
    self._groupNumber = groupNumber
    self._minSafeguardThreshold = minSafeguardThreshold
    self._coreCrhThreshold = coreCrhThreshold
    self._expansionCrhThreshold = expansionCrhThreshold
    assert self._minSafeguardThreshold is not None

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Flip notes from NMR to CRH based on group models and subject to core/expansion model safeguards."""
    # Identify notes blocked from CRH status due to FR/CRNH status in core or expansion
    coreRejects = noteStats[c.coreRatingStatusKey].isin({c.firmReject, c.currentlyRatedNotHelpful})
    expansionRejects = noteStats[c.expansionRatingStatusKey].isin(
      {c.firmReject, c.currentlyRatedNotHelpful}
    )
    blocked = coreRejects | (noteStats[c.coreRatingStatusKey].isna() & expansionRejects)
    noteStats = noteStats[~blocked]
    # Generate the set of note status updates
    probationaryCRHOrNYHNotes = noteStats[
      (noteStats[c.groupRatingStatusKey].isin({c.currentlyRatedHelpful, c.needsYourHelp}))
      & (noteStats[c.modelingGroupKey] == self._groupNumber)
    ][[c.noteIdKey, c.groupRatingStatusKey]]
    # Identify notes which are currently NMR or NYH.
    currentNMROrNYHNotes = currentLabels[
      currentLabels[statusColumn].isin({c.needsMoreRatings, c.needsYourHelp})
    ][[c.noteIdKey, statusColumn]]
    # Identify candidate note status updates.  This requires pruning to notes that have a NMR->NYH, NMR->CRH
    # or NYH->CRH transition - in other words dropping NYH->NYH transitions.
    noteStatusUpdates = probationaryCRHOrNYHNotes.merge(
      currentNMROrNYHNotes, on=c.noteIdKey, how="inner"
    )
    noteStatusUpdates = noteStatusUpdates[
      (noteStatusUpdates[c.groupRatingStatusKey] != c.needsYourHelp)
      | (noteStatusUpdates[statusColumn] != c.needsYourHelp)
    ][[c.noteIdKey, c.groupRatingStatusKey]]
    # If necessary, identify notes which pass score bound checks for expansion and core models.
    # Apply min and max threhsolds to core and expansion intercepts
    noteStats = noteStats[[c.noteIdKey, c.coreNoteInterceptKey, c.expansionNoteInterceptKey]].copy()
    noteStats["core"] = noteStats[c.coreNoteInterceptKey] > self._minSafeguardThreshold
    if self._coreCrhThreshold is not None:
      noteStats["core"] = noteStats["core"] & (
        noteStats[c.coreNoteInterceptKey] < self._coreCrhThreshold
      )
    noteStats.loc[noteStats[c.coreNoteInterceptKey].isna(), "core"] = np.nan
    noteStats["expansion"] = noteStats[c.expansionNoteInterceptKey] > self._minSafeguardThreshold
    if self._expansionCrhThreshold is not None:
      noteStats["expansion"] = noteStats["expansion"] & (
        noteStats[c.expansionNoteInterceptKey] < self._expansionCrhThreshold
      )
    noteStats.loc[noteStats[c.expansionNoteInterceptKey].isna(), "expansion"] = np.nan

    # Prioritize core over expansion intercepts when available
    def _get_value(row):
      idx = row.first_valid_index()
      # If either core or expansion had an intercept then return whether it was in the valid
      # range.  If neither had an intercept, return False.  Preference is given to core due
      # to the ordering when selecting columns from noteStats below.
      if idx is None:
        return False
      elif row[idx] == 1.0:
        return True
      elif row[idx] == 0.0:
        return False
      else:
        assert False, f"unexpected value: {row[idx]}"

    with c.time_block("Get value apply for group model"):
      noteStats["actionable"] = noteStats[["core", "expansion"]].apply(_get_value, axis=1)

    # Filter set of note status updates to only include actionable notes
    actionableNotes = noteStats[noteStats["actionable"]][[c.noteIdKey]]
    noteStatusUpdates = noteStatusUpdates.merge(actionableNotes, on=c.noteIdKey, how="inner")

    # Set note status and return
    noteStatusUpdates = noteStatusUpdates.rename(columns={c.groupRatingStatusKey: statusColumn})

    return (noteStatusUpdates, None)


class ApplyNMRGroupModelResult(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    groupNumber: int,
  ):
    """Set NMR status based on a modeling group result.

    This rule sets NMR note status based on group models subject to several criteria:
      * The note must have NMR status from the group model.
      * The note must currently be scored as CRH.  This criteria guarantees that (1) this group
        model strictly reduces coverage

    Args:
      ruleID: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      groupNumber: modeling group index which this instance of ApplyGroupModelResult should act on.
    """
    super().__init__(ruleID, dependencies)
    self._groupNumber = groupNumber

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Flip notes from CRH to NMR based on group models."""
    # Generate the set of note status updates
    probationaryNMRNotes = noteStats[
      (noteStats[c.groupRatingStatusKey].isin({c.needsMoreRatings, c.currentlyRatedNotHelpful}))
      & (noteStats[c.modelingGroupKey] == self._groupNumber)
    ][[c.noteIdKey]]
    # Identify notes which are currently CRH or NYH.
    currentCRHNotes = currentLabels[
      currentLabels[statusColumn].isin({c.currentlyRatedHelpful, c.needsYourHelp})
    ][[c.noteIdKey]]
    # Identify candidate note status updates
    noteStatusUpdates = probationaryNMRNotes.merge(currentCRHNotes, on=c.noteIdKey, how="inner")

    # Set note status and return
    noteStatusUpdates[statusColumn] = c.needsMoreRatings

    return (noteStatusUpdates, None)


class InsufficientExplanation(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    status: str,
    minRatingsToGetTag: int,
    minTagsNeededForStatus: int,
    tagsConsidered: Optional[List[str]] = None,
  ):
    """Set Top Tags, and set status to NMR for CRH / CRNH notes for which we don't have
        a strong enough explanation signal

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
      minRatingsToGetTag: min number occurrences to assign a tag to a note.
      minTagsNeededForStatus: min tags assigned before a note can be CRH/CRNH
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._minRatingsToGetTag = minRatingsToGetTag
    self._minTagsNeededForStatus = minTagsNeededForStatus
    self._tagsConsidered = tagsConsidered

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sets Top Tags inplace on noteStats,
    returns notes on track for CRH / CRNH with insufficient to receive NMR status."""
    noteStats[c.firstTagKey] = noteStats[c.firstTagKey].astype(object)
    noteStats[c.secondTagKey] = noteStats[c.secondTagKey].astype(object)

    if self._tagsConsidered is None:
      # Set Top CRH Tags
      crh_idx = noteStats[c.noteIdKey].isin(
        currentLabels.loc[currentLabels[statusColumn] == c.currentlyRatedHelpful, c.noteIdKey]
      )
      topCrhTags = get_top_two_tags_for_note(
        noteStats.loc[crh_idx, :],
        self._minTagsNeededForStatus,
        self._minRatingsToGetTag,
        c.helpfulTagsTiebreakOrder,
      )
      noteStats.set_index(c.noteIdKey, inplace=True)
      noteStats.loc[topCrhTags[c.noteIdKey], c.firstTagKey] = topCrhTags[c.firstTagKey]
      noteStats.loc[topCrhTags[c.noteIdKey], c.secondTagKey] = topCrhTags[c.secondTagKey]
      noteStats.reset_index(inplace=True)

      # Set Top CRNH Tags
      crnh_idx = noteStats[c.noteIdKey].isin(
        currentLabels.loc[currentLabels[statusColumn] == c.currentlyRatedNotHelpful, c.noteIdKey]
      )
      topCrnhTags = get_top_two_tags_for_note(
        noteStats.loc[crnh_idx, :],
        self._minRatingsToGetTag,
        self._minTagsNeededForStatus,
        c.notHelpfulTagsTiebreakOrder,
      )
      noteStats.set_index(c.noteIdKey, inplace=True)
      noteStats.loc[topCrnhTags[c.noteIdKey], c.firstTagKey] = topCrnhTags[c.firstTagKey]
      noteStats.loc[topCrnhTags[c.noteIdKey], c.secondTagKey] = topCrnhTags[c.secondTagKey]
      noteStats.reset_index(inplace=True)
    else:
      topTags = get_top_two_tags_for_note(
        noteStats,
        self._minRatingsToGetTag,
        self._minTagsNeededForStatus,
        self._tagsConsidered,
      )
      noteStats.loc[:, c.firstTagKey] = topTags[c.firstTagKey]
      noteStats.loc[:, c.secondTagKey] = topTags[c.secondTagKey]

    noteStats[c.firstTagKey] = noteStats[c.firstTagKey].astype(object)
    noteStats[c.secondTagKey] = noteStats[c.secondTagKey].astype(object)

    with c.time_block("Insufficient explanation: post-processing"):
      # Prune noteStats to only include CRH / CRNH notes.
      crNotes = currentLabels[
        (currentLabels[statusColumn] == c.currentlyRatedHelpful)
        | (currentLabels[statusColumn] == c.currentlyRatedNotHelpful)
      ][[c.noteIdKey]]
      crStats = noteStats.merge(crNotes, on=c.noteIdKey, how="inner")
      logger.info(
        f"CRH / CRNH notes prior to filtering for insufficient explanation: {len(crStats)}"
      )

      # Identify impacted notes.
      noteStatusUpdates = crStats.loc[
        (~crStats[[c.firstTagKey, c.secondTagKey]].isna()).sum(axis=1)
        < self._minTagsNeededForStatus
      ][[c.noteIdKey]]

      pd.testing.assert_frame_equal(noteStatusUpdates, noteStatusUpdates.drop_duplicates())

      logger.info(f"Total notes impacted by explanation filtering: {len(noteStatusUpdates)}")
      noteStatusUpdates[statusColumn] = self._status

    return (noteStatusUpdates, None)


class NMtoCRNH(ScoringRule):
  def __init__(
    self, ruleID: RuleID, dependencies: Set[RuleID], status: str, crnhThresholdNMIntercept: float
  ):
    """Configure a ScoringRule to set low scoring, non-misleading notes to CRNH.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR).
      crnhThresholdNMIntercept: Intercept for setting notes on non-misleading tweets to CRNH.
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._crnhThresholdNMIntercept = crnhThresholdNMIntercept

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Returns noteIds for low scoring notes on non-misleading tweets."""
    noteStatusUpdates = noteStats.loc[
      (noteStats[c.internalNoteInterceptKey] < self._crnhThresholdNMIntercept)
      # Require that that the classification is "not misleading" to explicitly exclude deleted
      # notes where the classification is nan.
      & (noteStats[c.classificationKey] == c.noteSaysTweetIsNotMisleadingKey)
    ][[c.noteIdKey]]
    noteStatusUpdates[statusColumn] = self._status
    return (noteStatusUpdates, None)


class AddCRHInertia(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    status: str,
    threshold: float,
    expectedMax: float,
    minRatingsNeeded: int,
  ):
    """Scores notes as CRH contingent on whether the note is already CRH.

    This rule should be applied after other CRH scoring logic to add CRH status for
    notes with intercepts in a defined range (e.g. 0.01 below the general threshold)
    contingent on whether the note is currently rated as CRH on the BW site.  The
    objective of this rule is to decrease scoring changes due to small variations
    in note intercepts around the threshold.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
      threshold: minimum threshold for marking a note as CRH.
      expectedMax: raise an AssertionError if any note scores above this threshold.
      minRatingsNeeded: Minimum number of ratings for a note to have status.
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._threshold = threshold
    self._expectedMax = expectedMax
    self._minRatingsNeeded = minRatingsNeeded

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Returns noteIds for notes already have CRH status but now fall slightly below a threshold."""
    # This scoring only impacts notes which don't already have CRH status - there is no need to
    # act on notes that already have CRH status.
    noteIds = currentLabels[currentLabels[statusColumn] != c.currentlyRatedHelpful][[c.noteIdKey]]
    noteIds = noteIds.merge(
      noteStats.loc[
        # Must have minimum number of ratings to receive CRH status.
        (noteStats[c.numRatingsKey] >= self._minRatingsNeeded)
        # Score must exceed defined threshold for actionability.
        & (noteStats[c.internalNoteInterceptKey] >= self._threshold)
        # Note must have been rated CRH during the last scoring run.
        & (noteStats[c.currentLabelKey] == c.currentlyRatedHelpful)
        # Check for inequality with "not misleading" to include notes whose classificaiton
        # is nan (i.e. deleted notes).
        & (noteStats[c.classificationKey] != c.noteSaysTweetIsNotMisleadingKey)
      ][[c.noteIdKey]],
      on=c.noteIdKey,
      how="inner",
    )
    # Validate that all note scores were within the expected range
    noteIntercepts = noteStats.merge(noteIds, on=c.noteIdKey, how="inner")[
      c.internalNoteInterceptKey
    ]

    assert sum(noteIntercepts > self._expectedMax) == 0, f"""{sum(noteIntercepts > self._expectedMax)} notes (out of {len(noteIntercepts)}) had intercepts above expected maximum of {self._expectedMax}. 
      The highest was {max(noteIntercepts)}."""
    noteStatusUpdates = noteIds[[c.noteIdKey]]
    noteStatusUpdates[statusColumn] = self._status
    return (noteStatusUpdates, None)


class ScoringDriftGuard(ScoringRule):
  def __init__(self, ruleID: RuleID, dependencies: Set[RuleID], lockedStatus: pd.DataFrame):
    """Guards against drift in scoring over time by applying historical note status.

    If a locked status is available for a note, and the locked status diverges from
    the current status, this rule will set the status to the locked status and preserve
    the current status in an additional column.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      lockedStatus: pd.DataFrame containing {noteId, status} pairs for all notes
    """
    super().__init__(ruleID, dependencies)
    self._lockedStatus = lockedStatus

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Returns locked status when necessary to impact scoring outcomes."""
    # identify impacted notes where we need to change a label
    lockedStatusAvailable = ~pd.isna(self._lockedStatus[c.lockedStatusKey])
    lockedStatusDF = self._lockedStatus.loc[lockedStatusAvailable, [c.noteIdKey, c.lockedStatusKey]]
    mergedLabels = lockedStatusDF.merge(currentLabels, on=c.noteIdKey, how="inner")
    mergedLabels = mergedLabels.loc[mergedLabels[c.lockedStatusKey] != mergedLabels[statusColumn]]
    # prepare matrix with new note status
    noteStatusUpdates = pd.DataFrame(mergedLabels[[c.noteIdKey, c.lockedStatusKey]])
    noteStatusUpdates = noteStatusUpdates.rename(columns={c.lockedStatusKey: statusColumn})
    # save current labels so we can persist them in a new column
    unlockedStatus = mergedLabels[[c.noteIdKey, statusColumn]].copy()
    unlockedStatus = unlockedStatus.rename(columns={statusColumn: c.unlockedRatingStatusKey})
    return noteStatusUpdates, unlockedStatus


class ApplyTopicModelResult(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    topic: Topics,
    topicNMRInterceptThreshold: Optional[float] = 0.24,
    topicNMRFactorThreshold: Optional[float] = 0.51,
  ):
    """Set any note scored by a topic model to NMR if the note is presently CRH and has low topic intercept.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      topic: apply the rule to notes scored by a given topic model.
    """
    super().__init__(ruleID, dependencies)
    self._topic = topic
    self._topicNMRInterceptThreshold = topicNMRInterceptThreshold
    self._topicNMRFactorThreshold = topicNMRFactorThreshold

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Flip notes from CRH to NMR based on the topic model results."""
    # Identify notes which are currently CRH or NYH.
    currentCRHNotes = currentLabels[
      currentLabels[statusColumn].isin({c.currentlyRatedHelpful, c.needsYourHelp})
    ][[c.noteIdKey]]
    # Identify notes which are below threshld from the applicable topic model.
    topicLowNotes = noteStats[
      (
        (noteStats[c.topicNoteInterceptKey] < self._topicNMRInterceptThreshold)
        | (noteStats[c.topicNoteFactor1Key].abs() > self._topicNMRFactorThreshold)
      )
      & (noteStats[c.topicNoteConfidentKey])
      & (noteStats[c.noteTopicKey] == self._topic.name)
    ][[c.noteIdKey]]
    # Set note status for candidates and return
    noteStatusUpdates = currentCRHNotes.merge(topicLowNotes, on=c.noteIdKey, how="inner")
    noteStatusUpdates[statusColumn] = c.needsMoreRatings

    return (noteStatusUpdates, None)


def apply_scoring_rules(
  noteStats: pd.DataFrame,
  rules: List[ScoringRule],
  statusColumn: str,
  ruleColumn: str,
  decidedByColumn: Optional[str] = None,
) -> pd.DataFrame:
  """Apply a list of ScoringRules to note inputs and return noteStats augmented with scoring results.

  This function applies a list of ScoringRules in order.  Once each rule has run
  a final ratingStatus is set for each note. An additional column is added to capture
  which rules acted on the note and any additional columns generated by the ScoringRules
  are merged with the scored notes to generate the final return value.

  Args:
    noteStats: attributes, aggregates and raw scoring signals for each note.
    rules: ScoringRules which will be applied in the order given.
    statusColumn: str indicating column where status should be assigned.
    ruleColumn: str indicating column where active rules should be stored.
    decidedByColumn: None or str indicating column where the last rule to act on a note is stored.

  Returns:
    noteStats with additional columns representing scoring results.
  """
  # Initialize empty dataframes to store labels for each note and which rules impacted
  # scoring for each note.
  noteLabels = pd.DataFrame.from_dict(
    {c.noteIdKey: pd.Series([], dtype=np.int64), statusColumn: pd.Series([], dtype=object)}
  )
  noteRules = pd.DataFrame.from_dict(
    {c.noteIdKey: pd.Series([], dtype=np.int64), ruleColumn: pd.Series([], dtype=object)}
  )
  noteColumns = pd.DataFrame.from_dict({c.noteIdKey: pd.Series([], dtype=np.int64)})

  # Establish state to enforce rule dependencies.
  ruleIDs: Set[RuleID] = set()

  # Successively apply each rule
  for rule in rules:
    with c.time_block(f"Applying scoring rule: {rule.get_name()}"):
      logger.info(f"Applying scoring rule: {rule.get_name()}")
      rule.check_dependencies(ruleIDs)
      assert rule.get_rule_id() not in ruleIDs, f"repeat ruleID: {rule.get_name()}"
      ruleIDs.add(rule.get_rule_id())
      with c.time_block(f"Calling score_notes: {rule.get_name()}"):
        noteStatusUpdates, additionalColumns = rule.score_notes(noteStats, noteLabels, statusColumn)
      if (
        additionalColumns is not None
        # This rule updates both status and NmrDueToStableCrhTime (in additional column), they can
        # be on different rows.
        and rule.get_rule_id() != RuleID.NMR_DUE_TO_MIN_STABLE_CRH_TIME
      ):
        assert set(noteStatusUpdates[c.noteIdKey]) == set(additionalColumns[c.noteIdKey])

      # Update noteLabels, which will always hold at most one label per note.
      unsafeAllowed = {c.internalRatingStatusKey, c.finalRatingStatusKey, c.defaultIndexKey}
      noteLabels = (
        pd.concat([noteLabels, noteStatusUpdates], unsafeAllowed=unsafeAllowed)
        .groupby(c.noteIdKey)
        .tail(1)
      )
      # Update note rules to have one row per rule which was active for a note
      noteRules = pd.concat(
        [
          noteRules,
          pd.DataFrame.from_dict(
            {c.noteIdKey: noteStatusUpdates[c.noteIdKey], ruleColumn: rule.get_name()}
          ),
        ],
        unsafeAllowed={c.internalActiveRulesKey, c.defaultIndexKey, c.metaScorerActiveRulesKey},
      )
      if additionalColumns is not None:
        # Merge any additional columns into current set of new columns
        assert {c.noteIdKey} == (set(noteColumns.columns) & set(additionalColumns.columns))
        noteColumns = noteColumns.merge(
          additionalColumns, on=c.noteIdKey, how="outer", unsafeAllowed=c.defaultIndexKey
        )

  with c.time_block("Condense noteRules after applying all scoring rules"):
    # Having applied all scoring rules, condense noteRules to have one row per note representing
    # all of the ScoringRuless which were active for the note.
    noteRules = noteRules.groupby(c.noteIdKey).aggregate(list).reset_index()
    if decidedByColumn:
      noteRules[decidedByColumn] = [rules[-1] for rules in noteRules[ruleColumn]]
    noteRules[ruleColumn] = [",".join(activeRules) for activeRules in noteRules[ruleColumn]]
    # Validate that there are labels and assigned rules for each note
    assert set(noteStats[c.noteIdKey]) == set(noteLabels[c.noteIdKey])
    assert set(noteStats[c.noteIdKey]) == set(noteRules[c.noteIdKey])
    assert len(set(noteColumns[c.noteIdKey]) - set(noteStats[c.noteIdKey])) == 0
    # Merge note labels, active rules and new columns into noteStats to form scoredNotes
    scoredNotes = noteStats.merge(noteLabels, on=c.noteIdKey, how="inner")
    scoredNotes = scoredNotes.merge(noteRules, on=c.noteIdKey, how="inner")
    scoredNotes = scoredNotes.merge(noteColumns, on=c.noteIdKey, how="left")
    # Add all of the individual model rules to the active rules column
    assert len(scoredNotes) == len(noteStats)
    # Set boolean columns indicating scoring outcomes
    scoredNotes[c.currentlyRatedHelpfulBoolKey] = (
      scoredNotes[statusColumn] == c.currentlyRatedHelpful
    )
    scoredNotes[c.currentlyRatedNotHelpfulBoolKey] = (
      scoredNotes[statusColumn] == c.currentlyRatedNotHelpful
    )
    scoredNotes[c.awaitingMoreRatingsBoolKey] = scoredNotes[statusColumn] == c.needsMoreRatings

  # Return completed DF including original noteStats signals merged wtih scoring results
  return scoredNotes
