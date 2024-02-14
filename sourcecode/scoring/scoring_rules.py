from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
from typing import Callable, List, Optional, Set, Tuple

from . import constants as c, tag_filter
from .explanation_tags import top_tags

import numpy as np
import pandas as pd


RuleAndVersion = namedtuple("RuleAndVersion", ["ruleName", "ruleVersion", "lockingEnabled"])
"""namedtuple identifying ScoringRule with a name and tracking revisions with a version."""


class RuleID(Enum):
  """Each RuleID must have a unique ruleName and can be assigned to at most one ScoringRule."""

  # Rules used by matrix_factorization_scorer.
  INITIAL_NMR = RuleAndVersion("InitialNMR", "1.0", False)
  GENERAL_CRH = RuleAndVersion("GeneralCRH", "1.0", False)
  GENERAL_CRNH = RuleAndVersion("GeneralCRNH", "1.0", False)
  UCB_CRNH = RuleAndVersion("UcbCRNH", "1.0", False)
  TAG_OUTLIER = RuleAndVersion("TagFilter", "1.0", False)
  ELEVATED_CRH = RuleAndVersion("CRHSuperThreshold", "1.0", False)
  NM_CRNH = RuleAndVersion("NmCRNH", "1.0", False)
  GENERAL_CRH_INERTIA = RuleAndVersion("GeneralCRHInertia", "1.0", False)
  ELEVATED_CRH_INERTIA = RuleAndVersion("ElevatedCRHInertia", "1.0", False)
  INCORRECT_OUTLIER = RuleAndVersion("FilterIncorrect", "1.0", False)
  LOW_DILIGENCE = RuleAndVersion("FilterLowDiligence", "1.0", False)
  LARGE_FACTOR = RuleAndVersion("FilterLargeFactor", "1.0", False)

  # Rules used in _meta_score.
  META_INITIAL_NMR = RuleAndVersion("MetaInitialNMR", "1.0", False)
  EXPANSION_MODEL = RuleAndVersion("ExpansionModel", "1.1", False)
  EXPANSION_PLUS_MODEL = RuleAndVersion("ExpansionPlusModel", "1.1", False)
  CORE_MODEL = RuleAndVersion("CoreModel", "1.1", True)
  COVERAGE_MODEL = RuleAndVersion("CoverageModel", "1.1", False)
  GROUP_MODEL_1 = RuleAndVersion("GroupModel01", "1.1", False)
  GROUP_MODEL_2 = RuleAndVersion("GroupModel02", "1.1", False)
  GROUP_MODEL_3 = RuleAndVersion("GroupModel03", "1.1", False)
  GROUP_MODEL_4 = RuleAndVersion("GroupModel04", "1.1", False)
  GROUP_MODEL_5 = RuleAndVersion("GroupModel05", "1.1", False)
  GROUP_MODEL_6 = RuleAndVersion("GroupModel06", "1.1", False)
  GROUP_MODEL_7 = RuleAndVersion("GroupModel07", "1.1", False)
  GROUP_MODEL_8 = RuleAndVersion("GroupModel08", "1.1", False)
  GROUP_MODEL_9 = RuleAndVersion("GroupModel09", "1.1", False)
  GROUP_MODEL_10 = RuleAndVersion("GroupModel10", "1.1", False)
  GROUP_MODEL_11 = RuleAndVersion("GroupModel11", "1.1", False)
  GROUP_MODEL_12 = RuleAndVersion("GroupModel12", "1.1", False)
  GROUP_MODEL_13 = RuleAndVersion("GroupModel13", "1.1", False)
  GROUP_MODEL_14 = RuleAndVersion("GroupModel14", "1.1", False)
  INSUFFICIENT_EXPLANATION = RuleAndVersion("InsufficientExplanation", "1.0", True)
  SCORING_DRIFT_GUARD = RuleAndVersion("ScoringDriftGuard", "1.0", False)

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
  ):
    """Propagate the note status from sourceColumn when the status is not NaN.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      sourceColumn: column containing note status (CRH, CRNH, NMR) to propagate to output
    """
    super().__init__(ruleID, dependencies)
    self._sourceColumn = sourceColumn

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Propagates any status set in sourceColumn when it is non-NaN."""
    notesWithStatus = ~noteStats[self._sourceColumn].isna()
    assert (
      noteStats.loc[notesWithStatus, self._sourceColumn]
      .isin({c.currentlyRatedHelpful, c.currentlyRatedNotHelpful, c.needsMoreRatings})
      .all()
    ), "status must be set to CRH, CRNH or NMR"
    noteStatusUpdates = noteStats[notesWithStatus][[c.noteIdKey, self._sourceColumn]].rename(
      columns={self._sourceColumn: statusColumn}
    )
    return (noteStatusUpdates, None)


class FilterTagOutliers(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    status: str,
    crhSuperThreshold: float,
    minAdjustedTotal: float = 2.5,
    tagRatioPercentile: int = 95,
  ):
    """Filter CRH notes for outliers with high levels of any particular tag.

    Args:
      rule: enum corresponding to a namedtuple defining a rule name and version string for the ScoringRule.
      dependencies: Rules which must run before this rule can run.
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
      crhSuperThreshold: If the note intercept exceeds the crhSuperThreshold, then the
        tag filter is disabled.
      tagRatioPercentile: For a filter to trigger, the adjusted ratio value for a
        tag must exceed Nth percentile for notes currently rated as CRH.
      minAdjustedTotal: For a filter to trigger, the adjusted total of a tag must
        exceed the minAdjustedTotal.
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._tagRatioPercentile = tagRatioPercentile
    self._minAdjustedTotal = minAdjustedTotal
    self._crhSuperThreshold = crhSuperThreshold

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns notes on track for CRH with high levels of any tag to receive NMR status."""
    # Prune noteStats to only include CRH notes.
    crhNotes = currentLabels[currentLabels[statusColumn] == c.currentlyRatedHelpful][[c.noteIdKey]]
    crhStats = noteStats.merge(crhNotes, on=c.noteIdKey, how="inner")
    print(f"CRH notes prior to tag filtering: {len(crhStats)}")
    print(
      f"CRH notes above crhSuperThreshold: {sum(crhStats[c.internalNoteInterceptKey] > self._crhSuperThreshold)}"
    )
    # Identify impacted notes.
    thresholds = tag_filter.get_tag_thresholds(crhStats, self._tagRatioPercentile)
    impactedNotes = pd.DataFrame.from_dict({c.noteIdKey: [], c.activeFilterTagsKey: []}).astype(
      {c.noteIdKey: np.int64}
    )
    print("Checking note tags:")
    for tag in c.notHelpfulTagsTSVOrder:
      adjustedColumn = f"{tag}{c.adjustedSuffix}"
      adjustedRatioColumn = f"{adjustedColumn}{c.ratioSuffix}"
      print(tag)
      print(f"  ratio threshold: {thresholds[adjustedRatioColumn]}")
      if tag == c.notHelpfulHardToUnderstandKey:
        print(f"outlier filtering disabled for tag: {tag}")
        continue
      tagFilteredNotes = crhStats[
        # Adjusted total must pass minimum threhsold set across all tags.
        (crhStats[adjustedColumn] > self._minAdjustedTotal)
        # Adjusted ratio must exceed percentile based total for this specific tag.
        & (crhStats[adjustedRatioColumn] > thresholds[adjustedRatioColumn])
      ][c.noteIdKey]
      impactedNotes = pd.concat(
        [impactedNotes, pd.DataFrame({c.noteIdKey: tagFilteredNotes, c.activeFilterTagsKey: tag})]
      )
    # log and consolidate imapcted notes
    print(f"Total {{note, tag}} pairs where tag filter logic triggered: {len(impactedNotes)}")
    impactedNotes = impactedNotes.groupby(c.noteIdKey).aggregate(list).reset_index()
    impactedNotes[c.activeFilterTagsKey] = [
      ",".join(tags) for tags in impactedNotes[c.activeFilterTagsKey]
    ]
    print(f"Total unique notes impacted by tag filtering: {len(impactedNotes)}")
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
    superThreshold: Optional[float],
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
      superThreshold: if set, allow notes with an intercept above threshold to bypass the filter.
      colSuffix: string suffix to apply to lookup columns
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._tagThreshold = tagThreshold
    self._voteThreshold = voteThreshold
    self._weightedTotalVotes = weightedTotalVotes
    self._superThreshold = superThreshold

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns notes on track for CRH with high levels of any tag to receive NMR status."""
    # Prune noteStats to only include CRH notes.
    crhNotes = currentLabels[currentLabels[statusColumn] == c.currentlyRatedHelpful][[c.noteIdKey]]
    crhStats = noteStats.merge(crhNotes, on=c.noteIdKey, how="inner")

    # Identify impacted notes.
    noteStatusUpdates = crhStats.loc[
      (crhStats["notHelpfulIncorrect_interval"] >= self._tagThreshold)
      & (crhStats["num_voters_interval"] >= self._voteThreshold)
      & (crhStats["tf_idf_incorrect_interval"] >= self._weightedTotalVotes)
      & (
        True
        if self._superThreshold is None
        else crhStats[c.internalNoteInterceptKey] < self._superThreshold
      )
    ][[c.noteIdKey]]

    pd.testing.assert_frame_equal(noteStatusUpdates, noteStatusUpdates.drop_duplicates())

    print(f"Total notes impacted by incorrect filtering: {len(noteStatusUpdates)}")
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
    # Prune noteStats to only include CRH notes.
    crhNotes = currentLabels[currentLabels[statusColumn] == c.currentlyRatedHelpful][[c.noteIdKey]]
    crhStats = noteStats.merge(crhNotes, on=c.noteIdKey, how="inner")

    # Identify impacted notes.
    noteStatusUpdates = crhStats.loc[
      crhStats[c.lowDiligenceInterceptKey] > self._interceptThreshold
    ][[c.noteIdKey]]

    pd.testing.assert_frame_equal(noteStatusUpdates, noteStatusUpdates.drop_duplicates())

    print(f"Total notes impacted by low diligence filtering: {len(noteStatusUpdates)}")
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
    # Prune noteStats to only include CRH notes.
    crhNotes = currentLabels[currentLabels[statusColumn] == c.currentlyRatedHelpful][[c.noteIdKey]]
    crhStats = noteStats.merge(crhNotes, on=c.noteIdKey, how="inner")

    # Identify impacted notes.
    noteStatusUpdates = crhStats.loc[
      crhStats[c.internalNoteFactor1Key].abs() > self._factorThreshold
    ][[c.noteIdKey]]

    pd.testing.assert_frame_equal(noteStatusUpdates, noteStatusUpdates.drop_duplicates())

    print(f"Total notes impacted by large factor filtering: {len(noteStatusUpdates)}")
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
    minSafeguardThreshold: Optional[float] = 0.3,
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
    if self._minSafeguardThreshold is None:
      assert self._coreCrhThreshold is None
      assert self._expansionCrhThreshold is None
    else:
      assert self._coreCrhThreshold is not None
      assert self._expansionCrhThreshold is not None

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Flip notes from NMR to CRH based on group models and subject to core/expansion model safeguards."""
    # Identify notes which were CRH from the applicable group model.
    probationaryCRHNotes = noteStats[
      (noteStats[c.groupRatingStatusKey] == c.currentlyRatedHelpful)
      & (noteStats[c.modelingGroupKey] == self._groupNumber)
    ][[c.noteIdKey]]
    # Identify notes which are currently NMR.
    currentNMRNotes = currentLabels[currentLabels[statusColumn] == c.needsMoreRatings][
      [c.noteIdKey]
    ]
    # Identify candidate note status updates
    noteStatusUpdates = probationaryCRHNotes.merge(currentNMRNotes, on=c.noteIdKey, how="inner")
    # If necessary, identify notes which pass score bound checks for expansion and core models.
    if self._minSafeguardThreshold is not None:
      # Apply min and max threhsolds to core and expansion intercepts
      noteStats = noteStats[
        [c.noteIdKey, c.coreNoteInterceptKey, c.expansionNoteInterceptKey]
      ].copy()
      noteStats["core"] = (noteStats[c.coreNoteInterceptKey] < self._coreCrhThreshold) & (
        noteStats[c.coreNoteInterceptKey] > self._minSafeguardThreshold
      )
      noteStats.loc[noteStats[c.coreNoteInterceptKey].isna(), "core"] = np.nan
      noteStats["expansion"] = (
        noteStats[c.expansionNoteInterceptKey] < self._expansionCrhThreshold
      ) & (noteStats[c.expansionNoteInterceptKey] > self._minSafeguardThreshold)
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

      noteStats["actionable"] = noteStats[["core", "expansion"]].apply(_get_value, axis=1)

      # Filter set of note status updates to only include actionable notes
      actionableNotes = noteStats[noteStats["actionable"]][[c.noteIdKey]]
      noteStatusUpdates = noteStatusUpdates.merge(actionableNotes, on=c.noteIdKey, how="inner")

    # Set note status and return
    noteStatusUpdates[statusColumn] = c.currentlyRatedHelpful
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
      tagsConsidered: set of tags to consider for *all* notes.
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._minRatingsToGetTag = minRatingsToGetTag
    self._minTagsNeededForStatus = minTagsNeededForStatus
    self._tagsConsidered = tagsConsidered

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame, statusColumn: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sets Top Tags, returns notes on track for CRH / CRNH with insufficient to receive NMR status."""

    if self._tagsConsidered is None:
      # Set Top Tags
      crh_idx = noteStats[c.noteIdKey].isin(
        currentLabels.loc[currentLabels[statusColumn] == c.currentlyRatedHelpful, c.noteIdKey]
      )
      noteStats.loc[crh_idx, :] = noteStats.loc[crh_idx, :].apply(
        lambda row: top_tags(
          row, self._minRatingsToGetTag, self._minTagsNeededForStatus, c.helpfulTagsTiebreakOrder
        ),
        axis=1,
      )
      crnh_idx = noteStats[c.noteIdKey].isin(
        currentLabels.loc[currentLabels[statusColumn] == c.currentlyRatedNotHelpful, c.noteIdKey]
      )
      noteStats.loc[crnh_idx, :] = noteStats.loc[crnh_idx, :].apply(
        lambda row: top_tags(
          row, self._minRatingsToGetTag, self._minTagsNeededForStatus, c.notHelpfulTagsTiebreakOrder
        ),
        axis=1,
      )
    else:
      noteStats = noteStats.apply(
        lambda row: top_tags(
          row, self._minRatingsToGetTag, self._minTagsNeededForStatus, self._tagsConsidered
        ),
        axis=1,
      )
    # For unclear reasons, the "apply" above converts the noteId column to a float.  This cast
    # guarantees that the type of the noteId column remains int64.  Note that the cast will fail
    # if the noteId column includes nan values.
    #
    # See links below for more context:
    # https://stackoverflow.com/questions/40251948/stop-pandas-from-converting-int-to-float-due-to-an-insertion-in-another-column
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.convert_dtypes.html
    noteStats[c.noteIdKey] = noteStats[c.noteIdKey].astype(np.int64)

    # Prune noteStats to only include CRH / CRNH notes.
    crNotes = currentLabels[
      (currentLabels[statusColumn] == c.currentlyRatedHelpful)
      | (currentLabels[statusColumn] == c.currentlyRatedNotHelpful)
    ][[c.noteIdKey]]
    crStats = noteStats.merge(crNotes, on=c.noteIdKey, how="inner")
    print(f"CRH / CRNH notes prior to filtering for insufficient explanation: {len(crStats)}")

    # Identify impacted notes.
    noteStatusUpdates = crStats.loc[
      (~crStats[[c.firstTagKey, c.secondTagKey]].isna()).sum(axis=1) < self._minTagsNeededForStatus
    ][[c.noteIdKey]]

    pd.testing.assert_frame_equal(noteStatusUpdates, noteStatusUpdates.drop_duplicates())

    print(f"Total notes impacted by explanation filtering: {len(noteStatusUpdates)}")
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
  noteLabels = pd.DataFrame.from_dict({c.noteIdKey: [], statusColumn: []}).astype(
    {c.noteIdKey: np.int64}
  )
  noteRules = pd.DataFrame.from_dict({c.noteIdKey: [], ruleColumn: []}).astype(
    {c.noteIdKey: np.int64}
  )
  noteColumns = pd.DataFrame.from_dict({c.noteIdKey: []}).astype({c.noteIdKey: np.int64})
  # Establish state to enforce rule dependencies.
  ruleIDs: Set[RuleID] = set()

  # Successively apply each rule
  for rule in rules:
    print(f"Applying scoring rule: {rule.get_name()}")
    rule.check_dependencies(ruleIDs)
    assert rule.get_rule_id() not in ruleIDs, f"repeat ruleID: {rule.get_name()}"
    ruleIDs.add(rule.get_rule_id())
    noteStatusUpdates, additionalColumns = rule.score_notes(noteStats, noteLabels, statusColumn)
    if additionalColumns is not None:
      assert set(noteStatusUpdates[c.noteIdKey]) == set(additionalColumns[c.noteIdKey])
    # Update noteLabels, which will always hold at most one label per note.
    noteLabels = pd.concat([noteLabels, noteStatusUpdates]).groupby(c.noteIdKey).tail(1)
    # Update note rules to have one row per rule which was active for a note
    noteRules = pd.concat(
      [
        noteRules,
        pd.DataFrame.from_dict(
          {c.noteIdKey: noteStatusUpdates[c.noteIdKey], ruleColumn: rule.get_name()}
        ),
      ]
    )
    # Merge any additional columns into current set of new columns
    if additionalColumns is not None:
      assert {c.noteIdKey} == (set(noteColumns.columns) & set(additionalColumns.columns))
      noteColumns = noteColumns.merge(additionalColumns, on=c.noteIdKey, how="outer")

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
  assert len(scoredNotes) == len(noteStats)
  # Set boolean columns indicating scoring outcomes
  scoredNotes[c.currentlyRatedHelpfulBoolKey] = scoredNotes[statusColumn] == c.currentlyRatedHelpful
  scoredNotes[c.currentlyRatedNotHelpfulBoolKey] = (
    scoredNotes[statusColumn] == c.currentlyRatedNotHelpful
  )
  scoredNotes[c.awaitingMoreRatingsBoolKey] = scoredNotes[statusColumn] == c.needsMoreRatings

  # Return completed DF including original noteStats signals merged wtih scoring results
  return scoredNotes
