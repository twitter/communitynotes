from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
from typing import Callable, List, Optional, Set, Tuple

import constants as c, tag_filter
from explanation_tags import top_tags


import numpy as np
import pandas as pd


RuleAndVersion = namedtuple("RuleAndVersion", ["ruleName", "ruleVersion"])
"""namedtuple identifying ScoringRule with a name and tracking revisions with a version."""


class RuleID(Enum):
  """Each RuleID must have a unique ruleName and can be assigned to at most one ScoringRule."""

  INITIAL_NMR = RuleAndVersion("InitialNMR", "1.0")
  GENERAL_CRH = RuleAndVersion("GeneralCRH", "1.0")
  GENERAL_CRNH = RuleAndVersion("GeneralCRNH", "1.0")
  TAG_OUTLIER = RuleAndVersion("TagFilter", "1.0")
  NM_CRNH = RuleAndVersion("NmCRNH", "1.0")
  GENERAL_CRH_INERTIA = RuleAndVersion("GeneralCRHInertia", "1.0")
  ELEVATED_CRH_INERTIA = RuleAndVersion("ElevatedCRHInertia", "1.0")
  INSUFFICIENT_EXPLANATION = RuleAndVersion("InsufficientExplanation", "1.0")
  SCORING_DRIFT_GUARD = RuleAndVersion("ScoringDriftGuard", "1.0")


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
      status: valid ratingStatus to assign to notes where the ScoringRule is active.
      dependencies: Rules which must run before this rule can run.
    """
    self._ruleID = ruleID
    self._dependencies = dependencies

  def get_rule_id(self) -> RuleID:
    """Returns the RuleID uniquely identifying this ScoringRule."""
    return self._ruleID

  def get_name(self) -> str:
    """Returns a string combining the name and version to uniquely name the logic of the ScoringRule."""
    return f"{self._ruleID.value.ruleName} (v{self._ruleID.value.ruleVersion})"

  def check_dependencies(self, priorRules: Set[RuleID]) -> None:
    """Raise an AssertionError if rule dependencies have not been satisfied."""
    assert not (self._dependencies - priorRules)

  @abstractmethod
  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame
  ) -> (Tuple[pd.Series, Optional[pd.DataFrame]]):
    """Identify which notes the ScoringRule should be active for, and any new columns to add for those notes.

    Args:
      noteStats: Raw note attributes, scoring signals and attirbutes for notes.
      currentLabels: the ratingStatus assigned to each note from prior ScoringRules.

    Returns:
      Tuple[0]: DF specifying note IDs and associated status which this rule will update.
      Tuple[1]: DF containing noteIDs and any new columns to add to output
    """


class DefaultRule(ScoringRule):
  def __init__(self, ruleID: RuleID, dependencies: Set[RuleID], status: str):
    """Creates a ScoringRule which sets all note statuses to a default value.

    Args:
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
    """
    super().__init__(ruleID, dependencies)
    self._status = status

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame
  ) -> (Tuple[pd.Series, Optional[pd.DataFrame]]):
    """Returns all noteIDs to initialize all note ratings to a default status (e.g. NMR)."""
    noteStatusUpdates = pd.DataFrame(noteStats[[c.noteIdKey]])
    noteStatusUpdates[c.ratingStatusKey] = self._status
    return (noteStatusUpdates, None)


class RuleFromFunction(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    status: str,
    function: Callable[[pd.DataFrame], pd.Series],
  ):
    """Creates a ScoringRule which wraps a boolean function.

    Args:
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
      function: accepts noteStats as input and returns a boolean pd.Series corresponding to
        rows matched by the function.  For example, a valid function would be:
        "lambda noteStats: noteStats[c.noteInterceptKey] > 0.4"
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._function = function

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame
  ) -> (Tuple[pd.Series, Optional[pd.DataFrame]]):
    """Returns noteIDs for notes matched by the boolean function."""
    noteStatusUpdates = noteStats.loc[
      self._function(noteStats)
      # Check for inequality with "not misleading" to include notes whose classificaiton
      # is nan (i.e. deleted notes).
      & (noteStats[c.classificationKey] != c.noteSaysTweetIsNotMisleadingKey)
    ][[c.noteIdKey]]
    noteStatusUpdates[c.ratingStatusKey] = self._status
    return (noteStatusUpdates, None)


class FilterTagOutliers(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    status: str,
    tagRatioPercentile: int,
    minAdjustedTotal: float,
    crhSuperThreshold: float,
  ):
    """Filter CRH notes for outliers with high levels of any particular tag.

    Args:
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
      tagRatioPercentile: For a filter to trigger, the adjusted ratio value for a
        tag must exceed Nth percentile for notes currently rated as CRH.
      minAdjustedTotal: For a filter to trigger, the adjusted total of a tag must
        exceed the minAdjustedTotal.
      crhSuperThreshold: If the note intercept exceeds the crhSuperThreshold, then the
        tag filter is disabled.
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._tagRatioPercentile = tagRatioPercentile
    self._minAdjustedTotal = minAdjustedTotal
    self._crhSuperThreshold = crhSuperThreshold

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame
  ) -> (Tuple[pd.Series, pd.DataFrame]):
    """Returns notes on track for CRH with high levels of any tag to receive NMR status."""
    # Prune noteStats to only include CRH notes.
    crhNotes = currentLabels[currentLabels[c.ratingStatusKey] == c.currentlyRatedHelpful][
      [c.noteIdKey]
    ]
    crhStats = noteStats.merge(crhNotes, on=c.noteIdKey, how="inner")
    print(f"CRH notes prior to tag filtering: {len(crhStats)}")
    print(
      f"CRH notes above crhSuperThreshold: {sum(crhStats[c.noteInterceptKey] > self._crhSuperThreshold)}"
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
      if tag == c.notHelpfulHardToUnderstandKey or tag == c.notHelpfulNoteNotNeededKey:
        print(f"outlier filtering disabled for tag: {tag}")
        continue
      tagFilteredNotes = crhStats[
        # Adjusted total must pass minimum threhsold set across all tags.
        (crhStats[adjustedColumn] > self._minAdjustedTotal)
        # Adjusted ratio must exceed percentile based total for this specific tag.
        & (crhStats[adjustedRatioColumn] > thresholds[adjustedRatioColumn])
        # Note intercept must be lower than crhSuperThreshold which overrides tag filter.
        & (crhStats[c.noteInterceptKey] < self._crhSuperThreshold)
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
    noteStatusUpdates[c.ratingStatusKey] = self._status
    return (noteStatusUpdates, impactedNotes)


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
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame
  ) -> (Tuple[pd.Series, pd.DataFrame]):
    """Sets Top Tags, returns notes on track for CRH / CRNH with insufficient to receive NMR status."""

    if self._tagsConsidered is None:
      # Set Top Tags
      crh_idx = noteStats[c.noteIdKey].isin(
        currentLabels.loc[currentLabels[c.ratingStatusKey] == c.currentlyRatedHelpful, c.noteIdKey]
      )
      noteStats.loc[crh_idx, :] = noteStats.loc[crh_idx, :].apply(
        lambda row: top_tags(
          row, self._minRatingsToGetTag, self._minTagsNeededForStatus, c.helpfulTagsTiebreakOrder
        ),
        axis=1,
      )
      crnh_idx = noteStats[c.noteIdKey].isin(
        currentLabels.loc[
          currentLabels[c.ratingStatusKey] == c.currentlyRatedNotHelpful, c.noteIdKey
        ]
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

    # Prune noteStats to only include CRH / CRNH notes.
    crNotes = currentLabels[
      (currentLabels[c.ratingStatusKey] == c.currentlyRatedHelpful)
      | (currentLabels[c.ratingStatusKey] == c.currentlyRatedNotHelpful)
    ][[c.noteIdKey]]
    crStats = noteStats.merge(crNotes, on=c.noteIdKey, how="inner")
    print(f"CRH / CRNH notes prior to filtering for insufficient explanation: {len(crStats)}")

    # Identify impacted notes.
    noteStatusUpdates = crStats.loc[
      (~crStats[[c.firstTagKey, c.secondTagKey]].isna()).sum(axis=1) < self._minTagsNeededForStatus
    ][[c.noteIdKey]]

    pd.testing.assert_frame_equal(noteStatusUpdates, noteStatusUpdates.drop_duplicates())

    print(f"Total notes impacted by explanation filtering: {len(noteStatusUpdates)}")
    noteStatusUpdates[c.ratingStatusKey] = self._status

    return (noteStatusUpdates, None)


class NMtoCRNH(ScoringRule):
  def __init__(self, ruleID: RuleID, dependencies: Set[RuleID], status: str):
    """Configure a ScoringRule to set low scoring, non-misleading notes to CRNH.

    Args:
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
    """
    super().__init__(ruleID, dependencies)
    self._status = status

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame
  ) -> (Tuple[pd.Series, Optional[pd.DataFrame]]):
    """Returns noteIds for low scoring notes on non-misleading tweets."""
    noteStatusUpdates = noteStats.loc[
      (noteStats[c.noteInterceptKey] < c.crnhThresholdNMIntercept)
      # Require that that the classification is "not misleading" to explicitly exclude deleted
      # notes where the classification is nan.
      & (noteStats[c.classificationKey] == c.noteSaysTweetIsNotMisleadingKey)
    ][[c.noteIdKey]]
    noteStatusUpdates[c.ratingStatusKey] = self._status
    return (noteStatusUpdates, None)


class AddCRHInertia(ScoringRule):
  def __init__(
    self,
    ruleID: RuleID,
    dependencies: Set[RuleID],
    status: str,
    threshold: float,
    expectedMax: float,
  ):
    """Scores notes as CRH contingent on whether the note is already CRH.

    This rule should be applied after other CRH scoring logic to add CRH status for
    notes with intercepts in a defined range (e.g. 0.01 below the general threshold)
    contingent on whether the note is currently rated as CRH on the BW site.  The
    objective of this rule is to decrease scoring changes due to small variations
    in note intercepts around the threshold.

    Args:
      status: the status which each note should be set to (e.g. CRH, CRNH, NMR)
      threshold: minimum threshold for marking a note as CRH.
      expectedMax: raise an AssertionError if any note scores above this threshold.
    """
    super().__init__(ruleID, dependencies)
    self._status = status
    self._threshold = threshold
    self._expectedMax = expectedMax

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame
  ) -> (Tuple[pd.Series, Optional[pd.DataFrame]]):
    """Returns noteIds for notes already have CRH status but now fall slightly below a threshold."""
    # This scoring only impacts notes which don't already have CRH status - there is no need to
    # act on notes that already have CRH status.
    noteIds = currentLabels[currentLabels[c.ratingStatusKey] != c.currentlyRatedHelpful][
      [c.noteIdKey]
    ]
    noteIds = noteIds.merge(
      noteStats.loc[
        # Must have minimum number of ratings to receive CRH status.
        (noteStats[c.numRatingsKey] >= c.minRatingsNeeded)
        # Score must exceed defined threshold for actionability.
        & (noteStats[c.noteInterceptKey] >= self._threshold)
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
    noteIntercepts = noteStats.merge(noteIds, on=c.noteIdKey, how="inner")[c.noteInterceptKey]
    assert sum(noteIntercepts > self._expectedMax) == 0, "note exceeded expected maximum"
    noteStatusUpdates = noteIds[[c.noteIdKey]]
    noteStatusUpdates[c.ratingStatusKey] = self._status
    return (noteStatusUpdates, None)


class ScoringDriftGuard(ScoringRule):
  def __init__(self, ruleID: RuleID, dependencies: Set[RuleID]):
    """Guards against drift in scoring over time by applying historical note status.

    If a locked status is available for a note, and the locked status diverges from
    the current status, this rule will set the status to the locked status and preserve
    the current status in an additional column.
    """
    super().__init__(ruleID, dependencies)

  def score_notes(
    self, noteStats: pd.DataFrame, currentLabels: pd.DataFrame
  ) -> (Tuple[pd.Series, Optional[pd.DataFrame]]):
    """Returns locked status when necessary to impact scoring outcomes."""
    # identify impacted notes where we need to change a label
    lockedStatusAvailable = ~pd.isna(noteStats[c.lockedStatusKey])
    lockedStatusDF = noteStats.loc[lockedStatusAvailable, [c.noteIdKey, c.lockedStatusKey]]
    mergedLabels = lockedStatusDF.merge(currentLabels, on=c.noteIdKey, how="inner")
    mergedLabels = mergedLabels.loc[
      mergedLabels[c.lockedStatusKey] != mergedLabels[c.ratingStatusKey]
    ]
    # prepare matrix with new note status
    noteStatusUpdates = pd.DataFrame(mergedLabels[[c.noteIdKey, c.lockedStatusKey]])
    noteStatusUpdates = noteStatusUpdates.rename(columns={c.lockedStatusKey: c.ratingStatusKey})
    # save current labels so we can persist them in a new column
    unlockedStatus = mergedLabels[[c.noteIdKey, c.ratingStatusKey]].copy()
    unlockedStatus = unlockedStatus.rename(columns={c.ratingStatusKey: c.unlockedRatingStatusKey})
    return noteStatusUpdates, unlockedStatus


def apply_scoring_rules(noteStats: pd.DataFrame, rules: List[ScoringRule]) -> pd.DataFrame:
  """Apply a list of ScoringRules to note inputs and return noteStats augmented with scoring results.

  This function applies a list of ScoringRules in order.  Once each rule has run
  a final ratingStatus is set for each note. An additional column is added to capture
  which rules acted on the note and any additional columns generated by the ScoringRules
  are merged with the scored notes to generate the final return value.

  Args:
    noteStats: attributes, aggregates and raw scoring signals for each note.
    rules: ScoringRules which will be applied in the order given.

  Returns:
    noteStats with additional columns representing scoring results.
  """
  # Initialize empty dataframes to store labels for each note and which rules impacted
  # scoring for each note.
  noteLabels = pd.DataFrame.from_dict({c.noteIdKey: [], c.ratingStatusKey: []}).astype(
    {c.noteIdKey: np.int64}
  )
  noteRules = pd.DataFrame.from_dict({c.noteIdKey: [], c.activeRulesKey: []}).astype(
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
    noteStatusUpdates, additionalColumns = rule.score_notes(noteStats, noteLabels)
    if additionalColumns is not None:
      assert set(noteStatusUpdates[c.noteIdKey]) == set(additionalColumns[c.noteIdKey])
    # Update noteLabels, which will always hold at most one label per note.
    noteLabels = pd.concat([noteLabels, noteStatusUpdates]).groupby(c.noteIdKey).tail(1)
    # Update note rules to have one row per rule which was active for a note
    noteRules = pd.concat(
      [
        noteRules,
        pd.DataFrame.from_dict(
          {c.noteIdKey: noteStatusUpdates[c.noteIdKey], c.activeRulesKey: rule.get_name()}
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
  noteRules[c.activeRulesKey] = [
    ",".join(activeRules) for activeRules in noteRules[c.activeRulesKey]
  ]
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
  scoredNotes[c.currentlyRatedHelpfulBoolKey] = (
    scoredNotes[c.ratingStatusKey] == c.currentlyRatedHelpful
  )
  scoredNotes[c.currentlyRatedNotHelpfulBoolKey] = (
    scoredNotes[c.ratingStatusKey] == c.currentlyRatedNotHelpful
  )
  scoredNotes[c.awaitingMoreRatingsBoolKey] = scoredNotes[c.ratingStatusKey] == c.needsMoreRatings
  # Return completed DF including original noteStats signals merged wtih scoring results
  return scoredNotes
