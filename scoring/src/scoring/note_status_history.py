import logging
import time
from typing import Optional, Tuple

from . import constants as c
from .scoring_rules import RuleID

import numpy as np
import pandas as pd


logger = logging.getLogger("birdwatch.note_status_history")
logger.setLevel(logging.INFO)


def merge_note_info(oldNoteStatusHistory: pd.DataFrame, notes: pd.DataFrame) -> pd.DataFrame:
  """Add the creation time and authorId of notes to noteStatusHistory.

  Useful when you have some new notes; called as a pre-processing step.  Note that oldNoteStatusHistory
  is expected to consistently contain noteIds which are not in notes due to deletions, and notes
  *may* contain notes which are not in noteStatusHistory if new notes have been written.

  Args:
      oldNoteStatusHistory (pd.DataFrame)
      notes (pd.DataFrame)

  Returns:
      pd.DataFrame: noteStatusHistory
  """
  noteSuffix = "_notes"
  newNoteStatusHistory = oldNoteStatusHistory.merge(
    notes[[c.noteIdKey, c.createdAtMillisKey, c.noteAuthorParticipantIdKey, c.classificationKey]],
    on=c.noteIdKey,
    # use outer so we don't drop deleted notes from "oldNoteStatusHistory" or new notes from "notes"
    how="outer",
    suffixes=("", noteSuffix),
    unsafeAllowed={c.createdAtMillisKey, c.createdAtMillisKey + noteSuffix},
  )
  newNotes = pd.isna(newNoteStatusHistory[c.createdAtMillisKey])
  logger.info(f"total notes added to noteStatusHistory: {sum(newNotes)}")
  # Copy timestamp and authorship data over for new notes.
  newNoteStatusHistory.loc[newNotes, c.createdAtMillisKey] = newNoteStatusHistory.loc[
    newNotes, c.createdAtMillisKey + noteSuffix
  ]
  newNoteStatusHistory.loc[newNotes, c.noteAuthorParticipantIdKey] = newNoteStatusHistory.loc[
    newNotes, c.noteAuthorParticipantIdKey + noteSuffix
  ]
  # Validate expectations that notes is a subset of noteStatusHistory, and that timestamp
  # and authorship data match when applicable.
  assert len(notes) == len(
    notes[[c.noteIdKey]].drop_duplicates()
  ), "notes must not contain duplicates"
  assert len(newNoteStatusHistory) == len(
    newNoteStatusHistory[[c.noteIdKey]].drop_duplicates()
  ), "noteStatusHistory must not contain duplicates"
  assert len(notes) == len(
    newNoteStatusHistory[[c.noteIdKey, c.createdAtMillisKey]].merge(
      notes[[c.noteIdKey, c.createdAtMillisKey]],
      on=[c.noteIdKey, c.createdAtMillisKey],
      how="inner",
      unsafeAllowed=c.createdAtMillisKey,
    )
  ), "timestamps from notes and noteStatusHistory must match"
  assert len(notes) == len(
    newNoteStatusHistory[[c.noteIdKey, c.noteAuthorParticipantIdKey]].merge(
      notes[[c.noteIdKey, c.noteAuthorParticipantIdKey]],
      on=[c.noteIdKey, c.noteAuthorParticipantIdKey],
      how="inner",
    )
  ), "authorship from notes and noteStatusHistory must match"

  # Drop cols which were artifacts of the merge
  noteStatusHistory = newNoteStatusHistory.drop(
    columns=[c.createdAtMillisKey + noteSuffix, c.noteAuthorParticipantIdKey + noteSuffix]
  )
  return noteStatusHistory


def _update_single_note_status_history(mergedNote, currentTimeMillis, newScoredNotesSuffix="_sn"):
  """For a single row (representing one note) in note status history, which contains fields
  merged between the old note status history and the note's current status, compute the new
  note's status history.

  Args:
      mergedNote: row of a pd.DataFrame, result of joining noteStatusHistory with scoredNotes
      currentTimeMillis: the timestamp to use for the note's new status in note_status_history
      newScoredNotesSuffix (str, optional): Defaults to "_sn". Merge suffix to distinguish
        fields from previous noteStatusHistory vs new note status.

  Returns:
      row of pd.DataFrame
  """
  # This TS will be set by run_combine_scoring_outputs.
  mergedNote[c.timestampMinuteOfFinalScoringOutput] = np.nan

  if mergedNote[c.finalRatingStatusKey] != mergedNote[c.currentLabelKey]:
    # Changed status vs. previous run:
    mergedNote[c.timestampMillisOfMostRecentStatusChangeKey] = currentTimeMillis
  else:
    # No change in status vs. previous run
    # If the note has not changed status (since the launch of this feature on 2024/07/02),
    # then the timestamp of the most recent status change should be set to -1 by default.
    if c.timestampMillisOfMostRecentStatusChangeKey not in mergedNote.index:
      mergedNote[c.timestampMillisOfMostRecentStatusChangeKey] = -1
    elif pd.isna(mergedNote[c.timestampMillisOfMostRecentStatusChangeKey]):
      mergedNote[c.timestampMillisOfMostRecentStatusChangeKey] = -1

  # Update the current status in accordance with this scoring run.
  assert not pd.isna(mergedNote[c.finalRatingStatusKey])
  mergedNote[c.currentLabelKey] = mergedNote[c.finalRatingStatusKey]
  mergedNote[c.currentCoreStatusKey] = mergedNote[c.coreRatingStatusKey]
  mergedNote[c.currentExpansionStatusKey] = mergedNote[c.expansionRatingStatusKey]
  mergedNote[c.currentGroupStatusKey] = mergedNote[c.groupRatingStatusKey]
  mergedNote[c.currentDecidedByKey] = mergedNote[c.decidedByKey]
  mergedNote[c.currentModelingGroupKey] = mergedNote[c.modelingGroupKey]
  mergedNote[c.timestampMillisOfNoteCurrentLabelKey] = currentTimeMillis
  mergedNote[c.currentMultiGroupStatusKey] = mergedNote[c.multiGroupRatingStatusKey]
  mergedNote[c.currentModelingMultiGroupKey] = mergedNote[c.modelingMultiGroupKey]

  # Lock notes which are (1) not already locked, (2) old enough to lock and (3)
  # were decided by logic which has global display impact.  Criteria (3) guarantees
  # that any CRH note which is shown to all users will have the status locked, but
  # also means that if a note is scored as NMR by all trusted models and CRH by an
  # experimental model we will avoid locking the note to NMR even though the note was
  # in scope for a trusted model and scored as NMR at the time of status locking.
  # The note will continue to be CRH only as long as an experimental model continues
  # scoring the note as CRH (or a core model scores the note as CRH).  If at any point
  # both experimental models and core models score the note as NMR, then the note will
  # lock to NMR.

  # Check whether the note has already been locked.
  notAlreadyLocked = pd.isna(mergedNote[c.lockedStatusKey])
  # Check whether the note is old enough to be eligible for locking.
  lockEligible = c.noteLockMillis < (currentTimeMillis - mergedNote[c.createdAtMillisKey])
  # Check whether the note was decided by a rule which displays globally
  trustedRule = mergedNote[c.decidedByKey] in {
    rule.get_name() for rule in RuleID if rule.value.lockingEnabled
  }

  if notAlreadyLocked and lockEligible and trustedRule:
    mergedNote[c.lockedStatusKey] = mergedNote[c.finalRatingStatusKey]
    mergedNote[c.timestampMillisOfStatusLockKey] = currentTimeMillis

  if pd.notna(mergedNote[c.lockedStatusKey]):
    # Clear timestampMillisOfNmrDueToMinStableCrhTimeKey if the note is locked.
    mergedNote[c.timestampMillisOfNmrDueToMinStableCrhTimeKey] = -1
  else:
    # If note is unlocked, allow updates to the stabilization timestamp and first
    # stabilization timestamp.
    if not pd.isna(mergedNote[c.updatedTimestampMillisOfNmrDueToMinStableCrhTimeKey]):
      mergedNote[c.timestampMillisOfNmrDueToMinStableCrhTimeKey] = mergedNote[
        c.updatedTimestampMillisOfNmrDueToMinStableCrhTimeKey
      ]
      if pd.isna(mergedNote[c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey]):
        mergedNote[c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey] = mergedNote[
          c.timestampMillisOfNmrDueToMinStableCrhTimeKey
        ]

  if pd.isna(mergedNote[c.createdAtMillisKey + newScoredNotesSuffix]):
    # note used to be scored but isn't now; just retain old info
    return mergedNote

  # If the notes created before the deleted note cutfoff. Retain
  # the old data.
  if mergedNote[c.createdAtMillisKey] < c.deletedNoteTombstonesLaunchTime:
    return mergedNote

  if np.invert(pd.isna(mergedNote[c.createdAtMillisKey])) & np.invert(
    pd.isna(mergedNote[c.createdAtMillisKey + newScoredNotesSuffix])
  ):
    assert (
      mergedNote[c.createdAtMillisKey] == mergedNote[c.createdAtMillisKey + newScoredNotesSuffix]
    )

  if pd.isna(mergedNote[c.createdAtMillisKey]):
    raise Exception("This should be impossible, we already called add new notes")

  if mergedNote[c.finalRatingStatusKey] != c.needsMoreRatings:
    if pd.isna(mergedNote[c.firstNonNMRLabelKey]):
      # first time note has a status
      mergedNote[c.firstNonNMRLabelKey] = mergedNote[c.finalRatingStatusKey]
      mergedNote[c.timestampMillisOfNoteFirstNonNMRLabelKey] = currentTimeMillis
      mergedNote[c.mostRecentNonNMRLabelKey] = mergedNote[c.finalRatingStatusKey]
      mergedNote[c.timestampMillisOfNoteMostRecentNonNMRLabelKey] = currentTimeMillis

    if mergedNote[c.finalRatingStatusKey] != mergedNote[c.mostRecentNonNMRLabelKey]:
      # NOTE: By design, this branch captures label flips between CRH and CRNH but not NMR.
      mergedNote[c.mostRecentNonNMRLabelKey] = mergedNote[c.finalRatingStatusKey]
      mergedNote[c.timestampMillisOfNoteMostRecentNonNMRLabelKey] = currentTimeMillis

  return mergedNote


def check_flips(mergedStatuses: pd.DataFrame, noteSubset: c.NoteSubset) -> Tuple[bool, str]:
  """Validate that number of CRH notes remains within an accepted bound.

  Assert fails and scoring exits with error if maximum allowable churn is exceeded.

  Args:
    mergedStatuses: NSH DF with new and old data combined.
    maxCrhChurn: maximum fraction of unlocked notes to gain or lose CRH status.

  Returns:
    None
  """
  if len(mergedStatuses) > c.minNumNotesForProdData:
    # Prune notes to unlocked notes.
    mergedStatuses = mergedStatuses[mergedStatuses[c.timestampMillisOfStatusLockKey].isna()]
    # Prune to note subset
    logger.info(
      f"Checking Flip Rate for note subset: {noteSubset.description} (unlocked only), with max new CRH churn: {noteSubset.maxNewCrhChurnRate}, and max old CRH churn: {noteSubset.maxOldCrhChurnRate}"
    )
    if noteSubset.noteSet is not None:
      mergedStatuses = mergedStatuses[mergedStatuses[c.noteIdKey].isin(noteSubset.noteSet)]

    return _check_flips(
      mergedStatuses,
      noteSubset.maxNewCrhChurnRate,
      noteSubset.maxOldCrhChurnRate,
      description=noteSubset.description,
    )
  return False, ""


def _check_flips(
  mergedStatuses: pd.DataFrame,
  maxNewCrhChurn: float,
  maxOldCrhChurn: Optional[float] = None,
  smoothingCount: int = 100,
  description: Optional[c.RescoringRuleID] = None,
  sampleSizeToPrintInFailedAssert: int = 30,
) -> Tuple[bool, str]:
  desc = ""
  failedCheckFlips = False

  if maxOldCrhChurn is None:
    maxOldCrhChurn = maxNewCrhChurn

  # Identify new and old CRH notes.
  oldCrhNotes = frozenset(
    mergedStatuses[mergedStatuses[c.currentLabelKey] == c.currentlyRatedHelpful][c.noteIdKey]
  )
  newCrhNotes = frozenset(
    mergedStatuses[mergedStatuses[c.finalRatingStatusKey] == c.currentlyRatedHelpful][c.noteIdKey]
  )
  if len(oldCrhNotes) > 0 and len(newCrhNotes) > 0:
    # Validate that changes are within allowable bounds.
    smoothedNewNoteRatio = (len(newCrhNotes - oldCrhNotes)) / (len(oldCrhNotes) + smoothingCount)
    rawNewNoteRatio = (len(newCrhNotes - oldCrhNotes)) / len(oldCrhNotes)
    logger.info(
      f"Raw new note ratio: {rawNewNoteRatio}, smoothed new note ratio: {smoothedNewNoteRatio}. (newCrhNotes={len(newCrhNotes)}, oldCrhNotes={len(oldCrhNotes)}, delta={len(newCrhNotes - oldCrhNotes)}"
    )
    smoothedOldNoteRatio = (len(oldCrhNotes - newCrhNotes)) / (len(oldCrhNotes) + smoothingCount)
    rawOldNoteRatio = (len(oldCrhNotes - newCrhNotes)) / len(oldCrhNotes)
    logger.info(
      f"Raw old note ratio: {rawOldNoteRatio}, smoothed old note ratio: {smoothedOldNoteRatio}. (newCrhNotes={len(newCrhNotes)}, oldCrhNotes={len(oldCrhNotes)}, delta={len(oldCrhNotes - newCrhNotes)}"
    )

    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", max(20, sampleSizeToPrintInFailedAssert))

    if smoothedNewNoteRatio > maxNewCrhChurn:
      failedCheckFlips = True
      desc += f"""Too many new CRH notes (rescoringRule: {description}): 
      smoothedNewNoteRatio={smoothedNewNoteRatio}
      maxNewCrhChurn={maxNewCrhChurn}
      newCrhNotes={len(newCrhNotes)}
      oldCrhNotes={len(oldCrhNotes)}
      delta={len(newCrhNotes - oldCrhNotes)}
      Sample Notes: 
      {mergedStatuses[(mergedStatuses[c.noteIdKey].isin(newCrhNotes - oldCrhNotes))].sample(min(len(newCrhNotes - oldCrhNotes), sampleSizeToPrintInFailedAssert))}"""

    if smoothedOldNoteRatio > maxOldCrhChurn:
      failedCheckFlips = True
      desc += f"""Too many notes lost CRH status (rescoringRule: {description}): 
      smoothedOldNoteRatio={smoothedOldNoteRatio}
      maxOldCrhChurn={maxOldCrhChurn}
      oldCrhNotes={len(oldCrhNotes)}
      newCrhNotes={len(newCrhNotes)}
      delta={len(oldCrhNotes - newCrhNotes)}
      Sample Notes: 
      {mergedStatuses[(mergedStatuses[c.noteIdKey].isin(oldCrhNotes - newCrhNotes))].sample(min(len(oldCrhNotes - newCrhNotes), sampleSizeToPrintInFailedAssert))}"""

  return failedCheckFlips, desc


def merge_old_and_new_note_statuses(
  oldNoteStatusHistory: pd.DataFrame,
  scoredNotes: pd.DataFrame,
):
  newScoredNotesSuffix = "_sn"
  mergedStatuses = oldNoteStatusHistory.merge(
    scoredNotes[
      [
        c.noteIdKey,
        c.createdAtMillisKey,
        c.finalRatingStatusKey,
        c.decidedByKey,
        c.coreRatingStatusKey,
        c.expansionRatingStatusKey,
        c.groupRatingStatusKey,
        c.modelingGroupKey,
        c.updatedTimestampMillisOfNmrDueToMinStableCrhTimeKey,
        c.multiGroupRatingStatusKey,
        c.modelingMultiGroupKey,
      ]
    ].rename(
      {
        c.createdAtMillisKey: c.createdAtMillisKey + newScoredNotesSuffix,
      },
      axis=1,
    ),
    how="inner",
    suffixes=("", newScoredNotesSuffix),
  )
  assert len(mergedStatuses) == len(
    oldNoteStatusHistory
  ), "scoredNotes and oldNoteStatusHistory should both contain all notes"
  return mergedStatuses


def update_note_status_history(
  mergedStatuses: pd.DataFrame,
  newScoredNotesSuffix: str = "_sn",
) -> pd.DataFrame:
  """Generate new noteStatusHistory by merging in new note labels."""
  if c.useCurrentTimeInsteadOfEpochMillisForNoteStatusHistory:
    # When running in prod, we use the latest time possible, so as to include as many valid ratings
    # as possible, and be closest to the time the new note statuses are user-visible.
    currentTimeMillis = 1000 * time.time()
  else:
    # When running in test, we use the overridable epochMillis constant.
    currentTimeMillis = c.epochMillis

  def apply_update(mergedNote):
    return _update_single_note_status_history(
      mergedNote, currentTimeMillis=currentTimeMillis, newScoredNotesSuffix=newScoredNotesSuffix
    )

  newNoteStatusHistory = mergedStatuses.apply(apply_update, axis=1)

  assert pd.isna(newNoteStatusHistory[c.noteAuthorParticipantIdKey]).sum() == 0
  assert pd.isna(newNoteStatusHistory[c.createdAtMillisKey]).sum() == 0
  return newNoteStatusHistory[c.noteStatusHistoryTSVColumns]
