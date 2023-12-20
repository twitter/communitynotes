import time

from . import constants as c
from .scoring_rules import RuleID

import numpy as np
import pandas as pd


# Delay specifying when to lock note status, currently set to two weeks.
_noteLockMillis = 14 * 24 * 60 * 60 * 1000


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
  )
  newNotes = pd.isna(newNoteStatusHistory[c.createdAtMillisKey])
  print(f"total notes added to noteStatusHistory: {sum(newNotes)}")
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
  # Update the current status in accordance with this scoring run.
  assert not pd.isna(mergedNote[c.finalRatingStatusKey])
  mergedNote[c.currentLabelKey] = mergedNote[c.finalRatingStatusKey]
  mergedNote[c.currentCoreStatusKey] = mergedNote[c.coreRatingStatusKey]
  mergedNote[c.currentExpansionStatusKey] = mergedNote[c.expansionRatingStatusKey]
  mergedNote[c.currentGroupStatusKey] = mergedNote[c.groupRatingStatusKey]
  mergedNote[c.currentDecidedByKey] = mergedNote[c.decidedByKey]
  mergedNote[c.currentModelingGroupKey] = mergedNote[c.modelingGroupKey]
  mergedNote[c.timestampMillisOfNoteCurrentLabelKey] = currentTimeMillis

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
  lockEligible = _noteLockMillis < (currentTimeMillis - mergedNote[c.createdAtMillisKey])
  # Check whether the note was decided by a rule which displays globally
  trustedRule = mergedNote[c.decidedByKey] in {
    rule.get_name() for rule in RuleID if rule.value.lockingEnabled
  }

  if notAlreadyLocked and lockEligible and trustedRule:
    mergedNote[c.lockedStatusKey] = mergedNote[c.finalRatingStatusKey]
    mergedNote[c.timestampMillisOfStatusLockKey] = currentTimeMillis

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


def _check_flips(mergedStatuses: pd.DataFrame, maxCrhChurn=0.25) -> None:
  """Validate that number of CRH notes remains within an accepted bound.

  Assert fails and scoring exits with error if maximum allowable churn is exceeded.

  Args:
    mergedStatuses: NSH DF with new and old data combined.
    maxCrhChurn: maximum fraction of unlocked notes to gain or lose CRH status.

  Returns:
    None
  """
  # Prune to unlocked notes.
  mergedStatuses = mergedStatuses[mergedStatuses[c.timestampMillisOfStatusLockKey].isna()]
  # Identify new and old CRH notes.
  oldCrhNotes = frozenset(
    mergedStatuses[mergedStatuses[c.currentLabelKey] == c.currentlyRatedHelpful][c.noteIdKey]
  )
  newCrhNotes = frozenset(
    mergedStatuses[mergedStatuses[c.finalRatingStatusKey] == c.currentlyRatedHelpful][c.noteIdKey]
  )
  # Validate that changes are within allowable bounds.
  assert (
    (len(newCrhNotes - oldCrhNotes) / len(oldCrhNotes)) < maxCrhChurn
  ), f"Too many new CRH notes: newCrhNotes={len(newCrhNotes)}, oldCrhNotes={len(oldCrhNotes)}, delta={len(newCrhNotes - oldCrhNotes)}"
  assert (
    (len(oldCrhNotes - newCrhNotes) / len(oldCrhNotes)) < maxCrhChurn
  ), f"Too few new CRH notes: newCrhNotes={len(newCrhNotes)}, oldCrhNotes={len(oldCrhNotes)}, delta={len(oldCrhNotes - newCrhNotes)}"


def update_note_status_history(
  oldNoteStatusHistory: pd.DataFrame,
  scoredNotes: pd.DataFrame,
) -> pd.DataFrame:
  """Generate new noteStatusHistory by merging in new note labels.

  Args:
      oldNoteStatusHistory (pd.DataFrame)
      scoredNotes (pd.DataFrame)

  Returns:
      pd.DataFrame: noteStatusHistory
  """
  if c.useCurrentTimeInsteadOfEpochMillisForNoteStatusHistory:
    # When running in prod, we use the latest time possible, so as to include as many valid ratings
    # as possible, and be closest to the time the new note statuses are user-visible.
    currentTimeMillis = 1000 * time.time()
  else:
    # When running in test, we use the overridable epochMillis constant.
    currentTimeMillis = c.epochMillis
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
  if len(mergedStatuses) > c.minNumNotesForProdData:
    _check_flips(mergedStatuses)

  def apply_update(mergedNote):
    return _update_single_note_status_history(
      mergedNote, currentTimeMillis=currentTimeMillis, newScoredNotesSuffix=newScoredNotesSuffix
    )

  newNoteStatusHistory = mergedStatuses.apply(apply_update, axis=1)

  assert pd.isna(newNoteStatusHistory[c.noteAuthorParticipantIdKey]).sum() == 0
  assert pd.isna(newNoteStatusHistory[c.createdAtMillisKey]).sum() == 0
  return newNoteStatusHistory[c.noteStatusHistoryTSVColumns]
