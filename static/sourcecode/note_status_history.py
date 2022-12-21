from time import time

import numpy as np
import pandas as pd

import constants as c


def merge_note_info(
  oldNoteStatusHistory: pd.DataFrame,
  notes: pd.DataFrame
) -> pd.DataFrame:
  """
  Add the creation time and authorId of notes to noteStatusHistory.

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
  """
  For a single row (representing one note) in note status history, which contains fields
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
  mergedNote[c.currentLabelKey] = mergedNote[c.ratingStatusKey]
  mergedNote[c.timestampMillisOfNoteCurrentLabelKey] = currentTimeMillis

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

  if mergedNote[c.ratingStatusKey] != c.needsMoreRatings:
    if pd.isna(mergedNote[c.firstNonNMRLabelKey]):
      # first time note has a status
      mergedNote[c.firstNonNMRLabelKey] = mergedNote[c.ratingStatusKey]
      mergedNote[c.timestampMillisOfNoteFirstNonNMRLabelKey] = currentTimeMillis
      mergedNote[c.mostRecentNonNMRLabelKey] = mergedNote[c.ratingStatusKey]
      mergedNote[c.timestampMillisOfNoteMostRecentNonNMRLabelKey] = currentTimeMillis

    if mergedNote[c.ratingStatusKey] != mergedNote[c.mostRecentNonNMRLabelKey]:
      # NOTE: By design, this branch captures label flips between CRH and CRNH but not NMR.
      mergedNote[c.mostRecentNonNMRLabelKey] = mergedNote[c.ratingStatusKey]
      mergedNote[c.timestampMillisOfNoteMostRecentNonNMRLabelKey] = currentTimeMillis

  return mergedNote


def update_note_status_history(
  oldNoteStatusHistory: pd.DataFrame,
  scoredNotes: pd.DataFrame,
) -> pd.DataFrame:
  """
  Generate new noteStatusHistory by merging in new note labels.

  Args:
      oldNoteStatusHistory (pd.DataFrame)
      scoredNotes (pd.DataFrame)

  Returns:
      pd.DataFrame: noteStatusHistory
  """
  currentTimeMillis = 1000 * time()
  newScoredNotesSuffix = "_sn"
  mergedStatuses = oldNoteStatusHistory.merge(
    scoredNotes[
      [c.noteIdKey, c.noteAuthorParticipantIdKey, c.createdAtMillisKey, c.ratingStatusKey]
    ].rename(
      {
        c.createdAtMillisKey: c.createdAtMillisKey + newScoredNotesSuffix,
      },
      axis=1,
    ),
    how="outer",
    suffixes=("", newScoredNotesSuffix),
  )

  def apply_update(mergedNote):
    return _update_single_note_status_history(
      mergedNote, currentTimeMillis=currentTimeMillis, newScoredNotesSuffix=newScoredNotesSuffix
    )

  newNoteStatusHistory = mergedStatuses.apply(apply_update, axis=1)

  assert pd.isna(newNoteStatusHistory[c.noteAuthorParticipantIdKey]).sum() == 0
  assert pd.isna(newNoteStatusHistory[c.createdAtMillisKey]).sum() == 0
  return newNoteStatusHistory[
    [col.replace(c.participantIdKey, c.noteAuthorParticipantIdKey)
     for col in c.noteStatusHistoryTSVColumns]]
