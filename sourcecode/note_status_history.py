from time import time

import constants as c

import numpy as np
import pandas as pd


def add_new_notes(oldNoteStatusHistory: pd.DataFrame, notes: pd.DataFrame) -> pd.DataFrame:
  """Add the creation time and authorId of notes to noteStatusHistory.
  Useful when you have some new notes; called as a pre-processing step.

  Args:
      oldNoteStatusHistory (pd.DataFrame)
      notes (pd.DataFrame)

  Returns:
      pd.DataFrame: noteStatusHistory
  """
  newNoteStatusHistory = oldNoteStatusHistory.merge(
    notes[[c.noteIdKey, c.createdAtMillisKey, c.noteAuthorParticipantIdKey]],
    on=c.noteIdKey,
    how="outer",
    suffixes=("", "_notes"),
  )
  newNotes = pd.isna(newNoteStatusHistory[c.createdAtMillisKey])
  newNoteStatusHistory.loc[newNotes, c.createdAtMillisKey] = newNoteStatusHistory.loc[
    newNotes, c.createdAtMillisKey + "_notes"
  ]
  newNoteStatusHistory.loc[newNotes, c.participantIdKey] = newNoteStatusHistory.loc[
    newNotes, c.noteAuthorParticipantIdKey
  ]

  return newNoteStatusHistory


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
  mergedNote[c.currentLabelKey] = mergedNote[c.ratingStatusKey]
  mergedNote[c.timestampMillisOfNoteCurrentLabelKey] = currentTimeMillis

  if pd.isna(mergedNote[c.createdAtMillisKey + newScoredNotesSuffix]):
    # note used to be scored but isn't now; just retain old info
    return mergedNote

  # If the note's created before the deleted note cutfoff, retain
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
      mergedNote[c.timestampMillisOfNoteFirstNonNMRLabelKey] = currentTimeMillis

    if mergedNote[c.ratingStatusKey] != mergedNote[c.mostRecentNonNMRLabelKey]:
      # label flip
      mergedNote[c.mostRecentNonNMRLabelKey] = mergedNote[c.ratingStatusKey]
      mergedNote[c.timestampMillisOfNoteMostRecentNonNMRLabelKey] = currentTimeMillis

  return mergedNote


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

  assert pd.isna(newNoteStatusHistory[c.participantIdKey]).sum() == 0
  assert pd.isna(newNoteStatusHistory[c.createdAtMillisKey]).sum() == 0

  return newNoteStatusHistory[c.noteStatusHistoryTSVColumns]
