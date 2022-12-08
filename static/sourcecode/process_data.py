from io import StringIO
from typing import Tuple

import constants as c, note_status_history

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def get_data(
  notesPath: str,
  ratingsPath: str,
  noteStatusHistoryPath: str,
  userEnrollmentPath: str,
  shouldFilterNotMisleadingNotes: bool = True,
  logging: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """All-in-one function for reading Birdwatch notes and ratings from TSV files.
  It does both reading and pre-processing.

  Args:
      notesPath (str): file path
      ratingsPath (str): file path
      noteStatusHistoryPath (str): file path
      shouldFilterNotMisleadingNotes (bool, optional): Throw out not-misleading notes if True. Defaults to True.
      logging (bool, optional): Print out debug output. Defaults to True.

  Returns:
      Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: notes, ratings, noteStatusHistory, userEnrollment
  """
  notes, ratings, noteStatusHistory, userEnrollment = read_from_tsv(
    notesPath, ratingsPath, noteStatusHistoryPath, userEnrollmentPath
  )
  notes, ratings, noteStatusHistory = preprocess_data(
    notes, ratings, noteStatusHistory, shouldFilterNotMisleadingNotes, logging
  )
  return notes, ratings, noteStatusHistory, userEnrollment


def read_from_strings(
  notesStr: str, ratingsStr: str, noteStatusHistoryStr: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Read from TSV formatted String.

  Args:
      notesStr (str): tsv-formatted notes dataset
      ratingsStr (str): tsv-formatted ratings dataset
      noteStatusHistoryStr (str): tsv-formatted note status history dataset

  Returns:
     Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: notes, ratings, noteStatusHistory
  """
  notes = pd.read_csv(
    StringIO(notesStr), sep="\t", names=c.noteTSVColumns, dtype=c.noteTSVTypeMapping
  )
  ratings = pd.read_csv(
    StringIO(ratingsStr), sep="\t", names=c.ratingTSVColumns, dtype=c.ratingTSVTypeMapping
  )
  noteStatusHistory = pd.read_csv(
    StringIO(noteStatusHistoryStr),
    sep="\t",
    names=c.noteStatusHistoryTSVColumns,
    dtype=c.noteStatusHistoryTSVTypeMapping,
  )

  return notes, ratings, noteStatusHistory


def tsv_reader(path: str, mapping, columns):
  try:
    return pd.read_csv(path, sep="\t", dtype=mapping, names=columns)
  except ValueError as e:
    print(e)
    return pd.read_csv(path, sep="\t", dtype=mapping)


def read_from_tsv(
  notesPath: str, ratingsPath: str, noteStatusHistoryPath: str, userEnrollmentPath: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Mini function to read notes, ratings, and noteStatusHistory from TSVs.

  Args:
      notesPath (str): path
      ratingsPath (str): path
      noteStatusHistoryPath (str): path
      userEnrollmentPath (str): path
  Returns:
      Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: notes, ratings, noteStatusHistory, userEnrollment
  """
  notes = tsv_reader(notesPath, c.noteTSVTypeMapping, c.noteTSVColumns)
  ratings = tsv_reader(ratingsPath, c.ratingTSVTypeMapping, c.ratingTSVColumns)
  noteStatusHistory = tsv_reader(
    noteStatusHistoryPath, c.noteStatusHistoryTSVTypeMapping, c.noteStatusHistoryTSVColumns
  )
  userEnrollment = tsv_reader(
    userEnrollmentPath, c.userEnrollmentTSVTypeMapping, c.userEnrollmentTSVColumns
  )

  assert len(notes.columns) == len(c.noteTSVColumns) and all(notes.columns == c.noteTSVColumns), (
    f"note columns don't match: \n{[col for col in notes.columns if not col in c.noteTSVColumns]} are extra columns, "
    + f"\n{[col for col in c.noteTSVColumns if not col in notes.columns]} are missing."
  )  # ensure constants file is up to date.
  notes = notes.rename(columns={c.participantIdKey: c.noteAuthorParticipantIdKey})

  assert len(ratings.columns.values) == len(c.ratingTSVColumns) and all(
    ratings.columns == c.ratingTSVColumns
  ), (
    f"ratings columns don't match: \n{[col for col in ratings.columns if not col in c.ratingTSVColumns]} are extra columns, "
    + f"\n{[col for col in c.ratingTSVColumns if not col in ratings.columns]} are missing."
  )  # ensure constants file is up to date.
  ratings = ratings.rename(columns={c.participantIdKey: c.raterParticipantIdKey})

  assert len(noteStatusHistory.columns.values) == len(c.noteStatusHistoryTSVColumns) and all(
    noteStatusHistory.columns == c.noteStatusHistoryTSVColumns
  ), (
    f"noteStatusHistory columns don't match: \n{[col for col in noteStatusHistory.columns if not col in c.noteStatusHistoryTSVColumns]} are extra columns, "
    + f"\n{[col for col in c.noteStatusHistoryTSVColumns if not col in noteStatusHistory.columns]} are missing."
  )
  noteStatusHistory = noteStatusHistory.rename(columns={c.participantIdKey: c.noteAuthorParticipantIdKey})

  assert len(userEnrollment.columns.values) == len(c.userEnrollmentTSVColumns) and all(
    userEnrollment.columns == c.userEnrollmentTSVColumns
  ), (
    f"userEnrollment columns don't match: \n{[col for col in userEnrollment.columns if not col in c.userEnrollmentTSVColumns]} are extra columns, "
    + f"\n{[col for col in c.userEnrollmentTSVColumns if not col in userEnrollment.columns]} are missing."
  )

  return notes, ratings, noteStatusHistory, userEnrollment


def _filter_misleading_notes(
  notes: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  logging: bool = True,
) -> pd.DataFrame:
  """
  This function actually filters ratings (not notes), based on which notes they rate.

  Filter out ratings of notes that say the Tweet isn't misleading.
  Also filter out ratings of deleted notes, unless they were deleted after
    c.deletedNotesTombstoneLaunchTime, and appear in noteStatusHistory.

  Args:
      notes (pd.DataFrame): _description_
      ratings (pd.DataFrame): _description_
      noteStatusHistory (pd.DataFrame): _description_
      logging (bool, optional): _description_. Defaults to True.

  Returns:
      pd.DataFrame: filtered ratings
  """
  ratings = ratings.merge(
    noteStatusHistory[[c.noteIdKey, c.createdAtMillisKey, c.classificationKey]],
    on=c.noteIdKey,
    how="left",
    suffixes=("", "_nsh"),
  )

  deletedNoteKey = "deletedNote"
  notDeletedMisleadingKey = "notDeletedMisleading"
  deletedButInNSHKey = "deletedButInNSH"
  createdAtMillisNSHKey = c.createdAtMillisKey + "_nsh"

  ratings[deletedNoteKey] = pd.isna(ratings[c.classificationKey])
  ratings[notDeletedMisleadingKey] = np.invert(ratings[deletedNoteKey]) & (
    ratings[c.classificationKey] == c.notesSaysTweetIsMisleadingKey
  )
  ratings[deletedButInNSHKey] = ratings[deletedNoteKey] & np.invert(
    pd.isna(ratings[createdAtMillisNSHKey])
  )

  deletedNotInNSH = (ratings[deletedNoteKey]) & pd.isna(ratings[createdAtMillisNSHKey])
  notDeletedNotMisleadingOldUI = (
    ratings[c.classificationKey] == c.noteSaysTweetIsNotMisleadingKey
  ) & (ratings[createdAtMillisNSHKey] <= c.notMisleadingUILaunchTime)
  notDeletedNotMisleadingNewUI = (
    ratings[c.classificationKey] == c.noteSaysTweetIsNotMisleadingKey
  ) & (ratings[createdAtMillisNSHKey] > c.notMisleadingUILaunchTime)

  if logging:
    print(
      f"Preprocess Data: Filter misleading notes, starting with {len(ratings)} ratings on {len(np.unique(ratings[c.noteIdKey]))} notes"
    )
    print(
      f"  Keeping {ratings[notDeletedMisleadingKey].sum()} ratings on {len(np.unique(ratings.loc[ratings[notDeletedMisleadingKey],c.noteIdKey]))} misleading notes"
    )
    print(
      f"  Keeping {ratings[deletedButInNSHKey].sum()} ratings on {len(np.unique(ratings.loc[ratings[deletedButInNSHKey],c.noteIdKey]))} deleted notes that were previously scored (in note status history)"
    )
    print(
      f"  Removing {notDeletedNotMisleadingOldUI.sum()} ratings on {len(np.unique(ratings.loc[notDeletedNotMisleadingOldUI, c.noteIdKey]))} older notes that aren't deleted, but are not-misleading."
    )
    print(
      f"  Removing {deletedNotInNSH.sum()} ratings on {len(np.unique(ratings.loc[deletedNotInNSH, c.noteIdKey]))} notes that were deleted and not in note status history (e.g. old)."
    )

  ratings = ratings[
    ratings[notDeletedMisleadingKey] | ratings[deletedButInNSHKey] | notDeletedNotMisleadingNewUI
  ]
  ratings = ratings.drop(
    columns=[
      createdAtMillisNSHKey,
      c.classificationKey,
      deletedNoteKey,
      notDeletedMisleadingKey,
      deletedButInNSHKey,
    ]
  )
  return ratings


def remove_duplicate_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
  """Drop duplicate ratings, then assert that there is exactly one rating per noteId per raterId.

  Args:
      ratings (pd.DataFrame) with possible duplicated ratings

  Returns:
      pd.DataFrame: ratings, with one record per userId, noteId.
  """
  # Construct a new DataFrame to avoid SettingWithCopyWarning
  ratings = pd.DataFrame(ratings.drop_duplicates())

  numRatings = len(ratings)
  numUniqueRaterIdNoteIdPairs = len(ratings.groupby([c.raterParticipantIdKey, c.noteIdKey]).head(1))
  assert (
    numRatings == numUniqueRaterIdNoteIdPairs
  ), f"Only {numUniqueRaterIdNoteIdPairs} unique raterId,noteId pairs but {numRatings} ratings"
  return ratings


def remove_duplicate_notes(notes: pd.DataFrame) -> pd.DataFrame:
  """Remove duplicate notes, then assert that there is only one copy of each noteId.

  Args:
      notes (pd.DataFrame): with possible duplicate notes

  Returns:
      notes (pd.DataFrame) with one record per noteId
  """
  # Construct a new DataFrame to avoid SettingWithCopyWarning
  notes = pd.DataFrame(notes.drop_duplicates())

  numNotes = len(notes)
  numUniqueNotes = len(np.unique(notes[c.noteIdKey]))
  assert (
    numNotes == numUniqueNotes
  ), f"Found only {numUniqueNotes} unique noteIds out of {numNotes} notes"

  return notes


def preprocess_data(
  notes: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  shouldFilterNotMisleadingNotes: bool = True,
  logging: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Populate helpfulNumKey, a unified column that merges the helpfulness answers from
  the V1 and V2 rating forms together, as described in
  https://twitter.github.io/birdwatch/ranking-notes/#helpful-rating-mapping.

  Also, filter notes that indicate the Tweet is misleading, if the flag is True.

  Args:
      notes (pd.DataFrame)
      ratings (pd.DataFrame)
      noteStatusHistory (pd.DataFrame)
      shouldFilterNotMisleadingNotes (bool, optional): Defaults to True.
      logging (bool, optional): Defaults to True.

  Returns:
      notes (pd.DataFrame)
      ratings (pd.DataFrame)
      noteStatusHistory (pd.DataFrame)
  """
  if logging:
    print(
      "Timestamp of latest rating in data: ",
      pd.to_datetime(ratings[c.createdAtMillisKey], unit="ms").max(),
    )
    print(
      "Timestamp of latest note in data: ",
      pd.to_datetime(notes[c.createdAtMillisKey], unit="ms").max(),
    )
  ratings = remove_duplicate_ratings(ratings)
  notes = remove_duplicate_notes(notes)

  ratings.loc[:, c.helpfulNumKey] = np.nan
  ratings.loc[ratings[c.helpfulKey] == 1, c.helpfulNumKey] = 1
  ratings.loc[ratings[c.notHelpfulKey] == 1, c.helpfulNumKey] = 0
  ratings.loc[ratings[c.helpfulnessLevelKey] == c.notHelpfulValueTsv, c.helpfulNumKey] = 0
  ratings.loc[ratings[c.helpfulnessLevelKey] == c.somewhatHelpfulValueTsv, c.helpfulNumKey] = 0.5
  ratings.loc[ratings[c.helpfulnessLevelKey] == c.helpfulValueTsv, c.helpfulNumKey] = 1
  ratings = ratings.loc[~pd.isna(ratings[c.helpfulNumKey])]

  notes[c.tweetIdKey] = notes[c.tweetIdKey].astype(np.str)

  noteStatusHistory = note_status_history.merge_note_info(noteStatusHistory, notes)

  if shouldFilterNotMisleadingNotes:
    ratings = _filter_misleading_notes(notes, ratings, noteStatusHistory, logging)

  if logging:
    print(
      "Num Ratings: %d, Num Unique Notes Rated: %d, Num Unique Raters: %d"
      % (
        len(ratings),
        len(np.unique(ratings[c.noteIdKey])),
        len(np.unique(ratings[c.raterParticipantIdKey])),
      )
    )
  return notes, ratings, noteStatusHistory


def filter_ratings(ratings: pd.DataFrame, logging: bool = True) -> pd.DataFrame:
  """Apply min number of ratings for raters & notes. Instead of iterating these filters
  until convergence, simply stop after going back and force once.

  Args:
      ratings (pd.DataFrame): unfiltered ratings
      logging (bool, optional): debug output. Defaults to True.

  Returns:
      pd.DataFrame: filtered ratings
  """

  n = ratings.groupby(c.noteIdKey).size().reset_index()
  notesWithMinNumRatings = n[n[0] >= c.minNumRatersPerNote]

  ratingsNoteFiltered = ratings.merge(notesWithMinNumRatings[[c.noteIdKey]], on=c.noteIdKey)

  if logging:
    print("Filter notes and ratings with too few ratings")
    print(
      "  After Filtering Notes w/less than %d Ratings, Num Ratings: %d, Num Unique Notes Rated: %d, Num Unique Raters: %d"
      % (
        c.minNumRatersPerNote,
        len(ratingsNoteFiltered),
        len(np.unique(ratingsNoteFiltered[c.noteIdKey])),
        len(np.unique(ratingsNoteFiltered[c.raterParticipantIdKey])),
      )
    )
  r = ratingsNoteFiltered.groupby(c.raterParticipantIdKey).size().reset_index()
  ratersWithMinNumRatings = r[r[0] >= c.minNumRatingsPerRater]

  ratingsDoubleFiltered = ratingsNoteFiltered.merge(
    ratersWithMinNumRatings[[c.raterParticipantIdKey]], on=c.raterParticipantIdKey
  )
  if logging:
    print(
      "  After Filtering Raters w/less than %s Notes, Num Ratings: %d, Num Unique Notes Rated: %d, Num Unique Raters: %d"
      % (
        c.minNumRatingsPerRater,
        len(ratingsDoubleFiltered),
        len(np.unique(ratingsDoubleFiltered[c.noteIdKey])),
        len(np.unique(ratingsDoubleFiltered[c.raterParticipantIdKey])),
      )
    )
  n = ratingsDoubleFiltered.groupby(c.noteIdKey).size().reset_index()
  notesWithMinNumRatings = n[n[0] >= c.minNumRatersPerNote]
  ratingsForTraining = ratingsDoubleFiltered.merge(
    notesWithMinNumRatings[[c.noteIdKey]], on=c.noteIdKey
  )
  if logging:
    print(
      "  After Final Filtering of Notes w/less than %d Ratings, Num Ratings: %d, Num Unique Notes Rated: %d, Num Unique Raters: %d"
      % (
        c.minNumRatersPerNote,
        len(ratingsForTraining),
        len(np.unique(ratingsForTraining[c.noteIdKey])),
        len(np.unique(ratingsForTraining[c.raterParticipantIdKey])),
      )
    )
  return ratingsForTraining


def visualize_parameters(noteParams: pd.DataFrame, raterParams: pd.DataFrame) -> None:
  """Plot/describe note and rater params.

  Args:
      noteParams (pd.DataFrame)
      raterParams (pd.DataFrame)
  """
  print(noteParams.describe())

  plt.figure()
  noteParams[c.noteInterceptKey].plot(kind="hist", bins=20)

  plt.figure()
  noteParams[c.noteFactor1Key].plot(kind="hist", bins=20)

  plt.figure()
  noteParams.plot(kind="scatter", x=c.noteFactor1Key, y=c.noteInterceptKey, alpha=0.05)

  print(raterParams.describe())

  plt.figure()
  raterParams[c.raterInterceptKey].plot(kind="hist", bins=20)

  plt.figure()
  raterParams[c.raterFactor1Key].plot(kind="hist", bins=20)


def visualize_helpfulness(helpfulness_scores: pd.DataFrame) -> None:
  """Plot user helpfulness scores.

  Args:
      helpfulness_scores (pd.DataFrame)
  """
  plt.figure()
  helpfulness_scores[c.crhCrnhRatioDifferenceKey].plot(kind="hist", bins=20)

  plt.figure()
  helpfulness_scores[c.meanNoteScoreKey].plot(kind="hist", bins=20)

  plt.figure()
  helpfulness_scores[c.raterAgreeRatioKey].plot(kind="hist", bins=20)


def write_scored_notes_with_summary(scoredNotes: pd.DataFrame, notes: pd.DataFrame) -> None:
  """Write scoredNotes as a TSV, with summary joined in.

  Args:
      scoredNotes (pd.DataFrame)
      notes (pd.DataFrame)
  """
  columnsToWrite = (
    [
      c.noteIdKey,
      c.tweetIdKey,
      c.numRatingsKey,
      c.noteInterceptKey,
      c.noteFactor1Key,
      c.ratingStatusKey,
      c.firstTagKey,
      c.secondTagKey,
      c.summaryKey,
    ]
    + c.helpfulTagsTSVOrder
    + c.notHelpfulTagsTSVOrder
  )
  scoredNotes = scoredNotes.merge(
    notes[[c.noteIdKey, c.summaryKey, c.tweetIdKey]], on=c.noteIdKey, how="inner"
  )

  scoredNotes[columnsToWrite].sort_values(by=c.noteInterceptKey, ascending=False).to_csv(
    c.scoredNotesOutputPath, sep="\t", index=False
  )


def write_scored_notes(scoredNotes: pd.DataFrame) -> None:
  """Write scoredNotes as a TSV.

  Args:
      scoredNotes (pd.DataFrame)
  """
  columnsToWrite = (
    [c.noteIdKey, c.helpfulNumKey]
    + c.helpfulTagsTSVOrder
    + c.notHelpfulTagsTSVOrder
    + [
      c.numRatingsKey,
      c.noteInterceptKey,
      c.noteFactor1Key,
      c.ratingStatusKey,
      c.firstTagKey,
      c.secondTagKey,
      c.createdAtMillisKey,
      c.noteAuthorParticipantIdKey,
    ]
  )

  dfToWrite = scoredNotes[columnsToWrite].sort_values(by=c.noteInterceptKey, ascending=False)
  dfToWrite.to_csv(c.scoredNotesOutputPath, sep="\t", index=False)
