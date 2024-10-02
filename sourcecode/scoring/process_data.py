from abc import ABC, abstractmethod
from io import StringIO
import logging
import os
from typing import Dict, List, Optional, Tuple

from . import constants as c, note_status_history
from .pandas_utils import get_df_info

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


logger = logging.getLogger("birdwatch.process_data")
logger.setLevel(logging.INFO)


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


def tsv_parser(
  rawTSV: str,
  mapping: Dict[str, type],
  columns: List[str],
  header: bool,
  useCols: Optional[List[str]] = None,
  chunkSize: Optional[int] = None,
  convertNAToNone: bool = True,
) -> pd.DataFrame:
  """Parse a TSV input and raise an Exception if the input is not formatted as expected.

  Args:
    rawTSV: str contianing entire TSV input
    mapping: Dict mapping column names to types
    columns: List of column names
    header: bool indicating whether the input will have a header
    useCols: Optional list of columns to return
    chunkSize: Optional number of rows to read at a time when returning a subset of columns

  Returns:
    pd.DataFrame containing parsed data
  """
  try:
    firstLine = rawTSV.split("\n")[0]
    num_fields = len(firstLine.split("\t"))
    if num_fields != len(columns):
      raise ValueError(f"Expected {len(columns)} columns, but got {num_fields}")

    if useCols and chunkSize:
      textParser = pd.read_csv(
        StringIO(rawTSV),
        sep="\t",
        names=columns,
        dtype=mapping,
        header=0 if header else None,
        index_col=[],
        usecols=useCols,
        chunksize=chunkSize,
      )
      data = pd.concat(textParser, ignore_index=True)
    else:
      data = pd.read_csv(
        StringIO(rawTSV),
        sep="\t",
        names=columns,
        dtype=mapping,
        header=0 if header else None,
        index_col=[],
        usecols=useCols,
      )
    if convertNAToNone:
      logger.info("Logging size effect of convertNAToNone")
      logger.info("Before conversion:")
      logger.info(get_df_info(data))
      # float types will be nan if missing; newer nullable types like "StringDtype" or "Int64Dtype" will by default
      # be pandas._libs.missing.NAType if missing. Set those to None and change the dtype back to object.
      for colname, coltype in mapping.items():
        # check if coltype is pd.BooleanDtype
        if coltype in set(
          [pd.StringDtype(), pd.BooleanDtype(), pd.Int64Dtype(), pd.Int32Dtype(), "boolean"]
        ):
          data[colname] = data[colname].astype(object)
          data.loc[pd.isna(data[colname]), colname] = None
      logger.info("After conversion:")
      logger.info(get_df_info(data))
    return data
  except (ValueError, IndexError) as e:
    raise ValueError(f"Invalid input: {e}")


def tsv_reader_single(
  path: str, mapping, columns, header=False, parser=tsv_parser, convertNAToNone=True
):
  """Read a single TSV file."""
  with open(path, "r", encoding="utf-8") as handle:
    return tsv_parser(handle.read(), mapping, columns, header, convertNAToNone=convertNAToNone)


def tsv_reader(
  path: str, mapping, columns, header=False, parser=tsv_parser, convertNAToNone=True
) -> pd.DataFrame:
  """Read a single TSV file or a directory of TSV files."""
  if os.path.isdir(path):
    dfs = [
      tsv_reader_single(
        os.path.join(path, filename),
        mapping,
        columns,
        header,
        parser,
        convertNAToNone=convertNAToNone,
      )
      for filename in os.listdir(path)
      if filename.endswith(".tsv")
    ]
    return pd.concat(dfs, ignore_index=True)
  else:
    return tsv_reader_single(
      path, mapping, columns, header, parser, convertNAToNone=convertNAToNone
    )


def read_from_tsv(
  notesPath: Optional[str],
  ratingsPath: Optional[str],
  noteStatusHistoryPath: Optional[str],
  userEnrollmentPath: Optional[str],
  headers: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Mini function to read notes, ratings, and noteStatusHistory from TSVs.

  Args:
      notesPath (str): path
      ratingsPath (str): path
      noteStatusHistoryPath (str): path
      userEnrollmentPath (str): path
      headers: If true, expect first row of input files to be headers.
  Returns:
      Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: notes, ratings, noteStatusHistory, userEnrollment
  """
  if notesPath is None:
    notes = None
  else:
    notes = tsv_reader(
      notesPath, c.noteTSVTypeMapping, c.noteTSVColumns, header=headers, convertNAToNone=False
    )
    assert len(notes.columns) == len(c.noteTSVColumns) and all(notes.columns == c.noteTSVColumns), (
      f"note columns don't match: \n{[col for col in notes.columns if not col in c.noteTSVColumns]} are extra columns, "
      + f"\n{[col for col in c.noteTSVColumns if not col in notes.columns]} are missing."
    )  # ensure constants file is up to date.

  if ratingsPath is None:
    ratings = None
  else:
    ratings = tsv_reader(
      ratingsPath, c.ratingTSVTypeMapping, c.ratingTSVColumns, header=headers, convertNAToNone=False
    )
    assert len(ratings.columns.values) == len(c.ratingTSVColumns) and all(
      ratings.columns == c.ratingTSVColumns
    ), (
      f"ratings columns don't match: \n{[col for col in ratings.columns if not col in c.ratingTSVColumns]} are extra columns, "
      + f"\n{[col for col in c.ratingTSVColumns if not col in ratings.columns]} are missing."
    )  # ensure constants file is up to date.

  if noteStatusHistoryPath is None:
    noteStatusHistory = None
  else:
    # TODO(jiansongc): clean up after new column is in production.
    try:
      noteStatusHistory = tsv_reader(
        noteStatusHistoryPath,
        c.noteStatusHistoryTSVTypeMapping,
        c.noteStatusHistoryTSVColumns,
        header=headers,
        convertNAToNone=False,
      )
      assert len(noteStatusHistory.columns.values) == len(c.noteStatusHistoryTSVColumns) and all(
        noteStatusHistory.columns == c.noteStatusHistoryTSVColumns
      ), (
        f"noteStatusHistory columns don't match: \n{[col for col in noteStatusHistory.columns if not col in c.noteStatusHistoryTSVColumns]} are extra columns, "
        + f"\n{[col for col in c.noteStatusHistoryTSVColumns if not col in noteStatusHistory.columns]} are missing."
      )
    except ValueError:
      noteStatusHistory = tsv_reader(
        noteStatusHistoryPath,
        c.noteStatusHistoryTSVTypeMappingOld,
        c.noteStatusHistoryTSVColumnsOld,
        header=headers,
        convertNAToNone=False,
      )
      noteStatusHistory[c.timestampMillisOfFirstNmrDueToMinStableCrhTimeKey] = np.nan
      assert len(noteStatusHistory.columns.values) == len(c.noteStatusHistoryTSVColumns) and all(
        noteStatusHistory.columns == c.noteStatusHistoryTSVColumns
      ), (
        f"noteStatusHistory columns don't match: \n{[col for col in noteStatusHistory.columns if not col in c.noteStatusHistoryTSVColumns]} are extra columns, "
        + f"\n{[col for col in c.noteStatusHistoryTSVColumns if not col in noteStatusHistory.columns]} are missing."
      )

  if userEnrollmentPath is None:
    userEnrollment = None
  else:
    userEnrollment = tsv_reader(
      userEnrollmentPath,
      c.userEnrollmentTSVTypeMapping,
      c.userEnrollmentTSVColumns,
      header=headers,
      convertNAToNone=False,
    )
    assert len(userEnrollment.columns.values) == len(c.userEnrollmentTSVColumns) and all(
      userEnrollment.columns == c.userEnrollmentTSVColumns
    ), (
      f"userEnrollment columns don't match: \n{[col for col in userEnrollment.columns if not col in c.userEnrollmentTSVColumns]} are extra columns, "
      + f"\n{[col for col in c.userEnrollmentTSVColumns if not col in userEnrollment.columns]} are missing."
    )

  return notes, ratings, noteStatusHistory, userEnrollment


def get_unique_size(df: pd.DataFrame, k: str, rows: pd.Index = None) -> int:
  """Return the number of unique values in a column `k` of a DataFrame `df` at `rows`.

  Args:
      df (pd.DataFrame): DataFrame
      k (str): column name
      rows (pd.Index, optional): rows to consider. Defaults to None.
      
  Returns:
      int: number of unique values in the column
  """
  if rows is not None:
    df = df.loc[rows]
  return len(np.unique(df[k]))


def _filter_misleading_notes(
  notes: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  log: bool = True,
) -> pd.DataFrame:
  """
  This function actually filters ratings (not notes), based on which notes they rate.

  For deleted notes (c.classificationKey is NaN):
    - Keep ratings of notes that appear in noteStatusHistory (previously scored)
    - Remove ratings of notes that do not appear in noteStatusHistory
  For still available notes (c.classificationKey is either MISINFORMED_OR_POTENTIALLY_MISLEADING or NOT_MISLEADING):
    - Keep ratings of notes saying the associated tweet is misleading
    - For those saying the associated tweet is not misleading:
      - Keep ratings after the new UI launch time, c.notMisleadingUILaunchTime
      - Remove ratings before the new UI launch time, c.notMisleadingUILaunchTime

  Args:
      notes (pd.DataFrame): _description_
      ratings (pd.DataFrame): _description_
      noteStatusHistory (pd.DataFrame): _description_
      log (bool, optional): _description_. Defaults to True.

  Returns:
      pd.DataFrame: filtered ratings
  """
  ratings = ratings.merge(
    noteStatusHistory[[c.noteIdKey, c.createdAtMillisKey, c.classificationKey]],
    on=c.noteIdKey,
    how="left",
    suffixes=("", "_nsh"),
    unsafeAllowed=c.createdAtMillisKey,
  )

  createdAtMillisNSHKey = c.createdAtMillisKey + "_nsh"

  # rows in ratings that are on deleted notes, check if the note is in noteStatusHistory
  deletedNote = pd.isna(ratings[c.classificationKey])

  # deleted but in noteStatusHistory, keep
  deletedButInNSHNote = deletedNote & pd.notna(ratings[createdAtMillisNSHKey])
  # deleted and not in noteStatusHistory, remove
  deletedNotInNSHNote = deletedNote & pd.isna(ratings[createdAtMillisNSHKey])

  # rows in ratings that are on still available notes, check if the note says the tweet is misleading or not
  availableNote = pd.notna(ratings[c.classificationKey])

  # not deleted and says the tweet is misleading, keep
  notDeletedMisleadingNote = ratings[c.classificationKey] == c.notesSaysTweetIsMisleadingKey

  # not deleted and says the tweet is not misleading, check if it's after or before the new UI launch time
  notDeletedNotMisleadingNote = ratings[c.classificationKey] == c.noteSaysTweetIsNotMisleadingKey

  # not deleted, says the tweet is not misleading, and after new UI launch time, keep
  notDeletedNotMisleadingNewUINote = (ratings[c.classificationKey] == c.noteSaysTweetIsNotMisleadingKey) & (ratings[createdAtMillisNSHKey] > c.notMisleadingUILaunchTime)
  # not deleted, says the tweet is not misleading, and before new UI launch time, remove
  notDeletedNotMisleadingOldUINote = (ratings[c.classificationKey] == c.noteSaysTweetIsNotMisleadingKey) & (ratings[createdAtMillisNSHKey] <= c.notMisleadingUILaunchTime)

  if log:
    logger.info(
      f"Finished filtering misleading notes\n"
      f"Preprocess Data: Filter misleading notes, starting with {len(ratings)} ratings on {get_unique_size(ratings, c.noteIdKey)} notes"
    )
    logger.info(
      f"For {deletedNote.sum()} ratings on {get_unique_size(ratings, c.noteIdKey, rows=deletedNote)} deleted notes"
    )
    logger.info(
      f"  Keep {deletedButInNSHNote.sum()} ratings on {get_unique_size(ratings, c.noteIdKey, rows=deletedButInNSHNote)} deleted notes that are in noteStatusHistory (e.g., previously scored)"
    )
    logger.info(
      f"  Remove {deletedNotInNSHNote.sum()} ratings on {get_unique_size(ratings, c.noteIdKey, rows=deletedNotInNSHNote)} deleted notes that are not in noteStatusHistory (e.g., old)"
    )
    logger.info(
      f"For {availableNote.sum()} ratings on {get_unique_size(ratings, c.noteIdKey, rows=availableNote)} still available notes"
    )
    logger.info(
      f"  Keep {notDeletedMisleadingNote.sum()} ratings on {get_unique_size(ratings, c.noteIdKey, rows=notDeletedMisleadingNote)} available notes saying the associated tweet is misleading"
    )
    logger.info(
      f"  For {notDeletedNotMisleadingNote.sum()} ratings on {get_unique_size(ratings, c.noteIdKey, rows=notDeletedNotMisleadingNote)} available notes saying the associated tweet is not misleading"
    )
    logger.info(
      f"    Keep {notDeletedNotMisleadingNewUINote.sum()} ratings on {get_unique_size(ratings, c.noteIdKey, rows=notDeletedNotMisleadingNewUINote)} available and not misleading notes, and after the new UI launch time"
    )
    logger.info(
      f"    Remove {notDeletedNotMisleadingOldUINote.sum()} ratings on {get_unique_size(ratings, c.noteIdKey, rows=notDeletedNotMisleadingOldUINote)} available and not misleading notes, and before the new UI launch time"
    )
  
  # Validate expectation that all notes with ratings are either deleted or not deleted
  assert len(ratings) == (
    deletedNote.sum() + availableNote.sum()
  ), "rows of ratings must equal to the sum of ratings on deleted notes and ratings on available notes"
  assert get_unique_size(ratings, c.noteIdKey) == (
    get_unique_size(ratings, c.noteIdKey, rows=deletedNote) + get_unique_size(ratings, c.noteIdKey, rows=availableNote)
  ), "rows of notes must equal to the sum of deleted notes and available notes"

  # Validate expectation that all deleted notes must be either in noteStatusHistory or not in noteStatusHistory
  assert deletedNote.sum() == (
    deletedButInNSHNote.sum() + deletedNotInNSHNote.sum()
  ), "all ratings on deleted notes must be either in noteStatusHistory or not in noteStatusHistory"
  assert get_unique_size(ratings, c.noteIdKey, rows=deletedNote) == (
    get_unique_size(ratings, c.noteIdKey, rows=deletedButInNSHNote) + get_unique_size(ratings, c.noteIdKey, rows=deletedNotInNSHNote)
  ), "all deleted notes must be either in noteStatusHistory or not in noteStatusHistory"

  # Validate expectation that all available notes must either say Tweet Is Misleading or Tweet Is Not Misleading
  assert availableNote.sum() == (
    notDeletedMisleadingNote.sum() + notDeletedNotMisleadingNote.sum()
  ), "all ratings on available notes must either say Tweet Is Misleading or Tweet Is Not Misleading"
  assert get_unique_size(ratings, c.noteIdKey, rows=availableNote) == (
    get_unique_size(ratings, c.noteIdKey, rows=notDeletedMisleadingNote) + get_unique_size(ratings, c.noteIdKey, rows=notDeletedNotMisleadingNote)
  ), "all available notes must either say Tweet Is Misleading or Tweet Is Not Misleading"

  # Validate expectation that all available and not misleading notes must be either after or before the new UI launch time
  assert notDeletedNotMisleadingNote.sum() == (
    notDeletedNotMisleadingNewUINote.sum() + notDeletedNotMisleadingOldUINote.sum()
  ), "all ratings on available and not misleading notes must be either after or before the new UI launch time"
  assert get_unique_size(ratings, c.noteIdKey, rows=notDeletedNotMisleadingNote) == (
    get_unique_size(ratings, c.noteIdKey, rows=notDeletedNotMisleadingNewUINote) + get_unique_size(ratings, c.noteIdKey, rows=notDeletedNotMisleadingOldUINote)
  ), "all available and not misleading notes must be either after or before the new UI launch time"

  ratings = ratings[
    deletedButInNSHNote | notDeletedMisleadingNote | notDeletedNotMisleadingNewUINote
  ]
  ratings = ratings.drop(
    columns=[
      createdAtMillisNSHKey,
      c.classificationKey,
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

def compute_helpful_num(ratings: pd.DataFrame):
  """
  Populate the "helpfulNum" column.
    not helpful: 0.0
    somewhat helpful: 0.5
    helpful: 1.0
  """
  ratings.loc[:, c.helpfulNumKey] = np.nan
  ratings.loc[ratings[c.helpfulKey] == 1, c.helpfulNumKey] = 1
  ratings.loc[ratings[c.notHelpfulKey] == 1, c.helpfulNumKey] = 0
  ratings.loc[ratings[c.helpfulnessLevelKey] == c.notHelpfulValueTsv, c.helpfulNumKey] = 0
  ratings.loc[ratings[c.helpfulnessLevelKey] == c.somewhatHelpfulValueTsv, c.helpfulNumKey] = 0.5
  ratings.loc[ratings[c.helpfulnessLevelKey] == c.helpfulValueTsv, c.helpfulNumKey] = 1
  ratings = ratings.loc[~pd.isna(ratings[c.helpfulNumKey])]
  return ratings

def preprocess_data(
  notes: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  shouldFilterNotMisleadingNotes: bool = True,
  log: bool = True,
  ratingsOnly: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Populate helpfulNumKey, a unified column that merges the helpfulness answers from 
  the V1 and V2 rating forms together, as described in
  https://twitter.github.io/communitynotes/ranking-notes/#helpful-rating-mapping.

  Also, remove notes that indicate the associated tweet is not misleading, 
  if the `shouldFilterNotMisleadingNotes` flag is True.

  Args:
      notes (pd.DataFrame)
      ratings (pd.DataFrame)
      noteStatusHistory (pd.DataFrame)
      shouldFilterNotMisleadingNotes (bool, optional): Defaults to True.
      log (bool, optional): Defaults to True.
      ratingsOnly (bool, optional): Defaults to False

  Returns:
      notes (pd.DataFrame)
      ratings (pd.DataFrame)
      noteStatusHistory (pd.DataFrame)
  """
  if log:
    logger.info(
      f"Timestamp of latest rating in data: {pd.to_datetime(ratings[c.createdAtMillisKey], unit='ms').max()}",
    )
    if not ratingsOnly:
      logger.info(
        f"Timestamp of latest note in data: {pd.to_datetime(notes[c.createdAtMillisKey], unit='ms').max()}",
      )
    logger.info(
      f"Original row numbers from provided tsv files\n",
      f"  notes: {len(notes)}\n",
      f"  ratings: {len(ratings)}\n",
      f"  noteStatusHistory: {len(noteStatusHistory)}\n",
    )
  
  # each rating must have a unique (noteId, raterParticipantId) pair
  ratings = remove_duplicate_ratings(ratings)

  if log:
    logger.info(
      f"After removing duplicates, there are {len(notes)} notes and {len(ratings)} ratings from {get_unique_size(ratings, c.noteIdKey)} notes\n"
      f"  Thus, {len(notes) - get_unique_size(ratings, c.noteIdKey)} notes have no ratings yet, removed..."
    )

  # add a new column `helpfulNum` to `ratings`
  # `helpfulNum` is a unified column that merges the helpfulness answers from the V1 and V2 rating forms together
  # `helpfulNum` is a float, with 0.0 for not helpful, 0.5 for somewhat helpful, and 1.0 for helpful
  ratings.loc[:, c.helpfulNumKey] = np.nan
  ratings.loc[ratings[c.helpfulKey] == 1, c.helpfulNumKey] = 1
  ratings.loc[ratings[c.notHelpfulKey] == 1, c.helpfulNumKey] = 0
  ratings.loc[ratings[c.helpfulnessLevelKey] == c.notHelpfulValueTsv, c.helpfulNumKey] = 0
  ratings.loc[ratings[c.helpfulnessLevelKey] == c.somewhatHelpfulValueTsv, c.helpfulNumKey] = 0.5
  ratings.loc[ratings[c.helpfulnessLevelKey] == c.helpfulValueTsv, c.helpfulNumKey] = 1
  num_raw_ratings = len(ratings)
  ratings = ratings.loc[~pd.isna(ratings[c.helpfulNumKey])]

  if log:
    logger.info(
      f"After populating helpfulNumKey, there are {len(ratings)} ratings from {get_unique_size(ratings, c.noteIdKey)} notes\n"
      f"  Thus, {num_raw_ratings - len(ratings)} ratings have no helpfulness labels (i.e., helpfulKey=0 and notHelpfulKey=0), removed..."
    )
  if ratingsOnly:
    return pd.DataFrame(), ratings, pd.DataFrame()

  # each note must have a unique noteId
  notes = remove_duplicate_notes(notes)

  notes[c.tweetIdKey] = notes[c.tweetIdKey].astype(str)

  # merge `notes` with `noteStatusHistory`
  # `noteStatusHistory` contains the status of all previously scored notes, including deleted ones
  # `notes` contains currently available notes, including the new ones (from last release timestamp) but excluding deleted ones
  # after the merge, `noteStatusHistory` will have a new column called `classification`, populated from `notes` dataframe
  # `classification` is the status of the note, which can be one of the following:
  # - MISINFORMED_OR_POTENTIALLY_MISLEADING
  # - NOT_MISLEADING
  # - NaN (if the note is deleted)
  noteStatusHistory = note_status_history.merge_note_info(noteStatusHistory, notes)

  if shouldFilterNotMisleadingNotes:
    ratings = _filter_misleading_notes(notes, ratings, noteStatusHistory, log)

  if log:
    logger.info(
      "After data preprocess, Num Ratings: %d, Num Unique Notes Rated: %d, Num Unique Raters: %d\n"
      % (
        len(ratings),
        get_unique_size(ratings, c.noteIdKey),
        get_unique_size(ratings, c.raterParticipantIdKey),
      )
    )
  return notes, ratings, noteStatusHistory


def filter_ratings(
  ratings: pd.DataFrame,
  minNumRatingsPerRater: int,
  minNumRatersPerNote: int,
  log: bool = True,
) -> pd.DataFrame:
  """Apply min number of ratings for raters & notes. Instead of iterating these filters
  until convergence, simply stop after going back and force once.

  Args:
      ratings: All ratings from Community Notes contributors.
      minNumRatingsPerRater: Minimum number of ratings which a rater must produce to be
        included in scoring.  Raters with fewer ratings are removed.
      minNumRatersPerNote: Minimum number of ratings which a note must have to be included
        in scoring.  Notes with fewer ratings are removed.
      log: Debug output. Defaults to True.

  Returns:
      pd.DataFrame: filtered ratings
  """

  def filter_notes(ratings):
    note_counts = ratings[c.noteIdKey].value_counts()
    valid_notes = note_counts[note_counts >= minNumRatersPerNote].index
    return ratings[ratings[c.noteIdKey].isin(valid_notes)]

  def filter_raters(ratings):
    rater_counts = ratings[c.raterParticipantIdKey].value_counts()
    valid_raters = rater_counts[rater_counts >= minNumRatingsPerRater].index
    return ratings[ratings[c.raterParticipantIdKey].isin(valid_raters)]

  ratings = filter_notes(ratings)
  ratings = filter_raters(ratings)
  ratings = filter_notes(ratings)

  if log:
    # Log final details
    unique_notes = ratings[c.noteIdKey].nunique()
    unique_raters = ratings[c.raterParticipantIdKey].nunique()
    logger.info(
      f"After applying min {minNumRatingsPerRater} ratings per rater and min {minNumRatersPerNote} raters per note: \n"
      + f"Num Ratings: {len(ratings)}, Num Unique Notes Rated: {unique_notes}, Num Unique Raters: {unique_raters}"
    )

  return ratings


def write_prescoring_output(
  prescoringNoteModelOutput: pd.DataFrame,
  prescoringRaterModelOutput: pd.DataFrame,
  noteTopicClassifier: Pipeline,
  prescoringMetaOutput: c.PrescoringMetaOutput,
  prescoringScoredNotesOutput: Optional[pd.DataFrame],
  noteModelOutputPath: str,
  raterModelOutputPath: str,
  noteTopicClassifierPath: str,
  prescoringMetaOutputPath: str,
  prescoringScoredNotesOutputPath: Optional[str],
  headers: bool = True,
):
  prescoringNoteModelOutput = prescoringNoteModelOutput[c.prescoringNoteModelOutputTSVColumns]
  assert all(prescoringNoteModelOutput.columns == c.prescoringNoteModelOutputTSVColumns)
  write_tsv_local(prescoringNoteModelOutput, noteModelOutputPath, headers=headers)

  prescoringRaterModelOutput = prescoringRaterModelOutput[c.prescoringRaterModelOutputTSVColumns]
  assert all(prescoringRaterModelOutput.columns == c.prescoringRaterModelOutputTSVColumns)
  write_tsv_local(prescoringRaterModelOutput, raterModelOutputPath, headers=headers)

  if prescoringScoredNotesOutput is not None and prescoringScoredNotesOutputPath is not None:
    write_tsv_local(prescoringScoredNotesOutput, prescoringScoredNotesOutputPath, headers=headers)

  joblib.dump(noteTopicClassifier, noteTopicClassifierPath)
  joblib.dump(prescoringMetaOutput, prescoringMetaOutputPath)


def write_tsv_local(df: pd.DataFrame, path: str, headers: bool = True) -> None:
  """Write DF as a TSV stored to local disk.

  Note that index=False (so the index column will not be written to disk), and header=True
  (so the first line of the output will contain row names).

  Args:
    df: pd.DataFrame to write to disk.
    path: location of file on disk.
  """

  assert path is not None
  assert df.to_csv(path, index=False, header=headers, sep="\t") is None


def write_parquet_local(
  df: pd.DataFrame, path: str, compression: str = "snappy", engine: str = "pyarrow"
) -> None:
  """Write DF as a parquet file stored to local disk. Compress with snappy
  and use pyarrow engine.

  Args:
    df: pd.DataFrame to write to disk.
    path: location of file on disk.
    compression: compression algorithm to use. Defaults to 'snappy'.
    engine: engine to use. Defaults to 'pyarrow'.
  """

  assert path is not None
  df.to_parquet(path, compression=compression, engine=engine)


class CommunityNotesDataLoader(ABC):
  """Base class which local and prod data loaders extend.

  The DataLoader base class stores necessary files and defines "get_data" function which can be passed to
  parallel scoring
  """

  def __init__(self) -> None:
    """Configure a new CommunityNotesDataLoader object.

    Args:
      local (bool, optional): if not None, seed value to ensure deterministic execution
    """

  @abstractmethod
  def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns notes, ratings, noteStatusHistory, and userEnrollment DataFrames"""

  @abstractmethod
  def get_prescoring_model_output(self) -> pd.DataFrame:
    """Returns first round rater model output."""


class LocalDataLoader(CommunityNotesDataLoader):
  def __init__(
    self,
    notesPath: str,
    ratingsPath: str,
    noteStatusHistoryPath: str,
    userEnrollmentPath: str,
    headers: bool,
    shouldFilterNotMisleadingNotes: bool = True,
    log: bool = True,
    prescoringNoteModelOutputPath: Optional[str] = None,
    prescoringRaterModelOutputPath: Optional[str] = None,
    prescoringNoteTopicClassifierPath: Optional[str] = None,
    prescoringMetaOutputPath: Optional[str] = None,
  ) -> None:
    """
    Args:
        notesPath (str): file path
        ratingsPath (str): file path
        noteStatusHistoryPath (str): file path
        userEnrollmentPath (str): file path
        headers: If true, expect first row of input files to be headers.
        shouldFilterNotMisleadingNotes (bool, optional): Throw out not-misleading notes if True. Defaults to True.
        log (bool, optional): Print out debug output. Defaults to True.
    """
    self.notesPath = notesPath
    self.ratingsPath = ratingsPath
    self.noteStatusHistoryPath = noteStatusHistoryPath
    self.userEnrollmentPath = userEnrollmentPath
    self.prescoringNoteModelOutputPath = prescoringNoteModelOutputPath
    self.prescoringRaterModelOutputPath = prescoringRaterModelOutputPath
    self.prescoringNoteTopicClassifierPath = prescoringNoteTopicClassifierPath
    self.prescoringMetaOutputPath = prescoringMetaOutputPath
    self.headers = headers
    self.shouldFilterNotMisleadingNotes = shouldFilterNotMisleadingNotes
    self.log = log

  def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """All-in-one function for reading Birdwatch notes and ratings from TSV files.
    It does both reading and pre-processing.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: notes, ratings, noteStatusHistory, userEnrollment
    """
    notes, ratings, noteStatusHistory, userEnrollment = read_from_tsv(
      self.notesPath,
      self.ratingsPath,
      self.noteStatusHistoryPath,
      self.userEnrollmentPath,
      self.headers,
    )
    notes, ratings, noteStatusHistory = preprocess_data(
      notes, ratings, noteStatusHistory, self.shouldFilterNotMisleadingNotes, self.log
    )
    return notes, ratings, noteStatusHistory, userEnrollment

  def get_prescoring_model_output(
    self,
  ) -> Tuple[pd.DataFrame, pd.DataFrame, Pipeline, c.PrescoringMetaOutput]:
    logger.info(
      f"Attempting to read prescoring model output from {self.prescoringNoteModelOutputPath}, {self.prescoringRaterModelOutputPath}, {self.prescoringNoteTopicClassifierPath}, {self.prescoringMetaOutputPath}"
    )
    if self.prescoringRaterModelOutputPath is None:
      prescoringRaterModelOutput = None
    else:
      prescoringRaterModelOutput = tsv_reader(
        self.prescoringRaterModelOutputPath,
        c.prescoringRaterModelOutputTSVTypeMapping,
        c.prescoringRaterModelOutputTSVColumns,
        header=self.headers,
      )
      assert len(prescoringRaterModelOutput.columns) == len(
        c.prescoringRaterModelOutputTSVColumns
      ) and all(prescoringRaterModelOutput.columns == c.prescoringRaterModelOutputTSVColumns), (
        f"Rater model output columns don't match: \n{[col for col in prescoringRaterModelOutput.columns if not col in c.prescoringRaterModelOutputTSVColumns]} are extra columns, "
        + f"\n{[col for col in c.prescoringRaterModelOutputTSVColumns if not col in prescoringRaterModelOutput.columns]} are missing."
      )  # ensure constants file is up to date.

    if self.prescoringNoteModelOutputPath is None:
      prescoringNoteModelOutput = None
    else:
      prescoringNoteModelOutput = tsv_reader(
        self.prescoringNoteModelOutputPath,
        c.prescoringNoteModelOutputTSVTypeMapping,
        c.prescoringNoteModelOutputTSVColumns,
        header=self.headers,
      )
      assert len(prescoringNoteModelOutput.columns) == len(
        c.prescoringNoteModelOutputTSVColumns
      ) and all(prescoringNoteModelOutput.columns == c.prescoringNoteModelOutputTSVColumns), (
        f"Note model output columns don't match: \n{[col for col in prescoringNoteModelOutput.columns if not col in c.prescoringNoteModelOutputTSVColumns]} are extra columns, "
        + f"\n{[col for col in c.prescoringNoteModelOutputTSVColumns if not col in prescoringNoteModelOutput.columns]} are missing."
      )  # ensure constants file is up to date.

    if self.prescoringNoteTopicClassifierPath is None:
      prescoringNoteTopicClassifier = None
    else:
      prescoringNoteTopicClassifier = joblib.load(self.prescoringNoteTopicClassifierPath)
    assert type(prescoringNoteTopicClassifier) == Pipeline

    if self.prescoringMetaOutputPath is None:
      prescoringMetaOutput = None
    else:
      prescoringMetaOutput = joblib.load(self.prescoringMetaOutputPath)
    assert type(prescoringMetaOutput) == c.PrescoringMetaOutput

    return (
      prescoringNoteModelOutput,
      prescoringRaterModelOutput,
      prescoringNoteTopicClassifier,
      prescoringMetaOutput,
    )


def filter_input_data_for_testing(
  notes: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  cutoffTimestampMillis: Optional[int] = None,
  excludeRatingsAfterANoteGotFirstStatusPlusNHours: Optional[int] = None,
  daysInPastToApplyPostFirstStatusFiltering: Optional[int] = 14,
  filterPrescoringInputToSimulateDelayInHours: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """
  Args:
    cutoffTimestampMillis: filter all notes and ratings after this time.

    excludeRatingsAfterANoteGotFirstStatusPlusNHours: set to 0 to throw out all
      ratings after a note was first CRH. Set to None to turn off.
    daysInPastToApplyPostFirstStatusFiltering: only apply the previous
      filter to notes created in the last this-many days.

    filterPrescoringInputToSimulateDelayInHours: Optional[int]: for system tests,
      simulate final scoring running this many hours after prescoring.

  Returns: notes, ratings, prescoringNotesInput, prescoringRatingsInput
  """
  logger.info(
    f"""Called filter_input_data_for_testing.
        Notes: {len(notes)}, Ratings: {len(ratings)}. Max note createdAt: {pd.to_datetime(notes[c.createdAtMillisKey].max(), unit='ms')}; Max rating createAt: {pd.to_datetime(ratings[c.createdAtMillisKey].max(), unit='ms')}"""
  )

  notes, ratings = filter_notes_and_ratings_after_particular_timestamp_millis(
    notes, ratings, cutoffTimestampMillis
  )
  logger.info(
    f"""After filtering notes and ratings after particular timestamp (={cutoffTimestampMillis}). 
        Notes: {len(notes)}, Ratings: {len(ratings)}. Max note createdAt: {pd.to_datetime(notes[c.createdAtMillisKey].max(), unit='ms')}; Max rating createAt: {pd.to_datetime(ratings[c.createdAtMillisKey].max(), unit='ms')}"""
  )

  ratings = filter_ratings_after_first_status_plus_n_hours(
    ratings,
    noteStatusHistory,
    excludeRatingsAfterANoteGotFirstStatusPlusNHours,
    daysInPastToApplyPostFirstStatusFiltering,
  )
  logger.info(
    f"""After filtering ratings after first status (plus {excludeRatingsAfterANoteGotFirstStatusPlusNHours} hours) for notes created in last {daysInPastToApplyPostFirstStatusFiltering} days. 
        Notes: {len(notes)}, Ratings: {len(ratings)}. Max note createdAt: {pd.to_datetime(notes[c.createdAtMillisKey].max(), unit='ms')}; Max rating createAt: {pd.to_datetime(ratings[c.createdAtMillisKey].max(), unit='ms')}"""
  )

  (
    prescoringNotesInput,
    prescoringRatingsInput,
  ) = filter_prescoring_input_to_simulate_delay_in_hours(
    notes, ratings, filterPrescoringInputToSimulateDelayInHours
  )
  logger.info(
    f"""After filtering prescoring notes and ratings to simulate a delay of {filterPrescoringInputToSimulateDelayInHours} hours: 
        Notes: {len(prescoringNotesInput)}, Ratings: {len(prescoringRatingsInput)}. Max note createdAt: {pd.to_datetime(prescoringNotesInput[c.createdAtMillisKey].max(), unit='ms')}; Max rating createAt: {pd.to_datetime(prescoringRatingsInput[c.createdAtMillisKey].max(), unit='ms')}"""
  )

  return notes, ratings, prescoringNotesInput, prescoringRatingsInput


def filter_ratings_after_first_status_plus_n_hours(
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  excludeRatingsAfterANoteGotFirstStatusPlusNHours: Optional[int] = None,
  daysInPastToApplyPostFirstStatusFiltering: Optional[int] = 14,
) -> pd.DataFrame:
  if excludeRatingsAfterANoteGotFirstStatusPlusNHours is None:
    return ratings

  if daysInPastToApplyPostFirstStatusFiltering is None:
    daysInPastToApplyPostFirstStatusFiltering = 14

  ratingCutoffTimeMillisKey = "ratingCutoffTimeMillis"

  # First: determine out which notes to apply this to (created in past
  #   daysInPastToApplyPostFirstStatusFiltering days)
  millisToLookBack = daysInPastToApplyPostFirstStatusFiltering * 24 * 60 * 60 * 1000
  cutoffTimeMillis = noteStatusHistory[c.createdAtMillisKey].max() - millisToLookBack
  nshToFilter = noteStatusHistory[noteStatusHistory[c.createdAtMillisKey] > cutoffTimeMillis]
  logger.info(
    f"  Notes to apply the post-first-status filter for (from last {daysInPastToApplyPostFirstStatusFiltering} days): {len(nshToFilter)}"
  )
  nshToFilter[ratingCutoffTimeMillisKey] = nshToFilter[
    c.timestampMillisOfNoteFirstNonNMRLabelKey
  ] + (excludeRatingsAfterANoteGotFirstStatusPlusNHours * 60 * 60 * 1000)

  # Next: join their firstStatusTime from NSH with their ratings
  ratingsWithNSH = ratings.merge(
    nshToFilter[[c.noteIdKey, ratingCutoffTimeMillisKey]], on=c.noteIdKey, how="left"
  )
  # And then filter out ratings made after that time. Don't filter any ratings for notes with
  #   nan cutoff time.
  ratingsWithNSH[ratingCutoffTimeMillisKey].fillna(
    ratingsWithNSH[c.createdAtMillisKey].max() + 1, inplace=True
  )
  ratingsWithNSH = ratingsWithNSH[
    ratingsWithNSH[c.createdAtMillisKey] < ratingsWithNSH[ratingCutoffTimeMillisKey]
  ]
  return ratingsWithNSH.drop(columns=[ratingCutoffTimeMillisKey])


def filter_notes_and_ratings_after_particular_timestamp_millis(
  notes: pd.DataFrame,
  ratings: pd.DataFrame,
  cutoffTimestampMillis: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  if cutoffTimestampMillis is not None:
    notes = notes[notes[c.createdAtMillisKey] <= cutoffTimestampMillis].copy()
    ratings = ratings[ratings[c.createdAtMillisKey] <= cutoffTimestampMillis].copy()
  return notes, ratings


def filter_prescoring_input_to_simulate_delay_in_hours(
  notes: pd.DataFrame,
  ratings: pd.DataFrame,
  filterPrescoringInputToSimulateDelayInHours: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  if filterPrescoringInputToSimulateDelayInHours is not None:
    latestRatingMillis = ratings[c.createdAtMillisKey].max()
    cutoffMillis = latestRatingMillis - (
      filterPrescoringInputToSimulateDelayInHours * 60 * 60 * 1000
    )
    logger.info(
      f"""
      Filtering input data for prescoring to simulate running prescoring earlier than final scoring.
      Latest rating timestamp: {pd.to_datetime(latestRatingMillis, unit='ms')}
      Cutoff timestamp: {pd.to_datetime(cutoffMillis, unit='ms')} ({filterPrescoringInputToSimulateDelayInHours} hours before)
    """
    )
    prescoringNotesInput = notes[notes[c.createdAtMillisKey] < cutoffMillis].copy()
    prescoringRatingsInput = ratings[ratings[c.createdAtMillisKey] < cutoffMillis].copy()
  else:
    prescoringNotesInput = notes
    prescoringRatingsInput = ratings
  return prescoringNotesInput, prescoringRatingsInput
