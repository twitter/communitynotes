from abc import ABC, abstractmethod
from io import StringIO
import logging
import os
from typing import Dict, List, Optional, Tuple

from . import constants as c, note_status_history
from .pandas_utils import get_df_info
from .pflip_plus_model import PFlipPlusModel

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
    StringIO(ratingsStr),
    sep="\t",
    names=c.ratingTSVColumns,
    dtype=c.ratingTSVTypeMapping,
  )
  noteStatusHistory = pd.read_csv(
    StringIO(noteStatusHistoryStr),
    sep="\t",
    names=c.noteStatusHistoryTSVColumns,
    dtype=c.noteStatusHistoryTSVTypeMapping,
  )

  return notes, ratings, noteStatusHistory


def read_prescoring_from_strings(
  noteModelOutputDataStr: str, raterModelOutputDataStr: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  noteModelOutput = pd.read_csv(
    StringIO(noteModelOutputDataStr), sep="\t", dtype=c.noteModelOutputTSVTypeMapping
  )
  raterModelOutput = pd.read_csv(StringIO(raterModelOutputDataStr), sep="\t")

  return noteModelOutput, raterModelOutput


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
          [
            pd.StringDtype(),
            pd.BooleanDtype(),
            pd.Int64Dtype(),
            pd.Int32Dtype(),
            "boolean",
          ]
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
      notesPath,
      c.noteTSVTypeMapping,
      c.noteTSVColumns,
      header=headers,
      convertNAToNone=False,
    )
    assert len(notes.columns) == len(c.noteTSVColumns) and all(notes.columns == c.noteTSVColumns), (
      f"note columns don't match: \n{[col for col in notes.columns if not col in c.noteTSVColumns]} are extra columns, "
      + f"\n{[col for col in c.noteTSVColumns if not col in notes.columns]} are missing."
    )  # ensure constants file is up to date.

  if ratingsPath is None:
    ratings = None
  else:
    ratings = tsv_reader(
      ratingsPath,
      c.ratingTSVTypeMapping,
      c.ratingTSVColumns,
      header=headers,
      convertNAToNone=False,
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


def _filter_misleading_notes(
  notes: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  log: bool = True,
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

  if log:
    logger.info(
      f"Preprocess Data: Filter misleading notes, starting with {len(ratings)} ratings on {len(np.unique(ratings[c.noteIdKey]))} notes"
    )
    logger.info(
      f"  Keeping {ratings[notDeletedMisleadingKey].sum()} ratings on {len(np.unique(ratings.loc[ratings[notDeletedMisleadingKey],c.noteIdKey]))} misleading notes"
    )
    logger.info(
      f"  Keeping {ratings[deletedButInNSHKey].sum()} ratings on {len(np.unique(ratings.loc[ratings[deletedButInNSHKey],c.noteIdKey]))} deleted notes that were previously scored (in note status history)"
    )
    logger.info(
      f"  Removing {notDeletedNotMisleadingOldUI.sum()} ratings on {len(np.unique(ratings.loc[notDeletedNotMisleadingOldUI, c.noteIdKey]))} older notes that aren't deleted, but are not-misleading."
    )
    logger.info(
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


def tag_high_volume_raters(ratings: pd.DataFrame, quantile=0.999):
  """Set field indicating whether a rating came from a high volume rater."""
  # Include all ratings if running on test data
  if ratings[c.noteIdKey].nunique() < c.minNumNotesForProdData:
    ratings[c.highVolumeRaterKey] = False
    return ratings

  # Identify high volume raters over last 1, 7 and 28 days
  raters = ratings[[c.raterParticipantIdKey]].drop_duplicates()
  logger.info(f"Total raters: {len(raters)}")
  highVolRaters = set()
  logger.info("Identifying high volume raters")
  logger.info(f"Total ratings: {len(ratings)}")
  for numDays in [7, 28]:
    cutoff = ratings[c.createdAtMillisKey].max() - (numDays * 1000 * 60 * 60 * 24)
    counts = raters.merge(
      ratings[ratings[c.createdAtMillisKey] > cutoff][c.raterParticipantIdKey]
      .value_counts()
      .to_frame()
      .reset_index(drop=False)
      .astype({"count": pd.Int64Dtype()})
    )
    highVolThrehsold = counts["count"].quantile(quantile)
    logger.info(f"High volume threshold for {numDays} days: {highVolThrehsold}")
    highVolRaters |= set(counts[counts["count"] > highVolThrehsold][c.raterParticipantIdKey])
    logger.info(
      f"High volume raters for {numDays} days: {(counts['count'] > highVolThrehsold).sum()}"
    )
  ratings[c.highVolumeRaterKey] = ratings[c.raterParticipantIdKey].isin(highVolRaters)
  logger.info(f"Total high volume raters: {len(highVolRaters)}")
  logger.info(f"Total ratings: {len(ratings)}")
  logger.info(f"Total ratings from high volume raters: {ratings[c.highVolumeRaterKey].sum()}")
  return ratings


def preprocess_data(
  notes: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  shouldFilterNotMisleadingNotes: bool = True,
  log: bool = True,
  ratingsOnly: bool = False,
  basic: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Populate helpfulNumKey, a unified column that merges the helpfulness answers from
  the V1 and V2 rating forms together, as described in
  https://twitter.github.io/communitynotes/ranking-notes/#helpful-rating-mapping.

  Also, filter notes that indicate the Tweet is misleading, if the flag is True.

  Args:
      notes (pd.DataFrame)
      ratings (pd.DataFrame)
      noteStatusHistory (pd.DataFrame)
      shouldFilterNotMisleadingNotes (bool, optional): Defaults to True.
      log (bool, optional): Defaults to True.
      ratingsOnly (bool, optional): Defaults to False
      basic (bool, optional): Defaults to false

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

  if len(ratings) > 0:
    ratings = remove_duplicate_ratings(ratings)
    ratings = compute_helpful_num(ratings)
    if not basic:
      ratings = tag_high_volume_raters(ratings)
      ratings[c.ratingSourceBucketedKey] = ratings[c.ratingSourceBucketedKey].astype("category")
    ratings[c.helpfulnessLevelKey] = ratings[c.helpfulnessLevelKey].astype("category")

  if ratingsOnly:
    return pd.DataFrame(), ratings, pd.DataFrame()

  notes = remove_duplicate_notes(notes)

  notes[c.tweetIdKey] = notes[c.tweetIdKey].astype(str)

  noteStatusHistory = note_status_history.merge_note_info(noteStatusHistory, notes)

  if shouldFilterNotMisleadingNotes:
    ratings = _filter_misleading_notes(notes, ratings, noteStatusHistory, log)

  if log:
    logger.info(
      "Num Ratings: %d, Num Unique Notes Rated: %d, Num Unique Raters: %d"
      % (
        len(ratings),
        len(np.unique(ratings[c.noteIdKey])),
        len(np.unique(ratings[c.raterParticipantIdKey])),
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
  pflipClassifier: PFlipPlusModel,
  prescoringMetaOutput: c.PrescoringMetaOutput,
  prescoringScoredNotesOutput: Optional[pd.DataFrame],
  noteModelOutputPath: str,
  raterModelOutputPath: str,
  noteTopicClassifierPath: str,
  pflipClassifierPath: str,
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
  with open(pflipClassifierPath, "wb") as handle:
    handle.write(pflipClassifier.serialize())
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
    prescoringPflipClassifierPath: Optional[str] = None,
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
    self.prescoringPflipClassifierPath = prescoringPflipClassifierPath
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
  ) -> Tuple[pd.DataFrame, pd.DataFrame, Pipeline, PFlipPlusModel, c.PrescoringMetaOutput]:
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

    if self.prescoringPflipClassifierPath is None:
      prescoringPflipClassifier = None
    else:
      prescoringPflipClassifier = joblib.load(self.prescoringPflipClassifierPath)
    assert type(prescoringPflipClassifier) == PFlipPlusModel

    if self.prescoringMetaOutputPath is None:
      prescoringMetaOutput = None
    else:
      prescoringMetaOutput = joblib.load(self.prescoringMetaOutputPath)
    assert type(prescoringMetaOutput) == c.PrescoringMetaOutput

    return (
      prescoringNoteModelOutput,
      prescoringRaterModelOutput,
      prescoringNoteTopicClassifier,
      prescoringPflipClassifier,
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
