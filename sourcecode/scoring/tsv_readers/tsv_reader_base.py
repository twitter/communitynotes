import abc
import io
import pathlib
import typing

import pandas as pd
from pandas import errors as pd_errors

from scoring.tsv_readers import exceptions


class TSVReaderBase(abc.ABC):
  _TSV_SEPERATOR = "\t"

  def __init__(self, buffer: typing.IO, location: str):
    if not buffer.seekable():
      # We read the first line twice for pre-parsing validation, se we must seek back to 0 afterwards.
      raise exceptions.IoInputNotSeekableError(tsv_reader=self)

    self._input_buffer = buffer
    self._location = location

  @property
  def reader_name(self) -> str:
    return self.__class__.__name__

  @property
  def location(self) -> str:
    return self._location

  @classmethod
  def from_file(cls, path: typing.Union[str, pathlib.Path], encoding="utf8", has_header: bool = True) -> pd.DataFrame:
    with open(file=path, mode="r", encoding=encoding) as tsv_file:
      return cls(buffer=tsv_file, location=path).read(has_header=has_header)

  @classmethod
  def from_string(cls, raw: str, has_header: bool = True) -> pd.DataFrame:
    return cls(buffer=io.StringIO(raw), location="string").read(has_header=has_header)

  @classmethod
  def from_buffer(cls, buffer: typing.IO, has_header: bool = True) -> pd.DataFrame:
    return cls(buffer=buffer, location="io").read(has_header=has_header)

  def read(self, has_header: bool = True) -> pd.DataFrame:
    """Reads a tsv file into a pandas DataFrame, and validates the columns read are expected.

    Args:
      has_header: bool indicating whether the input has a header row.

    Returns:
      The data read from the provided file path, as a DataFrame.
    """
    try:
      self._validate_num_columns_in_file()
      results = pd.read_csv(
        filepath_or_buffer=self._input_buffer,
        sep=self._TSV_SEPERATOR,
        dtype=self._datatype_mapping,
        names=None if has_header else self._columns,
        header=0 if has_header else None,
      )
      self._verify_read_columns(results)
    except exceptions.TSVReaderError:
      raise
    except pd_errors.EmptyDataError:
      if not has_header:
        return pd.DataFrame(columns=self._columns)
      raise exceptions.EmptyInputError(tsv_reader=self)
    except Exception as exc:
      raise exceptions.TSVReaderError(tsv_reader=self) from exc

    return results

  def _validate_num_columns_in_file(self) -> None:
    """Validate the number of columns in the file, against the expected amount.
    The validation is done on the raw file, by examining the first line and splitting it manually.
    """
    first_line = self._read_first_line()
    if not first_line:
      # Empty file should pass the validation.
      return

    num_actual_columns = len(first_line.split(self._TSV_SEPERATOR))
    num_expected_columns = len(self._columns)
    if num_actual_columns != num_expected_columns:
      raise exceptions.UnexpectedColumnCountError(
        num_expected_columns=num_expected_columns,
        num_actual_columns=num_actual_columns,
        tsv_reader=self,
      )

  def _read_first_line(self) -> str:
    first_line = self._input_buffer.readline()
    self._input_buffer.seek(0)
    return first_line

  def _verify_read_columns(self, results: pd.DataFrame) -> None:
    """
    Verify that the read file has exactly the columns we are expecting.
    Args:
        results (DataFrame): The DataFrame returned by the read method.
    Raises:
        UnexpectedColumnsError: if there are any extra or missing columns in the DataFrame.
    """
    actual_columns = set(results.columns.values)
    expected_columns = set(self._columns)
    if actual_columns != expected_columns:
      raise exceptions.UnexpectedColumnsError(
        extra_columns=actual_columns - expected_columns,
        missing_columns=expected_columns - actual_columns,
        tsv_reader=self,
      )

  @property
  @abc.abstractmethod
  def _datatype_mapping(self) -> typing.Mapping[str, typing.Type]:
    pass

  @property
  def _columns(self) -> typing.Tuple[str, ...]:
    return tuple(self._datatype_mapping.keys())

  def __repr__(self) -> str:
    return f"{self.reader_name} from {self.location}"
