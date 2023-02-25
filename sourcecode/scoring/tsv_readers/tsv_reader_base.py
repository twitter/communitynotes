import abc
import typing

import pandas as pd


class TSVReaderBase(abc.ABC):
  def __init__(self, path: str):
    self._path = path

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
        filepath_or_buffer=self._path,
        sep=self._separator,
        dtype=self._datatype_mapping,
        names=self._columns,
        header=0 if has_header else None,
      )
    except (ValueError, IndexError) as exc:
      raise ValueError(f"Invalid TSV input at {self._path}.") from exc

    self._verify_read_columns(results)
    return results

  def _validate_num_columns_in_file(self) -> None:
    """Validate the number of columns in the file, against the expected amount.
    The validation is done on the raw file, by examining the first line and splitting it manually.
    """
    with open(self._path, "r") as tsv_file:
      num_actual_columns = len(tsv_file.readline().split(self._separator))
      num_expected_columns = len(self._columns)
      if num_actual_columns != num_expected_columns:
        raise ValueError(f"Expected {num_expected_columns} columns in {self._path} but got {num_actual_columns}.")

  def _verify_read_columns(self, results: pd.DataFrame) -> None:
    """
    Verify that the read file has exactly the columns we are expecting.
    Args:
        results (DataFrame): The DataFrame returned by the read method.
    Raises:
        ValueError: if there are any extra or missing columns in the DataFrame.
    """
    actual_columns = set(results.columns.values)
    expected_columns = set(self._columns)
    if actual_columns != expected_columns:
      raise ValueError(
        f"Columns don't match for {self._path}:\n"
        f"{actual_columns - expected_columns} are extra columns,\n"
        f"{expected_columns - actual_columns} are missing.",
      )

  @property
  def _separator(self) -> str:
    return "\t"

  @property
  @abc.abstractmethod
  def _datatype_mapping(self) -> typing.Mapping[str, typing.Type]:
    pass

  @property
  def _columns(self) -> typing.Tuple[str, ...]:
    return tuple(self._datatype_mapping.keys())
