import os
import tempfile
import typing

import pytest

from scoring.tsv_readers import exceptions
from scoring.tsv_readers import tsv_reader_base


class TSVReader(tsv_reader_base.TSVReaderBase):
  DATATYPE_MAPPING = {
    "id": int,
    "fahrenheit_temp": float,
    "celsius_temp": float,
    "is_valid": bool,
    "note": str,
  }

  @property
  def _datatype_mapping(self) -> typing.Mapping[str, typing.Type]:
    return self.DATATYPE_MAPPING


class TestTSVReader:
  _VALID_HEADERS = tuple(TSVReader.DATATYPE_MAPPING.keys())
  _VALID_DATA = (
    (1, 32.0, 0.0, True, "Freezing"),
    (2, 212.0, 100.0, True, "Boiling"),
    (3, -100.2, 2.2, False, "Bad reading"),
  )

  def setup_method(self, _: typing.Callable) -> None:
    temp_fd, self._filepath = tempfile.mkstemp(text=True)
    os.close(temp_fd)

  def teardown_method(self, _: typing.Callable) -> None:
    os.remove(self._filepath)

  def test_read_from_string(self) -> None:
    self._write_file(headers=self._VALID_HEADERS, data=self._VALID_DATA)
    with open(file=self._filepath, mode="r") as temp:
      df = TSVReader.from_string(temp.read())
    assert df.columns.values.tolist() == list(self._VALID_HEADERS)
    assert len(df) == len(self._VALID_DATA)

  def test_read_from_buffer(self) -> None:
    self._write_file(headers=self._VALID_HEADERS, data=self._VALID_DATA)
    with open(file=self._filepath, mode="r") as temp:
      df = TSVReader.from_buffer(temp)
    assert df.columns.values.tolist() == list(self._VALID_HEADERS)
    assert len(df) == len(self._VALID_DATA)

  def test_read_from_file(self) -> None:
    self._write_file(headers=self._VALID_HEADERS, data=self._VALID_DATA)

    df = TSVReader.from_file(path=self._filepath)
    assert df.columns.values.tolist() == list(self._VALID_HEADERS)
    assert len(df) == len(self._VALID_DATA)

  def test_read_from_file_with_no_headers_row(self) -> None:
    self._write_file(headers=None, data=self._VALID_DATA)

    df = TSVReader.from_file(path=self._filepath, has_header=False)
    assert df.columns.values.tolist() == list(self._VALID_HEADERS)
    assert len(df) == len(self._VALID_DATA)

  def test_read_from_empty_file(self) -> None:
    with pytest.raises(exceptions.EmptyInputError):
      TSVReader.from_file(path=self._filepath, has_header=True)
    df = TSVReader.from_file(path=self._filepath, has_header=False)
    assert len(df) == 0

  def test_headers_count_mismatch(self) -> None:
    headers = ("invalid_column",)
    data = ((1,), (2,))
    self._write_file(headers=headers, data=data)

    with pytest.raises(exceptions.UnexpectedColumnCountError) as exc:
      TSVReader.from_file(path=self._filepath)

    assert exc.value.tsv_reader_name == TSVReader.__name__
    assert exc.value.input_location == self._filepath
    assert exc.value.num_expected_columns == len(TSVReader.DATATYPE_MAPPING)
    assert exc.value.num_actual_columns == len(headers)

  def test_header_name_mismatch(self) -> None:
    headers = self._VALID_HEADERS[:-1] + ("invalid_column",)
    self._write_file(headers=headers, data=self._VALID_DATA)

    with pytest.raises(exceptions.UnexpectedColumnsError) as exc:
      TSVReader.from_file(path=self._filepath)

    assert exc.value.tsv_reader_name == TSVReader.__name__
    assert exc.value.input_location == self._filepath
    assert exc.value.missing_columns == {self._VALID_HEADERS[-1]}
    assert exc.value.extra_columns == {headers[-1]}

  def _write_file(
    self,
    headers: typing.Optional[typing.Sequence[str]],
    data: typing.Sequence[typing.Sequence[typing.Any]],
  ) -> None:
    with open(file=self._filepath, mode="w") as temp:
      if headers is not None:
        temp.write("\t".join(headers) + os.linesep)
      for line in data:
        temp.write("\t".join(str(item) for item in line) + os.linesep)
