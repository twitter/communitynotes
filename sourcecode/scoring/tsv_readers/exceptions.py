import typing

if typing.TYPE_CHECKING:
  from scoring.tsv_readers import tsv_reader_base


class TSVReaderError(ValueError):
  def __init__(self, tsv_reader: "tsv_reader_base.TSVReaderBase"):
    self._tsv_reader_name = tsv_reader.reader_name
    self._input_location = tsv_reader.location

  @property
  def tsv_reader_name(self) -> str:
    return self._tsv_reader_name

  @property
  def input_location(self) -> str:
    return self._input_location

  def __str__(self) -> str:
    return f"Invalid TSV input for {self.tsv_reader_name} from {self.input_location}"


class IoInputNotSeekableError(TSVReaderError):
  def __str(self) -> str:
    super_str = super(IoInputNotSeekableError, self).__str__()
    return f"{super_str}: IO input must be seekable."


class EmptyInputError(TSVReaderError):
  def __str__(self) -> str:
    super_str = super(EmptyInputError, self).__str__()
    return f"{super_str}: Input is empty."


class UnexpectedColumnCountError(TSVReaderError):
  def __init__(self, num_expected_columns: int, num_actual_columns: int, *args, **kwargs):
    super(UnexpectedColumnCountError, self).__init__(*args, **kwargs)
    self._num_expected_columns = num_expected_columns
    self._num_actual_columns = num_actual_columns

  @property
  def num_expected_columns(self) -> int:
    return self._num_expected_columns

  @property
  def num_actual_columns(self) -> int:
    return self._num_actual_columns

  def __str__(self) -> str:
    super_str = super(UnexpectedColumnCountError, self).__str__()
    return f"{super_str}: Expected {self.num_expected_columns} columns but got {self.num_actual_columns}."


class UnexpectedColumnsError(TSVReaderError):
  def __init__(
    self,
    extra_columns: typing.Set[str],
    missing_columns: typing.Set[str],
    *args,
    **kwargs,
  ):
    super(UnexpectedColumnsError, self).__init__(*args, **kwargs)
    self._extra_columns = extra_columns
    self._missing_columns = missing_columns

  @property
  def extra_columns(self) -> typing.Set:
    return self._extra_columns

  @property
  def missing_columns(self) -> typing.Set:
    return self._missing_columns

  def __str__(self) -> str:
    super_str = super(UnexpectedColumnsError, self).__str__()
    return f"{super_str}: Columns don't match:\n" \
           f"{self.extra_columns} are extra columns,\n" \
           f"{self.missing_columns} are missing."
