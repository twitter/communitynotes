import typing

from scoring import constants
from sourcecode.scoring.tsv_readers import tsv_reader_base


class NotesStatusHistoryTSVReader(tsv_reader_base.TSVReaderBase):
  @property
  def _datatype_mapping(self) -> typing.Mapping[str, typing.Type]:
    return constants.noteStatusHistoryTSVTypeMapping
