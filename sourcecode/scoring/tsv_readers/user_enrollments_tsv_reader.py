import typing

import pandas as pd

from scoring import constants
from sourcecode.scoring.tsv_readers import tsv_reader_base


class UserEnrollmentsNewFormatTSVReader(tsv_reader_base.TSVReaderBase):
  @property
  def _datatype_mapping(self) -> typing.Mapping[str, typing.Type]:
    return constants.userEnrollmentTSVTypeMapping


class UserEnrollmentOldFormatTSVReader(tsv_reader_base.TSVReaderBase):
  def read(self, has_header: bool = True) -> pd.DataFrame:
    df = super(UserEnrollmentOldFormatTSVReader, self).read(has_header=has_header)
    df[constants.modelingPopulationKey] = constants.core
    return df

  @property
  def _datatype_mapping(self) -> typing.Mapping[str, typing.Type]:
    return constants.userEnrollmentTSVTypeMappingOld
