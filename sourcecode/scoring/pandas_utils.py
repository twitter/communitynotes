"""This module patches Pandas to alert or fail on unexpected dtype conversions.

The module corrently supports the merge, join and concat operations as these functions
can generate derived dataframes with type conversions.  The patch can be configured to
either log to stderr or assert False when an unexpected type conversion is detected.

This module should support type-related work in the scorer, including:
* Setting all input datatypes to the appropriate np (non-nullable) or pd (nullable) datatype
  for the associated input.  For example, noteIds should be np.int64, timestamp of first
  status should be pd.Int64Dtype, etc.
* Enforcing type expectations on outputs.  For example, validating that the participantId
  is an int64 and has not been converted to float.
* Fixing unexpected type conversion errors by specifying default values for rows that are
  lacking columns during a merge, join or concat.  For example, if we generate numRatings
  and then join with noteStatusHistory, we should be able to pass fillna={"numRatings": 0}
  to "merge" so that the resulting column should still have type np.int64 where missing
  values have been filled with 0 (as opposed to cast to a float with missing values set to
  np.NaN).
* Add an "allow_unsafe" keyword argument to merge, join and concat that overrides "fail"
  and instead logs to stderr.  This will allow us to default all current and new code to
  enforced safe behavior except for callsites that haven't been fixed yet.
"""

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from hashlib import sha256
import re
import sys
from threading import Lock
import traceback
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from . import constants as c

import numpy as np
import pandas as pd


def get_df_fingerprint(df, cols):
  """Fingerprint the order of select column values within a dataframe."""
  try:
    strs = [
      sha256(b"".join(map(lambda v: int(v).to_bytes(8, "big"), df[col]))).hexdigest()
      for col in cols
    ]
    return sha256(",".join(strs).encode("utf-8")).hexdigest()
  except ValueError:
    strs = [sha256(",".join(map(str, df[col])).encode("utf-8")).hexdigest() for col in cols]
    return sha256(",".join(strs).encode("utf-8")).hexdigest()


def keep_columns(df: pd.DataFrame, cols: List[str]):
  cols = [col for col in cols if col in df]
  return df[cols]


def get_df_info(
  df: pd.DataFrame,
  name: Optional[str] = None,
  deep: bool = False,
  counter: bool = False,
) -> str:
  """Log dtype and RAM usage stats for each input DataFrame."""
  stats = (
    df.dtypes.to_frame().reset_index(drop=False).rename(columns={"index": "column", 0: "dtype"})
  ).merge(
    # deep=True shows memory usage for the entire contained object (e.g. if the type
    # of a column is "object", then deep=True shows the size of the objects instead
    # of the size of the pointers.
    df.memory_usage(index=True, deep=deep)
    .to_frame()
    .reset_index(drop=False)
    .rename(columns={"index": "column", 0: "RAM"})
  )
  ramBytes = stats["RAM"].sum()
  if name is not None:
    lines = [f"""{name} total RAM: {ramBytes} bytes ({ramBytes * 1e-9:.3f} GB)"""]
  else:
    lines = [f"""total RAM: {ramBytes} bytes ({ramBytes * 1e-9:.3f} GB)"""]
  lines.extend(str(stats).split("\n"))
  if counter:
    for col, dtype in zip(stats["column"], stats["dtype"]):
      if dtype != object:
        continue
      lines.append(f"{col}: {Counter(type(obj) for obj in df[col])}")
  return "\n".join(lines)


class TypeErrorCounter(object):
  def __init__(self):
    self._callCounts: Dict[Tuple[str, str], int] = dict()
    self._typeErrors: Dict[Tuple[str, str], Counter[str]] = dict()
    self._lock = Lock()

  def log_errors(self, method: str, callsite: str, errors: List[str]) -> None:
    key = (method, callsite)
    with self._lock:
      if key not in self._callCounts:
        self._callCounts[key] = 0
      self._callCounts[key] += 1
      if key not in self._typeErrors:
        self._typeErrors[key] = Counter()
      for error in errors:
        self._typeErrors[key][error] += 1

  def get_summary(self):
    lines = []
    keys = [
      (method, -1 * count, callsite) for ((method, callsite), count) in self._callCounts.items()
    ]
    for method, count, callsite in sorted(keys):
      lines.append(f"{method}: {-1 * count} BAD CALLS AT: {callsite.rstrip()}")
      for error, errorCount in self._typeErrors[(method, callsite)].items():
        lines.append(f"  {errorCount:3d}x  {error}")
      lines.append("")
    return "\n".join(lines)


class LogLevel(Enum):
  # Raise an error if the expecatation is violated
  FATAL = 1
  # Log to stderr when the expectation is violated
  ERROR = 2
  # Log to stderr any time the column is observed
  INFO = 3


@dataclass
class TypeExpectation:
  dtype: type
  logLevel: LogLevel


class PandasPatcher(object):
  def __init__(
    self,
    fail: bool,
    typeOverrides: Dict[str, TypeExpectation] = dict(),
    silent: bool = False,
  ):
    """Initialize a PandasPatcher with particular failure and type expectations.

    Args:
      fail: Whether to raise errors or log to stderr when expectations are violated.
      expectations: Type expecatations for select columns.
    """
    self._silent = silent  # Set to True to basically disable
    self._fail = fail
    self._counter = TypeErrorCounter()
    self._origConcat = pd.concat
    self._origJoin = pd.DataFrame.join
    self._origMerge = pd.DataFrame.merge
    self._origApply = pd.DataFrame.apply
    self._origInit = pd.DataFrame.__init__
    self._origGetItem = pd.DataFrame.__getitem__
    self._origSetItem = pd.DataFrame.__setitem__
    self._origLocGetItem = pd.core.indexing._LocationIndexer.__getitem__
    self._origLocSetItem = pd.core.indexing._LocationIndexer.__setitem__
    self._expectations = {
      c.noteIdKey: TypeExpectation(np.int64, LogLevel.ERROR),
    }
    for column, expectation in typeOverrides.items():
      self._expectations[column] = expectation

  def get_summary(self) -> str:
    return f"\nTYPE WARNING SUMMARY\n{self._counter.get_summary()}"

  def _log_errors(self, method: str, callsite: str, lines: List[str]) -> None:
    if not lines:
      return
    self._counter.log_errors(method, callsite, lines)
    errorLines = "\n".join([f"  PandasTypeError: {l}" for l in lines])
    msg = f"\n{method} WARNING(S) AT: {callsite}\n{errorLines}\n"
    if not self._silent:
      print(msg, file=sys.stderr)

  def _get_check(self, lines: List[str], kwargs: Dict) -> Callable:
    """Return a function which will either assert a condition or append to a list of errors.

    Note that this function does not actually log to stderr, but rather appends to a list so
    that all
    """
    unsafeAllowed = set()
    if "unsafeAllowed" in kwargs:
      unsafeAllowedArg = kwargs["unsafeAllowed"]
      if isinstance(unsafeAllowedArg, str):
        unsafeAllowed = {unsafeAllowedArg}
      elif isinstance(unsafeAllowedArg, List):
        unsafeAllowed = set(unsafeAllowedArg)
      else:
        assert isinstance(unsafeAllowedArg, Set)
        unsafeAllowed = unsafeAllowedArg
      del kwargs["unsafeAllowed"]

    def _check(columns: Any, condition: bool, msg: str):
      if isinstance(columns, str):
        failDisabled = columns in unsafeAllowed
      elif isinstance(columns, List):
        failDisabled = all(col in unsafeAllowed for col in columns)
      else:
        # Note there are multiple circumstances where the type of Columns may not be a str
        # or List[str], including when we are concatenating a Series (column name will be
        # set to None), when there are mulit-level column names (column name will be a tuple)
        # or when Pandas has set column names to a RangeIndex.
        failDisabled = False
      if self._fail and not failDisabled:
        assert condition, msg
      elif not condition:
        if failDisabled:
          lines.append(f"{msg} (allowed)")
        else:
          lines.append(f"{msg} (UNALLOWED)")

    return _check

  def _get_callsite(self) -> str:
    """Return the file, function, line numer and pandas API call on a single line."""
    try:
      # Find the first relevant frame (not in pandas_utils.py or pandas)
      relevant_frame = None
      for line in traceback.format_stack()[::-1]:
        path = line.split(",")[0]
        if "/pandas_utils.py" in path:
          continue
        if "/pandas/" in path:
          continue
        relevant_frame = line
        break

      if not relevant_frame:
        return "callsite unknown (no relevant frame found)"

      # Handle paths resulting from bazel invocation
      match = re.match(
        r'^  File ".*?/site-packages(/.*?)", (.*?), (.*?)\n    (.*)\n$', relevant_frame
      )
      if match:
        return f"{match.group(1)}, {match.group(3)}, at {match.group(2)}: {match.group(4)}"
      # Handle paths fresulting from pytest invocation
      match = re.match(
        r'^  File ".*?/src/(test|main)/python(/.*?)", (.*?), (.*?)\n    (.*)\n$',
        relevant_frame,
      )
      if match:
        return f"{match.group(2)}, {match.group(4)}, at {match.group(3)}: {match.group(5)}"
      # Handle other paths (e.g. notebook, public code)
      match = re.match(r'^  File "(.*?)", (.*?), (.*?)\n    (.*)\n$', relevant_frame)
      if match:
        return f"{match.group(1)}, {match.group(3)}, at {match.group(2)}: {match.group(4)}"
      # Handle multiprocessing and other non-standard stack frames
      match = re.match(r'^  File "([^"]+)", line (\d+), in (.+)', relevant_frame)
      if match:
        file_path = match.group(1)
        line_num = match.group(2)
        func_name = match.group(3).strip()
        # Extract code if available
        code = ""
        if "\n" in relevant_frame:
          code_part = (
            relevant_frame.split("\n")[1].strip() if len(relevant_frame.split("\n")) > 1 else ""
          )
          if code_part:
            code = code_part
        return f"{file_path}, line {line_num}, in {func_name}: {code}"

      # Last resort: just return the frame without the full stack
      return f"callsite unknown: {relevant_frame.strip()}"
    except Exception as e:
      # If anything goes wrong, return a simple message without printing the stack
      return f"callsite unknown (error: {str(e)})"

  def _check_dtype(self, dtype: Any, expected: type) -> bool:
    """Return True IFF dtype corresponds to expected.

    Note that for non-nullable columns, dtype may equal type (e.g. np.int64), but for nullable
    columns the column type is actually an instance of a pandas dtype (e.g. pd.Int64Dtype)
    """
    assert expected != object, "expectation must be more specific than object"
    return dtype == expected or isinstance(dtype, expected)

  def _check_name_and_type(self, name: str, dtype: Any) -> List[str]:
    """Returns a list of type mismatches if any are found, or raises an error."""
    if name not in self._expectations:
      return []
    typeExpectation = self._expectations[name]
    msg = f"Type expectation mismatch on {name}: found={dtype} expected={typeExpectation.dtype.__name__}"
    match = self._check_dtype(dtype, typeExpectation.dtype)
    if typeExpectation.logLevel == LogLevel.INFO:
      return (
        [msg]
        if not match
        else [
          f"Type expectation match on {name}: found={dtype} expected={typeExpectation.dtype.__name__}"
        ]
      )
    elif typeExpectation.logLevel == LogLevel.ERROR or not self._fail:
      return [msg] if not match else []
    else:
      assert typeExpectation.logLevel == LogLevel.FATAL
      assert self._fail
      assert match, msg
      return []

  def _validate_series(self, series: pd.Series) -> List[str]:
    assert isinstance(series, pd.Series), f"unexpected type: {type(series)}"
    return self._check_name_and_type(series.name, series.dtype)

  def _validate_dataframe(self, df: pd.DataFrame) -> List[str]:
    """Returns a list of type mismatches if any are found, or raises an error."""
    assert isinstance(df, pd.DataFrame), f"unexpected type: {type(df)}"
    lines = []
    # Check index types
    if type(df.index) == pd.MultiIndex:
      for name, dtype in df.index.dtypes.to_dict().items():
        lines.extend(self._check_name_and_type(name, dtype))
    elif type(df.index) == pd.RangeIndex or df.index.name is None:
      # Index is uninteresting - none was specified by the caller.
      pass
    else:
      lines.extend(self._check_name_and_type(df.index.name, df.index.dtype))
    # Check column types
    for name, dtype in df.dtypes.to_dict().items():
      lines.extend(self._check_name_and_type(name, dtype))
    return lines

  def safe_init(self) -> Callable:
    """Return a modified __init__ function that checks type expectations."""

    def _safe_init(*args, **kwargs):
      """Wrapper around pd.concat

      Args:
        args: non-keyword arguments to pass through to merge.
        kwargs: keyword arguments to pass through to merge.
      """
      df = args[0]
      assert isinstance(df, pd.DataFrame), f"unexpected type: {type(df)}"
      retVal = self._origInit(*args, **kwargs)
      assert retVal is None
      lines = self._validate_dataframe(df)
      self._log_errors("INIT", self._get_callsite(), lines)
      return retVal

    return _safe_init

  def safe_concat(self) -> Callable:
    """Return a modified concat function that checks type stability."""

    def _safe_concat(*args, **kwargs):
      """Wrapper around pd.concat

      Args:
        args: non-keyword arguments to pass through to merge.
        kwargs: keyword arguments to pass through to merge.
      """
      lines = []
      check = self._get_check(lines, kwargs)
      # Validate that all objects being concatenated are either Series or DataFrames
      objs = args[0]
      assert type(objs) == list, f"expected first argument to be a list: type={type(objs)}"
      assert (
        all(type(obj) == pd.Series for obj in objs)
        or all(type(obj) == pd.DataFrame for obj in objs)
      ), f"Expected concat args to be either pd.Series or pd.DataFrame: {[type(obj) for obj in objs]}"
      if type(objs[0]) == pd.Series:
        if "axis" in kwargs and kwargs["axis"] == 1:
          # Since the call is concatenating Series as columns in a DataFrame, validate that the sequence
          # of Series dtypes matches the sequence of column dtypes in the dataframe.
          result = self._origConcat(*args, **kwargs)
          objDtypes = [obj.dtype for obj in objs]
          assert len(objDtypes) == len(
            result.dtypes
          ), f"dtype length mismatch: {len(objDtypes)} vs {len(result.dtypes)}"
          for col, seriesType, colType in zip(result.columns, objDtypes, result.dtypes):
            check(
              col,
              seriesType == colType,
              f"Series concat on {col}: {seriesType} vs {colType}",
            )
        else:
          # If Series, validate that all series were same type and return
          seriesTypes = set(obj.dtype for obj in objs)
          check(
            None,
            len(seriesTypes) == 1,
            f"More than 1 unique Series type: {seriesTypes}",
          )
          result = self._origConcat(*args, **kwargs)
      else:
        # If DataFrame, validate that all input columns with matching names have the same type
        # and build expectation for output column types
        assert type(objs[0]) == pd.DataFrame
        # Validate all inputs
        for dfArg in objs:
          lines.extend(self._validate_dataframe(dfArg))
        colTypes: Dict[str, List[type]] = dict()
        for df in objs:
          for col, dtype in df.reset_index(drop=False).dtypes.items():
            if col not in colTypes:
              colTypes[col] = []
            colTypes[col].append(dtype)
        # Perform concatenation and validate that there weren't any type changes
        result = self._origConcat(*args, **kwargs)
        for col, outputType in result.reset_index(drop=False).dtypes.items():
          check(
            col,
            all(inputType == outputType for inputType in colTypes[col]),
            f"DataFrame concat on {col}: output={outputType} inputs={colTypes[col]}",
          )
      if isinstance(result, pd.DataFrame):
        lines.extend(self._validate_dataframe(result))
      elif isinstance(result, pd.Series):
        lines.extend(self._validate_series(result))
      self._log_errors("CONCAT", self._get_callsite(), lines)
      return result

    return _safe_concat

  def safe_apply(self) -> Callable:
    """Return a modified apply function that checks type stability."""

    def _safe_apply(*args, **kwargs):
      """Wrapper around pd.DataFrame.apply

      Args:
        args: non-keyword arguments to pass through to merge.
        kwargs: keyword arguments to pass through to merge.
      """
      # TODO: Flesh this out with additional expectatoins around input and output types
      result = self._origApply(*args, **kwargs)
      if isinstance(result, pd.DataFrame):
        self._log_errors("APPLY", self._get_callsite(), self._validate_dataframe(result))
      elif isinstance(result, pd.Series):
        self._log_errors("APPLY", self._get_callsite(), self._validate_series(result))
      return result

    return _safe_apply

  def safe_merge(self) -> Callable:
    """Return a modified merge function that checks type stability."""

    def _safe_merge(*args, **kwargs):
      """Wrapper around pd.DataFrame.merge.

      Args:
        args: non-keyword arguments to pass through to merge.
        kwargs: keyword arguments to pass through to merge.
      """
      lines = []
      check = self._get_check(lines, kwargs)
      leftFrame = args[0]
      rightFrame = args[1]
      # Validate that argument types are as expected
      assert type(leftFrame) is pd.DataFrame
      assert type(rightFrame) is pd.DataFrame
      lines.extend(self._validate_dataframe(leftFrame))
      lines.extend(self._validate_dataframe(rightFrame))
      # Store dtypes and validate that any common columns have the same type
      leftDtypes = dict(leftFrame.reset_index(drop=False).dtypes)
      rightDtypes = dict(rightFrame.reset_index(drop=False).dtypes)
      for col in set(leftDtypes) & set(rightDtypes):
        check(
          col,
          leftDtypes[col] == rightDtypes[col],
          f"Input mismatch on {col}: left={leftDtypes[col]} vs right={rightDtypes[col]}",
        )
      # Identify the columns we are merging on, if left_on and right_on are unset
      if "on" in kwargs and type(kwargs["on"]) == str:
        onCols = set([kwargs["on"]])
      elif "on" in kwargs and type(kwargs["on"]) == list:
        onCols = set(kwargs["on"])
      elif "left_on" in kwargs:
        assert "on" not in kwargs, "not expecting both on and left_on"
        assert "right_on" in kwargs, "expecting both left_on and right_on to be set"
        onCols = set()
      else:
        assert "on" not in kwargs, f"""unexpected type for on: {type(kwargs["on"])}"""
        onCols = set(leftFrame.columns) & set(rightFrame.columns)
      # Validate that merge columns have matching types
      if "left_on" in kwargs:
        assert "right_on" in kwargs
        left_on = kwargs["left_on"]
        right_on = kwargs["right_on"]
        check(
          [left_on, right_on],
          leftDtypes[left_on] == rightDtypes[right_on],
          f"Merge key mismatch on type({left_on})={leftDtypes[left_on]} vs type({right_on})={rightDtypes[right_on]}",
        )
      else:
        assert len(onCols), "expected onCols to be defined since left_on was not"
        assert "right_on" not in kwargs, "did not expect onCols and right_on"
        for col in onCols:
          check(
            col,
            leftDtypes[col] == rightDtypes[col],
            f"Merge key mismatch on {col}: left={leftDtypes[col]} vs right={rightDtypes[col]}",
          )
      # Compute expected column types
      leftSuffix, rightSuffix = kwargs.get("suffixes", ("_x", "_y"))
      commonCols = set(leftFrame.columns) & set(rightFrame.columns)
      expectedColTypes = dict()
      for col in set(leftFrame.columns) | set(rightFrame.columns):
        if col in onCols:
          # Note that we check above whether leftDtypes[col] == rightDtypes[col] and either raise an
          # error or log as appropriate if there is a mismatch.
          if leftDtypes[col] == rightDtypes[col]:
            expectedColTypes[col] = leftDtypes[col]
          else:
            # Set expectation to None since we don't know what will happen, but do want to log an
            # error later
            expectedColTypes[col] = None
        elif col in commonCols:
          expectedColTypes[f"{col}{leftSuffix}"] = leftDtypes[col]
          expectedColTypes[f"{col}{rightSuffix}"] = rightDtypes[col]
        elif col in leftDtypes:
          assert col not in rightDtypes
          expectedColTypes[col] = leftDtypes[col]
        else:
          expectedColTypes[col] = rightDtypes[col]
      # Perform merge and validate results
      result = self._origMerge(*args, **kwargs)
      resultDtypes = dict(result.dtypes)
      for col in resultDtypes:
        check(
          col,
          resultDtypes[col] == expectedColTypes[col],
          f"Output mismatch on {col}: result={resultDtypes[col]} expected={expectedColTypes[col]}",
        )
      lines.extend(self._validate_dataframe(result))
      self._log_errors("MERGE", self._get_callsite(), lines)
      return result

    return _safe_merge

  def safe_join(self) -> Callable:
    """Return a modified merge function that checks type stability."""

    def _safe_join(*args, **kwargs):
      """Wrapper around pd.DataFrame.merge.

      Args:
        args: non-keyword arguments to pass through to merge.
        kwargs: keyword arguments to pass through to merge.
      """
      lines = []
      check = self._get_check(lines, kwargs)
      leftFrame = args[0]
      rightFrame = args[1]
      # Validate arguments are as expected
      assert type(leftFrame) is pd.DataFrame
      assert type(rightFrame) is pd.DataFrame
      lines.extend(self._validate_dataframe(leftFrame))
      lines.extend(self._validate_dataframe(rightFrame))
      assert len(set(kwargs) - {"lsuffix", "rsuffix", "how"}) == 0, f"unexpected kwargs: {kwargs}"
      # Validate the assumption that columns used as the join key in the index have the same type.
      # This is analogous to validating that onCols match and have the same types in _safe_merge.
      if len(leftFrame.index.names) == 1 and len(rightFrame.index.names) == 1:
        match = leftFrame.index.dtype == rightFrame.index.dtype
      elif len(leftFrame.index.names) == 1 and len(rightFrame.index.names) > 1:
        indexTypes = dict(rightFrame.index.dtypes)
        name = leftFrame.index.names[0]
        assert name in indexTypes, f"{name} not found in {indexTypes}"
        match = indexTypes[name] == leftFrame.index.dtype
      elif len(leftFrame.index.names) > 1 and len(rightFrame.index.names) == 1:
        indexTypes = dict(leftFrame.index.dtypes)
        name = rightFrame.index.names[0]
        assert name in indexTypes, f"{name} not found in {indexTypes}"
        match = indexTypes[name] == rightFrame.index.dtype
      else:
        assert (
          len(leftFrame.index.names) > 1
        ), f"unexpected left: {type(leftFrame.index)}, {leftFrame.index}"
        assert (
          len(rightFrame.index.names) > 1
        ), f"unexpected right: {type(rightFrame.index)}, {rightFrame.index}"
        leftIndexTypes = dict(leftFrame.index.dtypes)
        rightIndexTypes = dict(rightFrame.index.dtypes)
        match = True
        for col in set(leftIndexTypes) & set(rightIndexTypes):
          match = match & (leftIndexTypes[col] == rightIndexTypes[col])
      check(
        list(set(leftFrame.index.names) | set(rightFrame.index.names)),
        match,
        "Join index mismatch:\nleft:\n{left}\nvs\nright:\n{right}".format(
          left=leftFrame.index.dtype if len(leftFrame.index.names) == 1 else leftFrame.index.dtypes,
          right=rightFrame.index.dtype
          if len(rightFrame.index.names) == 1
          else rightFrame.index.dtypes,
        ),
      )
      # Validate that input columns with the same name have the same types
      leftDtypes = dict(leftFrame.dtypes)
      rightDtypes = dict(rightFrame.dtypes)
      for col in set(leftDtypes) & set(rightDtypes):
        check(
          col,
          leftDtypes[col] == rightDtypes[col],
          f"Input mismatch on {col}: left={leftDtypes[col]} vs right={rightDtypes[col]}",
        )
      # Validate that none of the columns in an index have the same name as a non-index column
      # in the opposite dataframe
      assert (
        len(set(leftFrame.index.names) & set(rightFrame.columns)) == 0
      ), f"left index: {set(leftFrame.index.names)}; right columns {set(rightFrame.columns)}"
      assert (
        len(set(rightFrame.index.names) & set(leftFrame.columns)) == 0
      ), f"right index: {set(rightFrame.index.names)}; left columns {set(leftFrame.columns)}"
      # Compute expected types for output columns
      commonCols = set(leftFrame.columns) & set(rightFrame.columns)
      expectedColTypes = dict()
      leftSuffix = kwargs.get("lsuffix", "")
      rightSuffix = kwargs.get("rsuffix", "")
      for col in set(leftFrame.columns) | set(rightFrame.columns):
        if col in commonCols:
          expectedColTypes[f"{col}{leftSuffix}"] = leftDtypes[col]
          expectedColTypes[f"{col}{rightSuffix}"] = rightDtypes[col]
        elif col in leftDtypes:
          assert col not in rightDtypes
          expectedColTypes[col] = leftDtypes[col]
        else:
          expectedColTypes[col] = rightDtypes[col]
      # Compute expected types for index columns
      leftIndexCols = set(leftFrame.index.names)
      rightIndexCols = set(rightFrame.index.names)
      if len(leftIndexCols) > 1:
        leftDtypes = dict(leftFrame.index.dtypes)
      else:
        leftDtypes = {leftFrame.index.name: rightFrame.index.dtype}
      if len(rightIndexCols) > 1:
        rightDtypes = dict(rightFrame.index.dtypes)
      else:
        rightDtypes = {rightFrame.index.name: rightFrame.index.dtype}
      for col in leftIndexCols & rightIndexCols:
        # For columns in both indices, type should not change if input types agree.  If input types
        # disagree, then we have no expectation.
        if leftDtypes[col] == rightDtypes[col]:
          expectedColTypes[col] = leftDtypes[col]
        else:
          expectedColTypes[col] = None
      for col in (leftIndexCols | rightIndexCols) - (leftIndexCols & rightIndexCols):
        # For columns in exactly one index, the expected output type should match the input column type
        # and the column name should not change because we have validated that the column does not
        # appear in the other dataframe
        if col in leftDtypes:
          assert col not in rightDtypes, f"unexpected column: {col}"
          expectedColTypes[col] = leftDtypes[col]
        else:
          expectedColTypes[col] = rightDtypes[col]
      # Perform join and validate results.  Note that we already validated that the indices had the
      # same columns and types, and that the "on" argument is unset, so now we only need to check
      # the non-index columns.
      result = self._origJoin(*args, **kwargs)
      # Note that we must reset index to force any NaNs in the index to emerge as float types.
      # See example below.
      # left = pd.DataFrame({"idx0": [1, 2], "idx1": [11, 12], "val1": [4, 5]}).set_index(["idx0", "idx1"])
      # right = pd.DataFrame({"idx0": [1, 2, 3], "idx2": [21, 22, 23], "val2": [7, 8, 9]}).set_index(["idx0", "idx2"])
      # print(dict(left.join(right, how="outer").index.dtypes))
      # print(dict(left.join(right, how="outer").reset_index(drop=False).dtypes))
      # $> {'idx0': dtype('int64'), 'idx1': dtype('int64'), 'idx2': dtype('int64')}
      # $> {'idx0': dtype('int64'), 'idx1': dtype('float64'), 'idx2': dtype('int64'), 'val1': dtype('float64'), 'val2': dtype('int64')}
      resultDtypes = dict(result.reset_index(drop=False).dtypes)
      # Add default type for index
      if "index" not in expectedColTypes:
        expectedColTypes["index"] = np.int64
      for col, dtype in resultDtypes.items():
        if len(col) == 2 and col[1] == "":
          col = col[0]
        check(
          col,
          dtype == expectedColTypes[col],
          f"Output mismatch on {col}: result={dtype} expected={expectedColTypes[col]}",
        )
      lines.extend(self._validate_dataframe(result))
      self._log_errors("JOIN", self._get_callsite(), lines)
      return result

    return _safe_join


# TODO: restore original functionality before return
# TODO: make enforce_types an explicit arguemnt so this is less error prone
def patch_pandas(main: Callable) -> Callable:
  """Return a decorator for wrapping main with pandas patching and logging

  Args:
    main: "main" function for program binary
  """

  @wraps(main)
  def _inner(*args, **kwargs) -> Any:
    """Determine patching behavior, apply patch and add logging."""
    print("Patching pandas")
    if "args" in kwargs:
      # Handle birdwatch/scoring/src/main/python/public/scoring/runner.py, which expects
      # args as a keyword argument and not as a positional argument.
      assert len(args) == 0, f"positional arguments not expected, but found {len(args)}"
      clArgs = kwargs["args"]
    else:
      # Handle the following, which expect args as the second positional argument:
      # birdwatch/scoring/src/main/python/run_post_selection_similarity.py
      # birdwatch/scoring/src/main/python/run_prescoring.py
      # birdwatch/scoring/src/main/python/run_final_scoring.py
      # birdwatch/scoring/src/main/python/run_contributor_scoring.py
      # birdwatch/scoring/src/main/python/run.py
      # birdwatch/scoring/src/main/python/public/scoring/run_scoring.py
      assert len(args) == 1, f"unexpected 1 positional args, but found {len(args)}"
      assert len(kwargs) == 0, f"expected kwargs to be empty, but found {len(kwargs)}"
      clArgs = args[0]

    # Apply patches, configured based on whether types should be enforced or logged
    patcher = PandasPatcher(clArgs.enforce_types)
    pd.concat = patcher.safe_concat()
    # Note that this will work when calling df1.merge(df2) because the first argument
    # to "merge" is df1 (i.e. self).
    pd.DataFrame.merge = patcher.safe_merge()
    pd.DataFrame.join = patcher.safe_join()
    pd.DataFrame.apply = patcher.safe_apply()
    pd.DataFrame.__init__ = patcher.safe_init()
    # Run main
    retVal = main(*args, **kwargs)
    # Log type error summary
    if hasattr(clArgs, "parallel") and not clArgs.parallel:
      print(patcher.get_summary(), file=sys.stderr)
    else:
      # Don't show type summary because counters will be inaccurate due to scorers running
      # in their own process.
      print("Type summary omitted when running in parallel.", file=sys.stderr)
    # Return result of main
    return retVal

  return _inner
