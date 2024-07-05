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
import sys
from threading import Lock
import traceback
from typing import Any, Callable, Dict, List, Set, Tuple

import numpy as np
import pandas as pd


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


def get_check(fail: Any, lines: List[str], unsafeAllowed: Set[str]) -> Callable:
  """Return a function which will either assert a condition or conditionally log."""

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
    if fail and not failDisabled:
      assert condition, msg
    elif not condition:
      if failDisabled:
        lines.append(f"{msg} (allowed)")
      else:
        lines.append(f"{msg} (UNALLOWED)")

  return _check


def _log_errors(method: str, callsite: str, lines: List[str], counter: TypeErrorCounter) -> None:
  if not lines:
    return
  counter.log_errors(method, callsite, lines)
  errorLines = "\n".join([f"  PandasTypeError: {l}" for l in lines])
  msg = f"\n{method} ERROR AT: {callsite}" f"{method} ERRORS:\n{errorLines}\n"
  print(msg, file=sys.stderr)


def safe_concat(fail: bool, counter: TypeErrorCounter) -> Callable:
  """Return a modified concat function that checks type stability.

  Args:
    fail: If True, unexpected type conversions should trigger a failed assert.  If False,
      unexpected conversions should log to stderr.
    counter: Tracker for summarizing problematic calls at the end of execution.
  """
  original = pd.concat

  def _safe_concat(*args, **kwargs):
    """Wrapper around pd.concat

    Args:
      args: non-keyword arguments to pass through to merge.
      kwargs: keyword arguments to pass through to merge.
    """
    lines = []
    if "unsafeAllowed" in kwargs:
      unsafeAllowed = kwargs["unsafeAllowed"]
      if isinstance(unsafeAllowed, str):
        unsafeAllowed = {unsafeAllowed}
      del kwargs["unsafeAllowed"]
    else:
      unsafeAllowed: Set[str] = set()
    check = get_check(fail, lines, unsafeAllowed)
    # Validate that all objects being concatenated are either Series or DataFrames
    objs = args[0]
    assert type(objs) == list, f"expected first argument to be a list: type={type(objs)}"
    assert all(type(obj) == pd.Series for obj in objs) or all(
      type(obj) == pd.DataFrame for obj in objs
    ), f"Expected concat args to be either pd.Series or pd.DataFrame: {[type(obj) for obj in objs]}"
    if type(objs[0]) == pd.Series:
      if "axis" in kwargs and kwargs["axis"] == 1:
        # Since the call is concatenating Series as columns in a DataFrame, validate that the sequence
        # of Series dtypes matches the sequence of column dtypes in the dataframe.
        result = original(*args, **kwargs)
        objDtypes = [obj.dtype for obj in objs]
        assert len(objDtypes) == len(
          result.dtypes
        ), f"dtype length mismatch: {len(objDtypes)} vs {len(result.dtypes)}"
        for col, seriesType, colType in zip(result.columns, objDtypes, result.dtypes):
          check(col, seriesType == colType, f"Series concat on {col}: {seriesType} vs {colType}")
      else:
        # If Series, validate that all series were same type and return
        seriesTypes = set(obj.dtype for obj in objs)
        check(None, len(seriesTypes) == 1, f"More than 1 unique Series type: {seriesTypes}")
        result = original(*args, **kwargs)
    else:
      # If DataFrame, validate that all input columns with matching names have the same type
      # and build expectation for output column types
      assert type(objs[0]) == pd.DataFrame
      colTypes: Dict[str, List[type]] = dict()
      for df in objs:
        for col, dtype in df.reset_index(drop=False).dtypes.items():
          if col not in colTypes:
            colTypes[col] = []
          colTypes[col].append(dtype)
      # Perform concatenation and validate that there weren't any type changes
      result = original(*args, **kwargs)
      for col, outputType in result.reset_index(drop=False).dtypes.items():
        check(
          col,
          all(inputType == outputType for inputType in colTypes[col]),
          f"DataFrame concat on {col}: output={outputType} inputs={colTypes[col]}",
        )
    _log_errors("CONCAT", traceback.format_stack()[-2], lines, counter)
    return result

  return _safe_concat


def safe_merge(fail: bool, counter: TypeErrorCounter) -> Callable:
  """Return a modified merge function that checks type stability.

  Args:
    fail: If True, unexpected type conversions should trigger a failed assert.  If False,
      unexpected conversions should log to stderr.
    counter: Tracker for summarizing problematic calls at the end of execution.
  """
  original = pd.DataFrame.merge

  def _safe_merge(*args, **kwargs):
    """Wrapper around pd.DataFrame.merge.

    Args:
      args: non-keyword arguments to pass through to merge.
      kwargs: keyword arguments to pass through to merge.
    """
    lines = []
    if "unsafeAllowed" in kwargs:
      unsafeAllowed = kwargs["unsafeAllowed"]
      if isinstance(unsafeAllowed, str):
        unsafeAllowed = {unsafeAllowed}
      del kwargs["unsafeAllowed"]
    else:
      unsafeAllowed: Set[str] = set()
    check = get_check(fail, lines, unsafeAllowed)
    leftFrame = args[0]
    rightFrame = args[1]
    # Validate that argument types are as expected
    assert type(leftFrame) is pd.DataFrame
    assert type(rightFrame) is pd.DataFrame
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
    result = original(*args, **kwargs)
    resultDtypes = dict(result.dtypes)
    for col in resultDtypes:
      check(
        col,
        resultDtypes[col] == expectedColTypes[col],
        f"Output mismatch on {col}: result={resultDtypes[col]} expected={expectedColTypes[col]}",
      )
    _log_errors("MERGE", traceback.format_stack()[-2], lines, counter)
    return result

  return _safe_merge


def safe_join(fail: bool, counter: TypeErrorCounter) -> Callable:
  """Return a modified merge function that checks type stability.

  Args:
    fail: If True, unexpected type conversions should trigger a failed assert.  If False,
      unexpected conversions should log to stderr.
    counter: Tracker for summarizing problematic calls at the end of execution.
  """
  original = pd.DataFrame.join

  def _safe_join(*args, **kwargs):
    """Wrapper around pd.DataFrame.merge.

    Args:
      args: non-keyword arguments to pass through to merge.
      kwargs: keyword arguments to pass through to merge.
    """
    lines = []
    if "unsafeAllowed" in kwargs:
      unsafeAllowed = kwargs["unsafeAllowed"]
      if isinstance(unsafeAllowed, str):
        unsafeAllowed = {unsafeAllowed}
      del kwargs["unsafeAllowed"]
    else:
      unsafeAllowed: Set[str] = set()
    check = get_check(fail, lines, unsafeAllowed)
    leftFrame = args[0]
    rightFrame = args[1]
    # Validate arguments are as expected
    assert type(leftFrame) is pd.DataFrame
    assert type(rightFrame) is pd.DataFrame
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
    assert match, f"Join index mismatch:\n{leftFrame.index}\nvs\n{rightFrame.index}"
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
      leftDtypes = {leftFrame.index.name: leftFrame.index.dtype}
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
    result = original(*args, **kwargs)
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
    _log_errors("JOIN", traceback.format_stack()[-2], lines, counter)
    return result

  return _safe_join


def patch_pandas(main: Callable) -> Callable:
  """Return a decorator for wrapping main with pandas patching and logging

  Args:
    main: "main" function for program binary
  """

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
      # birdwatch/scoring/src/main/python/run_prescoring.py
      # birdwatch/scoring/src/main/python/run_final_scoring.py
      # birdwatch/scoring/src/main/python/run_contributor_scoring.py
      # birdwatch/scoring/src/main/python/run.py
      assert len(args) == 1, f"unexpected 1 positional args, but found {len(args)}"
      assert len(kwargs) == 0, f"expected kwargs to be empty, but found {len(kwargs)}"
      clArgs = args[0]
    # Apply patches, configured based on whether types should be enforced or logged
    counter = TypeErrorCounter()
    pd.concat = safe_concat(clArgs.enforce_types, counter)
    # Note that this will work when calling df1.merge(df2) because the first argument
    # to "merge" is df1 (i.e. self).
    pd.DataFrame.merge = safe_merge(clArgs.enforce_types, counter)
    pd.DataFrame.join = safe_join(clArgs.enforce_types, counter)
    # Run main
    retVal = main(*args, **kwargs)
    # Log type error summary
    if hasattr(clArgs, "parallel") and not clArgs.parallel:
      print(f"\nTYPE WARNING SUMMARY\n{counter.get_summary()}", file=sys.stderr)
    else:
      # Don't show type summary because counters will be inaccurate due to scorers running
      # in their own process.
      print("Type summary omitted when running in parallel.", file=sys.stderr)
    # Return result of main
    return retVal

  return _inner
