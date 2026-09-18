"""Predict which notes will achieve CRH status from early ratings.

Trains a GradientBoosting model on pooled data across multiple N checkpoints,
using features from both all-rater and scored-rater subsets plus author reputation
computed from noteStatusHistory. At inference, scores each note at its current
rating count and applies per-N confidence thresholds with revoke logic.

Training:
  For each N in cascade_ns, truncate to first N ratings, build features, and
  add to the pooled training set with N as a feature. Only notes created after
  pcrhTrainingCutoffMillis are used for training.

Inference:
  Score notes whose current rating count is between min(cascade_ns) and 100.
  The note's N feature is set to its actual rating count (not snapped to a
  cascade bucket), so the model is queried at potentially unseen N values
  (e.g. N=12 even though training only saw N in {8, 10, 15, 20, 30, 50}).
  This matches the fine-grained N evaluation that performed best in offline
  experiments. The per-N exit threshold lookup falls back to the closest
  trained N when the actual N is not a cascade key. Features are computed
  from ALL current ratings (no truncation). Notes previously predicted above
  threshold can be revoked if the current score drops below the revoke
  threshold.
"""

import gc
import logging
from typing import Dict, List, Optional, Set, Tuple

from . import constants as c, scoring_rules

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger("birdwatch.pcrh_model")
logger.setLevel(logging.INFO)

_ALL_PREFIX = "pcrh_all_"
_SCORED_PREFIX = "pcrh_scored_"
_TOPIC_ALL_PREFIX = "pcrh_topic_all_"
_TOPIC_SCORED_PREFIX = "pcrh_topic_scored_"
_AUTHOR_PREFIX = "pcrh_author_"
_UNASSIGNED_TOPIC = "Unassigned"

# A note may only hold abovePcrhThreshold status within this window after its first
# positive timestamp (T0). Outside the window the field is stored as -T0 (not -1) so
# the first-time is preserved and re-entry is blocked. -1 (_PCRH_CLEARED_TIME) means
# lock/legacy clear with no recoverable first-time (a new positive is allowed).
_PCRH_STALE_SUPPRESSION_MILLIS = 12 * 60 * 60 * 1000
_PCRH_CLEARED_TIME = -1.0


def _pcrh_first_times_array(times: np.ndarray) -> np.ndarray:
  """Absolute first-above times; NaN where missing, NaN input, or cleared (-1)."""
  t = np.asarray(times, dtype=np.double)
  first = np.abs(t)
  first[np.isnan(t) | (t == _PCRH_CLEARED_TIME)] = np.nan
  return first


def _pcrh_mapped_first_times(
  noteIds: pd.Series, previousPcrhTimes: Optional[Dict[int, float]]
) -> np.ndarray:
  """First-times from previousPcrhTimes aligned to noteIds (NaN if unknown)."""
  if not previousPcrhTimes:
    return np.full(len(noteIds), np.nan, dtype=np.double)
  return _pcrh_first_times_array(noteIds.map(previousPcrhTimes).to_numpy(dtype=np.double))


def _pcrh_time_rows(df: Optional[pd.DataFrame]) -> pd.DataFrame:
  """noteId + pcrhAboveThresholdTime rows with non-null times, or empty frame."""
  if df is None or c.pcrhAboveThresholdTimeKey not in df.columns or c.noteIdKey not in df.columns:
    return pd.DataFrame(
      {
        c.noteIdKey: pd.Series(dtype=np.int64),
        c.pcrhAboveThresholdTimeKey: pd.Series(dtype=np.double),
      }
    )
  return (
    df[[c.noteIdKey, c.pcrhAboveThresholdTimeKey]]
    .dropna(subset=[c.pcrhAboveThresholdTimeKey])
    .astype({c.pcrhAboveThresholdTimeKey: np.double}, copy=False)
  )


def collect_previous_pcrh_state(
  previousScoredNotes: Optional[pd.DataFrame],
  noteStatusHistory: Optional[pd.DataFrame],
) -> Tuple[Set[int], Dict[int, float]]:
  """Merge PCRH timestamp history from previous scored notes and note status history.

  Per note, prefers: non-sentinel over -1, positive over non-positive, then earlier
  |t| (first show window). Returns currently-positive note ids and a noteId->time map.
  """
  combined = pd.concat(
    [_pcrh_time_rows(previousScoredNotes), _pcrh_time_rows(noteStatusHistory)],
    ignore_index=True,
  )
  if len(combined) == 0:
    return set(), {}

  t = combined[c.pcrhAboveThresholdTimeKey]
  # Sort so the preferred value is first per note, then keep one row per noteId.
  combined = combined.assign(
    _is_sentinel=t == _PCRH_CLEARED_TIME,
    _is_non_positive=t <= 0,
    _abs_time=t.abs(),
  ).sort_values(
    [c.noteIdKey, "_is_sentinel", "_is_non_positive", "_abs_time"],
    kind="mergesort",
  )
  best = combined.drop_duplicates(c.noteIdKey, keep="first")
  previousPcrhTimes: Dict[int, float] = dict(
    zip(best[c.noteIdKey].tolist(), best[c.pcrhAboveThresholdTimeKey].tolist())
  )
  previousPcrhNoteIds: Set[int] = set(
    best.loc[best[c.pcrhAboveThresholdTimeKey] > 0, c.noteIdKey].tolist()
  )
  return previousPcrhNoteIds, previousPcrhTimes


_RATINGS_BASE_COLS = [
  c.noteIdKey,
  c.raterParticipantIdKey,
  c.createdAtMillisKey,
  c.helpfulnessLevelKey,
]


def _ratings_keep_cols(ratings: pd.DataFrame) -> Tuple[List[str], List[str]]:
  """Return (keepCols, tagCols) for the given ratings DataFrame."""
  tagCols = [t for t in c.helpfulTagsTSVOrder + c.notHelpfulTagsTSVOrder if t in ratings.columns]
  return _RATINGS_BASE_COLS + tagCols, tagCols


class PCRHModel:
  def __init__(
    self,
    perNThresholds: Optional[Dict[int, float]] = None,
    revokeThreshold: float = c.pcrhRevokeThreshold,
    trainingCutoffMillis: int = c.pcrhTrainingCutoffMillis,
    nEstimators: int = 200,
    maxDepth: int = 5,
    learningRate: float = 0.07,
    seed: Optional[int] = 42,
    modelName: str = "MFCoreScorer",
    excludeAlreadyDecided: bool = True,
  ):
    self._perNThresholds = perNThresholds or dict(c.pcrhPerNThresholds)
    self._cascadeNs = sorted(self._perNThresholds.keys())
    self._revokeThreshold = revokeThreshold
    self._trainingCutoffMillis = trainingCutoffMillis
    self._modelName = modelName
    self._excludeAlreadyDecided = excludeAlreadyDecided
    self._nEstimators = nEstimators
    self._maxDepth = maxDepth
    self._learningRate = learningRate
    self._seed = seed
    self._model: Optional[HistGradientBoostingClassifier] = None
    self._scaler: Optional[StandardScaler] = None
    self._featureCols: Optional[List[str]] = None

  # ── Shared helpers ──

  _MODEL_GROUPS: Dict[str, Set[int]] = {
    "MFCoreScorer": c.coreGroups,
    "MFExpansionScorer": c.coreGroups | c.expansionGroups,
  }

  def _filter_ratings_by_group(
    self,
    ratings: pd.DataFrame,
    userEnrollment: Optional[pd.DataFrame],
  ) -> pd.DataFrame:
    """Restrict ratings to raters whose modelingGroup is in the allowed set for modelName."""
    allowedGroups = self._MODEL_GROUPS.get(self._modelName)
    if allowedGroups is None or userEnrollment is None:
      return ratings
    eligibleIds = set(
      userEnrollment.loc[
        userEnrollment[c.modelingGroupKey].isin(allowedGroups), c.participantIdKey
      ].astype(str)
    )
    before = len(ratings)
    mask = ratings[c.raterParticipantIdKey].astype(str).isin(eligibleIds)
    ratings = ratings[mask]
    logger.info(f"PCRH group filter ({self._modelName}): {before:,} -> {len(ratings):,} ratings")
    return ratings

  @staticmethod
  def _extract_factor_lookup(
    prescoringRaterModelOutput: pd.DataFrame,
    scorer: str,
    factorColOut: str = "raterFactor1",
    useFirstRoundFallback: bool = False,
  ) -> pd.DataFrame:
    """Extract a raterParticipantId -> factor lookup for a single scorer."""
    rmo = prescoringRaterModelOutput[prescoringRaterModelOutput[c.scorerNameKey] == scorer]
    factor = rmo[c.internalRaterFactor1Key]
    if useFirstRoundFallback and c.internalFirstRoundRaterFactor1Key in rmo.columns:
      factor = factor.combine_first(rmo[c.internalFirstRoundRaterFactor1Key])
    out = pd.DataFrame(
      {c.raterParticipantIdKey: rmo[c.raterParticipantIdKey].values, factorColOut: factor.values}
    )
    out = out[out[factorColOut].notna()].drop_duplicates(
      subset=[c.raterParticipantIdKey], keep="first"
    )
    out[c.raterParticipantIdKey] = out[c.raterParticipantIdKey].astype(str)
    return out

  def _get_rater_factor_lookup(
    self, prescoringRaterModelOutput: pd.DataFrame
  ) -> Tuple[pd.DataFrame, Set]:
    primary = self._extract_factor_lookup(
      prescoringRaterModelOutput, "MFCoreScorer", useFirstRoundFallback=True
    )
    if self._modelName == "MFExpansionScorer":
      fallback = self._extract_factor_lookup(
        prescoringRaterModelOutput, "MFExpansionScorer", useFirstRoundFallback=True
      )
      missing = fallback[~fallback[c.raterParticipantIdKey].isin(primary[c.raterParticipantIdKey])]
      primary = pd.concat([primary, missing], ignore_index=True)

    scoredRaterIds = set(
      prescoringRaterModelOutput[
        (prescoringRaterModelOutput[c.scorerNameKey] == self._modelName)
        & prescoringRaterModelOutput[c.internalRaterFactor1Key].notna()
      ][c.raterParticipantIdKey].astype(str)
    )
    return primary, scoredRaterIds

  _QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

  def _compute_distributional_features_vectorized(
    self,
    subset: pd.DataFrame,
    prefix: str,
    label: str,
  ) -> pd.DataFrame:
    """Compute quantile + skewness features using vectorized groupby operations."""
    if len(subset) == 0:
      return pd.DataFrame()

    g = subset.groupby(c.noteIdKey)["raterFactor1"]

    parts = []

    for q in self._QUANTILES:
      qdf = g.quantile(q).rename(f"{prefix}{label}_p{int(q*100)}")
      parts.append(qdf)

    p75 = g.quantile(0.75)
    p25 = g.quantile(0.25)
    iqr = (p75 - p25).rename(f"{prefix}{label}_iqr")
    parts.append(iqr)

    skew = g.apply(lambda x: x.skew() if len(x) >= 3 else 0.0)
    skew.name = f"{prefix}{label}_skewness"
    parts.append(skew)

    return pd.concat(parts, axis=1)

  def _compute_factor_stats(
    self,
    r: pd.DataFrame,
    prefix: str,
    includeDistributional: bool = False,
  ) -> pd.DataFrame:
    """Aggregate raterFactor1 stats (mean/std/count) per helpfulness level per note.

    r must contain noteIdKey, helpfulnessLevelKey, and raterFactor1 columns.
    """
    hasFactor = r["raterFactor1"].notna()
    allDfs = []
    for level, label in [("HELPFUL", "h"), ("NOT_HELPFUL", "nh"), ("SOMEWHAT_HELPFUL", "sh")]:
      subset = r.loc[hasFactor & (r[c.helpfulnessLevelKey] == level), [c.noteIdKey, "raterFactor1"]]
      agg = subset.groupby(c.noteIdKey)["raterFactor1"].agg(["mean", "std", "count"])
      agg.columns = [
        f"{prefix}{label}_factor1_mean",
        f"{prefix}{label}_factor1_std",
        f"{prefix}{label}_factor1_count",
      ]
      if includeDistributional:
        dist = self._compute_distributional_features_vectorized(subset, prefix, label)
        agg = agg.join(dist, how="outer") if len(dist) > 0 else agg
      allDfs.append(agg)
    return pd.concat(allDfs, axis=1)

  def _compute_rating_features(
    self,
    ratingsDf: pd.DataFrame,
    raterFactorLookup: pd.DataFrame,
    prefix: str,
  ) -> pd.DataFrame:
    if len(ratingsDf) == 0:
      return pd.DataFrame({c.noteIdKey: pd.Series(dtype=np.int64)})
    keepCols, tagCols = _ratings_keep_cols(ratingsDf)
    r = ratingsDf[[col for col in keepCols if col in ratingsDf.columns]]
    r = r.merge(raterFactorLookup, on=c.raterParticipantIdKey, how="left")
    factorDf = self._compute_factor_stats(r, prefix, includeDistributional=True)
    tagCounts = r.groupby(c.noteIdKey)[tagCols].sum()
    tagCounts.columns = [f"{prefix}tag_{col}" for col in tagCounts.columns]
    features = factorDf.join(tagCounts, how="outer").fillna(0)
    features.index.name = c.noteIdKey
    return features.reset_index()

  def _build_topic_factor_lookups(
    self, prescoringRaterModelOutput: pd.DataFrame
  ) -> Dict[str, pd.DataFrame]:
    """Build a dict mapping topic_name -> rater factor lookup for that topic's scorer."""
    topicLookups = {}
    for scorerName in prescoringRaterModelOutput[c.scorerNameKey].unique():
      if not str(scorerName).startswith("MFTopicScorer_"):
        continue
      topicName = str(scorerName).replace("MFTopicScorer_", "")
      lookup = self._extract_factor_lookup(
        prescoringRaterModelOutput, scorerName, factorColOut="raterFactor1"
      )
      topicLookups[topicName] = lookup
    return topicLookups

  def _compute_topic_rating_features(
    self,
    ratingsDf: pd.DataFrame,
    noteTopics: pd.DataFrame,
    topicFactorLookups: Dict[str, pd.DataFrame],
    prefix: str,
  ) -> pd.DataFrame:
    """Compute per-note aggregate features using topic-specific rater factors.

    For each note, looks up its topic, then uses that topic's rater factor
    to compute H/NH/SH factor mean/std/count.
    """
    if len(ratingsDf) == 0 or len(noteTopics) == 0:
      return pd.DataFrame({c.noteIdKey: pd.Series(dtype=np.int64)})
    r = ratingsDf[[c.noteIdKey, c.raterParticipantIdKey, c.helpfulnessLevelKey]]
    r = r.merge(noteTopics[[c.noteIdKey, c.noteTopicKey]], on=c.noteIdKey, how="left")
    r = r[r[c.noteTopicKey].notna() & (r[c.noteTopicKey] != _UNASSIGNED_TOPIC)]
    if len(r) == 0:
      return pd.DataFrame({c.noteIdKey: pd.Series(dtype=np.int64)})
    parts = []
    for topicName, grp in r.groupby(c.noteTopicKey):
      lookup = topicFactorLookups.get(str(topicName))
      if lookup is None or len(lookup) == 0:
        continue
      parts.append(grp.merge(lookup, on=c.raterParticipantIdKey, how="left"))
    if not parts:
      return pd.DataFrame({c.noteIdKey: pd.Series(dtype=np.int64)})
    r = pd.concat(parts, ignore_index=True)
    features = self._compute_factor_stats(r, prefix).fillna(0)
    features.index.name = c.noteIdKey
    return features.reset_index()

  # ── Training ──

  _AUTHOR_WINDOWS = ((14, "14d"), (90, "90d"))
  _AUTHOR_STATUSES = (
    (c.currentlyRatedHelpful, "crh"),
    (c.currentlyRatedNotHelpful, "crnh"),
    (c.needsMoreRatings, "nmr"),
  )

  def _compute_author_stats(
    self,
    noteIds: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    refTime: pd.DataFrame,
  ) -> pd.DataFrame:
    """Author CRH/CRNH/NMR counts in 14d/90d windows.

    Args:
      noteIds: DataFrame with noteIdKey.
      noteStatusHistory: full note status history.
      refTime: DataFrame with noteIdKey and '_ref_millis' -- per-note reference
        timestamp. Other notes' labels must precede this time and fall within
        the window.

    Implementation notes:
      Avoids a cartesian author self-join (scored notes x all notes by author),
      which OOMs when prolific authors have many historical notes. Instead, for
      each author sorts label timestamps once and answers each query with
      searchsorted on [ref - window, ref), then subtracts the query note's own
      label when it falls in-window.
    """
    authorCols = [
      f"{_AUTHOR_PREFIX}{label}_{suffix}"
      for _, suffix in self._AUTHOR_WINDOWS
      for _, label in self._AUTHOR_STATUSES
    ]
    authorStats = noteIds[[c.noteIdKey]].drop_duplicates(subset=[c.noteIdKey]).copy()
    for col in authorCols:
      authorStats[col] = 0.0
    if len(authorStats) == 0:
      return authorStats

    nsh = noteStatusHistory[
      [
        c.noteIdKey,
        c.noteAuthorParticipantIdKey,
        c.currentLabelKey,
        c.timestampMillisOfNoteCurrentLabelKey,
      ]
    ]
    noteAuthors = nsh[[c.noteIdKey, c.noteAuthorParticipantIdKey]].drop_duplicates(
      subset=[c.noteIdKey]
    )
    queries = authorStats[[c.noteIdKey]].merge(noteAuthors, on=c.noteIdKey, how="left")
    queries = queries.merge(refTime, on=c.noteIdKey, how="left")
    ownLabels = nsh[
      [c.noteIdKey, c.currentLabelKey, c.timestampMillisOfNoteCurrentLabelKey]
    ].drop_duplicates(subset=[c.noteIdKey])
    ownLabels = ownLabels.rename(
      columns={
        c.currentLabelKey: "_own_label",
        c.timestampMillisOfNoteCurrentLabelKey: "_own_millis",
      }
    )
    queries = queries.merge(ownLabels, on=c.noteIdKey, how="left")

    hist = nsh.loc[
      nsh[c.timestampMillisOfNoteCurrentLabelKey].notna(),
      [
        c.noteIdKey,
        c.noteAuthorParticipantIdKey,
        c.currentLabelKey,
        c.timestampMillisOfNoteCurrentLabelKey,
      ],
    ]
    if len(hist) == 0:
      return authorStats

    # noteId -> row position in authorStats for vectorized writes
    posByNoteId = pd.Series(np.arange(len(authorStats)), index=authorStats[c.noteIdKey].to_numpy())
    out = {col: authorStats[col].to_numpy(copy=True) for col in authorCols}

    queriesByAuthor = {
      author: grp
      for author, grp in queries.groupby(c.noteAuthorParticipantIdKey, sort=False)
      if pd.notna(author)
    }
    for author, hgrp in hist.groupby(c.noteAuthorParticipantIdKey, sort=False):
      qgrp = queriesByAuthor.get(author)
      if qgrp is None or len(qgrp) == 0:
        continue

      times = hgrp[c.timestampMillisOfNoteCurrentLabelKey].to_numpy(dtype=np.float64)
      labels = hgrp[c.currentLabelKey].to_numpy()
      order = np.argsort(times, kind="mergesort")
      times = times[order]
      labels = labels[order]

      prefixes = {}
      for status, label in self._AUTHOR_STATUSES:
        isStatus = (labels == status).astype(np.int64, copy=False)
        prefixes[label] = np.concatenate([[0], np.cumsum(isStatus)])

      qNoteIds = qgrp[c.noteIdKey].to_numpy()
      refs = qgrp["_ref_millis"].to_numpy(dtype=np.float64)
      ownMillis = qgrp["_own_millis"].to_numpy(dtype=np.float64)
      ownLabel = qgrp["_own_label"].to_numpy()
      # Skip queries with missing ref time (matches old join filters).
      validRef = np.isfinite(refs)
      if not validRef.any():
        continue
      qNoteIds = qNoteIds[validRef]
      refs = refs[validRef]
      ownMillis = ownMillis[validRef]
      ownLabel = ownLabel[validRef]
      positions = posByNoteId.reindex(qNoteIds).to_numpy()
      validPos = np.isfinite(positions)
      if not validPos.any():
        continue
      positions = positions[validPos].astype(np.int64)
      qNoteIds = qNoteIds[validPos]
      refs = refs[validPos]
      ownMillis = ownMillis[validPos]
      ownLabel = ownLabel[validPos]

      for windowDays, suffix in self._AUTHOR_WINDOWS:
        windowMillis = windowDays * 24 * 60 * 60 * 1000
        left = np.searchsorted(times, refs - windowMillis, side="left")
        right = np.searchsorted(times, refs, side="left")
        ownInWindow = (
          np.isfinite(ownMillis) & (ownMillis < refs) & ((refs - ownMillis) <= windowMillis)
        )
        for status, label in self._AUTHOR_STATUSES:
          col = f"{_AUTHOR_PREFIX}{label}_{suffix}"
          counts = prefixes[label][right] - prefixes[label][left]
          selfHit = ownInWindow & (ownLabel == status)
          counts = counts - selfHit.astype(np.int64, copy=False)
          out[col][positions] = counts

    for col in authorCols:
      authorStats[col] = out[col]
    return authorStats

  def _compute_author_stats_at_n(
    self,
    noteIds: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    nthRatingMillis: pd.DataFrame,
  ) -> pd.DataFrame:
    refTime = nthRatingMillis.rename(columns={"nth_rating_createdAtMillis": "_ref_millis"})
    return self._compute_author_stats(noteIds, noteStatusHistory, refTime)

  def _truncate_ratings_for_cascade_training(self, ratings: pd.DataFrame) -> pd.DataFrame:
    """Keep only ratings needed to build cascade features for all N <= max(cascade_ns).

    firstNAll for N uses rank <= N. firstNScored uses createdAtMillis <= time(rank N),
    which can include extra rows that share that timestamp. Ranks > max_n only affect
    those filters when they share the max_n-th rating's timestamp, so we keep:
      rank <= max_n  OR  createdAtMillis <= time(rank == max_n)
    for notes that have at least max_n ratings (notes with fewer keep all ratings).

    This is behavior-identical to keeping all ratings for every cascade N, while
    dropping the long tail of ratings beyond the cascade horizon.
    """
    maxN = max(self._cascadeNs)
    r = ratings.sort_values([c.noteIdKey, c.createdAtMillisKey], kind="mergesort")
    r["_global_rank"] = r.groupby(c.noteIdKey, sort=False).cumcount() + 1
    boundary = r.loc[r["_global_rank"] == maxN, [c.noteIdKey, c.createdAtMillisKey]].rename(
      columns={c.createdAtMillisKey: "_boundary_millis"}
    )
    r = r.merge(boundary, on=c.noteIdKey, how="left")
    keep = r["_global_rank"].le(maxN) | (
      r["_boundary_millis"].notna() & r[c.createdAtMillisKey].le(r["_boundary_millis"])
    )
    before = len(r)
    r = r.loc[keep].drop(columns=["_boundary_millis"])
    logger.info(
      f"PCRH rating truncate (max_n={maxN}): {before:,} -> {len(r):,} ratings "
      f"(keeps rank<=max_n plus timestamp ties at boundary)"
    )
    return r

  def _get_first_n_ratings(
    self, ratings: pd.DataFrame, scoredRaterIds: Set, n: int
  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (firstNAll, firstNScored, timingDf).

    firstNAll: the first N ratings by createdAtMillis for each note.
    firstNScored: scored-rater ratings with createdAtMillis <= Nth rating time
      (may exceed N rows when timestamps tie).
    timingDf: noteId -> nth_rating_createdAtMillis (timestamp of the Nth rating).
    """
    if "_global_rank" in ratings.columns:
      r = ratings
      rankCol = "_global_rank"
    else:
      r = ratings.sort_values([c.noteIdKey, c.createdAtMillisKey], kind="mergesort")
      r["_rank"] = r.groupby(c.noteIdKey, sort=False).cumcount() + 1
      rankCol = "_rank"
    noteCounts = r.groupby(c.noteIdKey, sort=False).size()
    notesWithEnough = noteCounts[noteCounts >= n].index
    r = r[r[c.noteIdKey].isin(notesWithEnough)]
    firstNAll = r[r[rankCol] <= n]
    if rankCol == "_rank":
      firstNAll = firstNAll.drop(columns=["_rank"])
    nthTimestamp = (
      firstNAll.groupby(c.noteIdKey, sort=False)[c.createdAtMillisKey]
      .max()
      .reset_index()
      .rename(columns={c.createdAtMillisKey: "nth_millis"})
    )
    rScored = r[r[c.raterParticipantIdKey].isin(scoredRaterIds)]
    rScored = rScored.merge(nthTimestamp, on=c.noteIdKey, how="inner")
    rScored = rScored[rScored[c.createdAtMillisKey] <= rScored["nth_millis"]]
    rScored = rScored.drop(columns=["nth_millis"], errors="ignore")
    timingDf = nthTimestamp.rename(columns={"nth_millis": "nth_rating_createdAtMillis"})
    dropCols = [col for col in ["_rank", "_global_rank"] if col in firstNAll.columns]
    if dropCols:
      firstNAll = firstNAll.drop(columns=dropCols)
    dropCols = [col for col in ["_rank", "_global_rank"] if col in rScored.columns]
    if dropCols:
      rScored = rScored.drop(columns=dropCols)
    return firstNAll, rScored, timingDf

  def _build_training_dataset_at_n(
    self,
    nVal: int,
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    raterFactorLookup: pd.DataFrame,
    scoredRaterIds: Set,
    labelDf: pd.DataFrame,
    noteTopics: Optional[pd.DataFrame] = None,
    topicFactorLookups: Optional[Dict[str, pd.DataFrame]] = None,
    excludeAlreadyDecided: bool = False,
  ) -> Optional[pd.DataFrame]:
    firstNAll, firstNScored, timingDf = self._get_first_n_ratings(ratings, scoredRaterIds, nVal)
    if len(firstNAll) == 0:
      return None
    featsAll = self._compute_rating_features(firstNAll, raterFactorLookup, _ALL_PREFIX)
    featsScored = self._compute_rating_features(firstNScored, raterFactorLookup, _SCORED_PREFIX)
    feats = featsAll.merge(featsScored, on=c.noteIdKey, how="outer")
    del featsAll, featsScored
    if noteTopics is not None and topicFactorLookups:
      topicFeatsAll = self._compute_topic_rating_features(
        firstNAll, noteTopics, topicFactorLookups, _TOPIC_ALL_PREFIX
      )
      topicFeatsScored = self._compute_topic_rating_features(
        firstNScored, noteTopics, topicFactorLookups, _TOPIC_SCORED_PREFIX
      )
      if c.noteIdKey in topicFeatsAll.columns:
        feats = feats.merge(topicFeatsAll, on=c.noteIdKey, how="left")
      if c.noteIdKey in topicFeatsScored.columns:
        feats = feats.merge(topicFeatsScored, on=c.noteIdKey, how="left")
      del topicFeatsAll, topicFeatsScored
    authorStats = self._compute_author_stats_at_n(feats[[c.noteIdKey]], noteStatusHistory, timingDf)
    feats = feats.merge(authorStats, on=c.noteIdKey, how="left")
    del authorStats
    del firstNAll, firstNScored
    feats = feats.merge(labelDf[[c.noteIdKey, "is_crh"]], on=c.noteIdKey, how="inner")
    feats["N"] = nVal
    if excludeAlreadyDecided:
      nshTiming = noteStatusHistory[
        [c.noteIdKey, c.timestampMillisOfNoteFirstNonNMRLabelKey, c.firstNonNMRLabelKey]
      ].drop_duplicates(subset=[c.noteIdKey])
      featsWithTiming = feats.merge(nshTiming, on=c.noteIdKey, how="left")
      del nshTiming
      featsWithTiming = featsWithTiming.merge(timingDf, on=c.noteIdKey, how="left")
      alreadyDecided = (
        (featsWithTiming[c.firstNonNMRLabelKey] == c.currentlyRatedHelpful)
        & featsWithTiming[c.timestampMillisOfNoteFirstNonNMRLabelKey].notna()
        & (
          featsWithTiming[c.timestampMillisOfNoteFirstNonNMRLabelKey]
          < featsWithTiming["nth_rating_createdAtMillis"]
        )
      )
      nExcluded = int(alreadyDecided.sum())
      if nExcluded > 0:
        logger.info(f"PCRH N={nVal}: excluding {nExcluded:,} notes already CRH before Nth rating")
      feats = feats.loc[~alreadyDecided.to_numpy()].copy()
      del featsWithTiming, alreadyDecided
    del timingDf
    if len(feats) < 50:
      logger.info(f"PCRH skip N={nVal}: only {len(feats)} notes")
      return None
    return feats

  def fit(
    self,
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    prescoringRaterModelOutput: pd.DataFrame,
    userEnrollment: Optional[pd.DataFrame] = None,
    noteTopics: Optional[pd.DataFrame] = None,
  ) -> None:
    ratings = self._filter_ratings_by_group(ratings, userEnrollment)
    keepCols, _ = _ratings_keep_cols(ratings)
    ratings = ratings.loc[:, [col for col in keepCols if col in ratings.columns]].copy()
    ratings[c.raterParticipantIdKey] = ratings[c.raterParticipantIdKey].astype(str)
    raterFactorLookup, scoredRaterIds = self._get_rater_factor_lookup(prescoringRaterModelOutput)
    topicFactorLookups = self._build_topic_factor_lookups(prescoringRaterModelOutput)
    if topicFactorLookups:
      logger.info(f"PCRH topic factor lookups: {sorted(topicFactorLookups.keys())}")
    labelDf = noteStatusHistory[[c.noteIdKey, c.currentLabelKey]].copy()
    labelDf["is_crh"] = (labelDf[c.currentLabelKey] == c.currentlyRatedHelpful).astype(int)
    labelDf = labelDf[[c.noteIdKey, "is_crh"]]
    recentNoteMask = notes[c.createdAtMillisKey] > self._trainingCutoffMillis
    recentNoteIds = notes.loc[recentNoteMask, c.noteIdKey]
    nRecentNotes = int(recentNoteMask.sum())
    recentRatings = ratings[ratings[c.noteIdKey].isin(set(recentNoteIds.to_numpy()))]
    del ratings, recentNoteIds, recentNoteMask
    gc.collect()
    logger.info(
      f"PCRH training: {nRecentNotes:,} notes, {len(recentRatings):,} ratings (pre-truncate)"
    )
    recentRatings = self._truncate_ratings_for_cascade_training(recentRatings)
    gc.collect()
    datasets = []
    for nVal in self._cascadeNs:
      df = self._build_training_dataset_at_n(
        nVal,
        notes,
        recentRatings,
        noteStatusHistory,
        raterFactorLookup,
        scoredRaterIds,
        labelDf,
        noteTopics=noteTopics,
        topicFactorLookups=topicFactorLookups,
        excludeAlreadyDecided=self._excludeAlreadyDecided,
      )
      if df is not None:
        datasets.append(df)
        logger.info(f"PCRH N={nVal}: {len(df):,} notes, CRH={df['is_crh'].sum():,}")
      gc.collect()
    del recentRatings, labelDf, raterFactorLookup, scoredRaterIds, topicFactorLookups
    gc.collect()
    if not datasets:
      logger.warning("PCRH: no training data available")
      return
    pooled = pd.concat(datasets, ignore_index=True)
    del datasets
    gc.collect()
    dropCols = {c.noteIdKey, "is_crh"}
    self._featureCols = [
      col for col in pooled.columns if col not in dropCols and pooled[col].dtype != object
    ]
    X = pooled[self._featureCols].fillna(0).to_numpy(dtype=np.float64, copy=True)
    y = pooled["is_crh"].to_numpy(copy=True)
    nPooled = len(pooled)
    crhRate = float(y.mean()) if nPooled else 0.0
    del pooled
    gc.collect()
    self._scaler = StandardScaler()
    xScaled = self._scaler.fit_transform(X)
    del X
    gc.collect()
    self._model = HistGradientBoostingClassifier(
      max_iter=self._nEstimators,
      max_depth=self._maxDepth,
      learning_rate=self._learningRate,
      random_state=self._seed,
      early_stopping=True,
      n_iter_no_change=10,
      validation_fraction=0.1,
    )
    self._model.fit(xScaled, y)
    del xScaled, y
    gc.collect()
    logger.info(
      f"PCRH model trained: {nPooled:,} rows, {len(self._featureCols)} features, "
      f"CRH rate={crhRate:.3f}"
    )

  # ── Inference ──

  def _compute_author_stats_current(
    self,
    noteIds: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
  ) -> pd.DataFrame:
    """Author CRH/CRNH/NMR counts in 14d/90d windows relative to now."""
    nowMillis = noteStatusHistory[c.timestampMillisOfNoteCurrentLabelKey].max()
    refTime = noteIds[[c.noteIdKey]].copy()
    refTime["_ref_millis"] = nowMillis
    return self._compute_author_stats(noteIds, noteStatusHistory, refTime)

  def _build_prediction_features(
    self,
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    raterFactorLookup: pd.DataFrame,
    scoredRaterIds: Set,
    noteTopics: Optional[pd.DataFrame] = None,
    topicFactorLookups: Optional[Dict[str, pd.DataFrame]] = None,
  ) -> pd.DataFrame:
    featsAll = self._compute_rating_features(ratings, raterFactorLookup, _ALL_PREFIX)
    rScored = ratings[ratings[c.raterParticipantIdKey].isin(scoredRaterIds)]
    featsScored = self._compute_rating_features(rScored, raterFactorLookup, _SCORED_PREFIX)
    feats = featsAll.merge(featsScored, on=c.noteIdKey, how="outer")
    if noteTopics is not None and topicFactorLookups:
      topicFeatsAll = self._compute_topic_rating_features(
        ratings, noteTopics, topicFactorLookups, _TOPIC_ALL_PREFIX
      )
      topicFeatsScored = self._compute_topic_rating_features(
        rScored, noteTopics, topicFactorLookups, _TOPIC_SCORED_PREFIX
      )
      if c.noteIdKey in topicFeatsAll.columns:
        feats = feats.merge(topicFeatsAll, on=c.noteIdKey, how="left")
      if c.noteIdKey in topicFeatsScored.columns:
        feats = feats.merge(topicFeatsScored, on=c.noteIdKey, how="left")
    authorStats = self._compute_author_stats_current(feats[[c.noteIdKey]], noteStatusHistory)
    feats = feats.merge(authorStats, on=c.noteIdKey, how="left")
    return feats

  def predict(
    self,
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    prescoringRaterModelOutput: pd.DataFrame,
    previousPcrhNoteIds: Optional[Set] = None,
    userEnrollment: Optional[pd.DataFrame] = None,
    noteTopics: Optional[pd.DataFrame] = None,
  ) -> pd.DataFrame:
    if self._model is None or self._scaler is None or self._featureCols is None:
      logger.warning("PCRH model not fitted, returning empty predictions")
      return pd.DataFrame(
        {
          c.noteIdKey: pd.Series(dtype=np.int64),
          c.pcrhAboveThresholdTimeKey: pd.Series(dtype=bool),
        }
      )
    if previousPcrhNoteIds is None:
      previousPcrhNoteIds = set()
    ratings = self._filter_ratings_by_group(ratings, userEnrollment)
    keepCols, _ = _ratings_keep_cols(ratings)
    ratings = ratings[keepCols].copy()
    ratings[c.raterParticipantIdKey] = ratings[c.raterParticipantIdKey].astype(str)
    raterFactorLookup, scoredRaterIds = self._get_rater_factor_lookup(prescoringRaterModelOutput)
    topicFactorLookups = self._build_topic_factor_lookups(prescoringRaterModelOutput)
    ratingCounts = ratings.groupby(c.noteIdKey).size().reset_index(name="n_ratings")
    minN = min(self._cascadeNs)
    eligible = ratingCounts[
      (ratingCounts["n_ratings"] >= minN) & (ratingCounts["n_ratings"] <= 100)
    ].copy()
    if len(eligible) == 0:
      return pd.DataFrame(
        {
          c.noteIdKey: pd.Series(dtype=np.int64),
          c.pcrhAboveThresholdTimeKey: pd.Series(dtype=bool),
        }
      )

    eligible["N"] = eligible["n_ratings"]
    eligibleNids = set(eligible[c.noteIdKey])
    notesE = notes[notes[c.noteIdKey].isin(eligibleNids)]
    ratingsE = ratings[ratings[c.noteIdKey].isin(eligibleNids)]
    feats = self._build_prediction_features(
      notesE,
      ratingsE,
      noteStatusHistory,
      raterFactorLookup,
      scoredRaterIds,
      noteTopics=noteTopics,
      topicFactorLookups=topicFactorLookups,
    )
    feats = feats.merge(eligible[[c.noteIdKey, "N"]], on=c.noteIdKey, how="inner")
    X = np.zeros((len(feats), len(self._featureCols)))
    for j, col in enumerate(self._featureCols):
      if col in feats.columns:
        X[:, j] = feats[col].fillna(0).values
    proba = self._model.predict_proba(self._scaler.transform(X))[:, 1]
    results = []
    for i in range(len(feats)):
      nid = feats[c.noteIdKey].iloc[i]
      nVal = int(feats["N"].iloc[i])
      p = float(proba[i])
      wasAbove = nid in previousPcrhNoteIds
      thresh = self._perNThresholds.get(nVal)
      if thresh is None:
        closest = min(self._perNThresholds.keys(), key=lambda k: abs(k - nVal))
        thresh = self._perNThresholds[closest]
      if wasAbove:
        revoked = p < self._revokeThreshold
        results.append(
          {
            c.noteIdKey: nid,
            c.pcrhAboveThresholdTimeKey: not revoked,
            c.pcrhExitNKey: float(nVal),
            c.pcrhExitProbaKey: p,
            c.pcrhRevokedKey: revoked,
            c.pcrhRevokeNKey: float(nVal) if revoked else None,
            c.pcrhFinalPredKey: not revoked,
          }
        )
      else:
        above = p >= thresh
        results.append(
          {
            c.noteIdKey: nid,
            c.pcrhAboveThresholdTimeKey: above,
            c.pcrhExitNKey: float(nVal),
            c.pcrhExitProbaKey: p,
            c.pcrhRevokedKey: False,
            c.pcrhRevokeNKey: None,
            c.pcrhFinalPredKey: above,
          }
        )
    if not results:
      return pd.DataFrame(
        {
          c.noteIdKey: pd.Series(dtype=np.int64),
          c.pcrhAboveThresholdTimeKey: pd.Series(dtype=bool),
        }
      )
    resultDf = pd.DataFrame(results)
    if len(resultDf) > 0:
      nNew = (
        resultDf[c.pcrhAboveThresholdTimeKey] & ~resultDf[c.noteIdKey].isin(previousPcrhNoteIds)
      ).sum()
      nMaintained = (
        resultDf[c.pcrhAboveThresholdTimeKey] & resultDf[c.noteIdKey].isin(previousPcrhNoteIds)
      ).sum()
      nRevoked = resultDf[c.pcrhRevokedKey].sum()
      logger.info(
        f"PCRH: {len(resultDf):,} notes, "
        f"{nNew:,} new, {nMaintained:,} maintained, {nRevoked:,} revoked"
      )
    return resultDf


def _convert_predictions_to_timestamps(
  pcrhPredictions: pd.DataFrame,
  previousPcrhTimes: Dict[int, float],
  currentTimeMillis: float,
) -> pd.DataFrame:
  """Convert boolean pcrhAboveThresholdTime values to millis timestamps.

  Encoding for pcrhAboveThresholdTimeKey:
    positive millis T0  -- currently above; T0 is first time ever above (one show window)
    negative -T0 (T0>1) -- not currently above, but was; preserves first-time for gating
    NaN                 -- never achieved above-threshold
    -1                  -- lock/legacy clear only (produced elsewhere); treated as no history

  A note may only be positive within [T0, T0+12h]. Re-entry inside the window restores
  +T0 (not now). Outside the window, even model-above is forced to -T0.
  """
  aboveNow = pcrhPredictions[c.pcrhAboveThresholdTimeKey].astype(bool).to_numpy()
  firstTime = _pcrh_mapped_first_times(pcrhPredictions[c.noteIdKey], previousPcrhTimes)
  hasFirst = ~np.isnan(firstTime)
  withinWindow = hasFirst & ((currentTimeMillis - firstTime) <= _PCRH_STALE_SUPPRESSION_MILLIS)
  # above+hasFirst+in window -> +T0; above+hasFirst+stale -> -T0;
  # above+no history -> now; not above+hasFirst -> -T0; else NaN.
  timeVals = np.where(
    aboveNow & hasFirst & withinWindow,
    firstTime,
    np.where(
      aboveNow & hasFirst,
      -firstTime,
      np.where(aboveNow, currentTimeMillis, np.where(hasFirst, -firstTime, np.nan)),
    ),
  )

  pcrhPredictions[c.pcrhAboveThresholdTimeKey] = timeVals
  nAbove = int((timeVals > 0).sum())
  nHistorical = int(((timeVals < 0) & (timeVals != _PCRH_CLEARED_TIME)).sum())
  logger.info(
    f"PCRH prediction summary: {nAbove} above, {nHistorical} historical (negated first-time)"
  )
  return pcrhPredictions


_EMPTY_PCRH_PREDICTIONS = {
  c.noteIdKey: pd.Series(dtype=np.int64),
  c.pcrhAboveThresholdTimeKey: pd.Series(dtype=np.double),
}

PCRH_AUX_COLS = [
  c.pcrhExitNKey,
  c.pcrhExitProbaKey,
  c.pcrhRevokedKey,
  c.pcrhRevokeNKey,
  c.pcrhFinalPredKey,
]


def compute_pcrh_predictions(
  pcrhClassifier: Optional["PCRHModel"],
  notes: pd.DataFrame,
  ratings: pd.DataFrame,
  noteStatusHistory: pd.DataFrame,
  prescoringRaterModelOutput: pd.DataFrame,
  previousScoredNotes: Optional[pd.DataFrame],
  currentTimeMillis: float,
  userEnrollment: Optional[pd.DataFrame] = None,
  noteTopics: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
  """Run PCRH prediction and convert to timestamps in one call.

  Previous state is taken from previousScoredNotes and noteStatusHistory so
  negated first-times survive across runs. previousPcrhNoteIds is only notes
  currently positive (for revoke-vs-entry threshold in predict).
  """
  previousPcrhNoteIds, previousPcrhTimes = collect_previous_pcrh_state(
    previousScoredNotes, noteStatusHistory
  )

  if pcrhClassifier is None:
    return pd.DataFrame(_EMPTY_PCRH_PREDICTIONS)

  preds = pcrhClassifier.predict(
    notes,
    ratings,
    noteStatusHistory,
    prescoringRaterModelOutput,
    previousPcrhNoteIds=previousPcrhNoteIds,
    userEnrollment=userEnrollment,
    noteTopics=noteTopics,
  )
  return _convert_predictions_to_timestamps(preds, previousPcrhTimes, currentTimeMillis)


def _write_suppressed_pcrh_times(
  times: np.ndarray,
  suppressMask: np.ndarray,
  firstTime: np.ndarray,
) -> np.ndarray:
  """For suppressed rows, set -T0 when first-time is known; else -1 if clearing a positive.

  Does not overwrite an existing historical negative (-T0) with -1.
  `firstTime` is aligned to `times` (NaN where no recoverable first-time).
  """
  if not suppressMask.any():
    return times
  hasFirst = ~np.isnan(firstTime)
  # Prefer -T0 when recoverable; else -1 for positive or missing current values.
  # Leave existing historical negatives and cleared -1 unchanged when no first-time.
  return np.where(
    suppressMask & hasFirst,
    -firstTime,
    np.where(
      suppressMask & (np.isnan(times) | (times > 0)),
      _PCRH_CLEARED_TIME,
      times,
    ),
  )


def suppress_pcrh_ineligible(
  scoredNotes: pd.DataFrame,
  previousPcrhNoteIds: Optional[Set] = None,
  previousPcrhTimes: Optional[Dict[int, float]] = None,
  currentTimeMillis: Optional[float] = None,
) -> pd.DataFrame:
  """Clear ineligible or out-of-window PCRH above-threshold status.

  Ineligible: not decided by CoreModel, firm-rejected by core, or
  classified NOT_MISLEADING. Must be called after decidedBy is assigned.

  On loss, writes -T0 (negated first-time) when a recoverable first-time is known,
  preserving the single show window for future gating. Falls back to -1 only when
  there is no recoverable first-time (legacy / previousPcrhNoteIds-only).

  Additionally, any note whose first-time is more than 12 hours before
  currentTimeMillis cannot remain positive -- forced to -T0 even if model-above.
  Existing historical negatives are left intact (not clobbered to -1).
  """
  if c.pcrhAboveThresholdTimeKey not in scoredNotes.columns:
    return scoredNotes
  if c.decidedByKey not in scoredNotes.columns:
    return scoredNotes

  coreRuleName = scoring_rules.RuleID.CORE_MODEL.get_name()
  notCore = scoredNotes[c.decidedByKey].fillna("") != coreRuleName
  coreFirmRejected = (scoredNotes[c.decidedByKey].fillna("") == coreRuleName) & scoredNotes[
    c.coreRatingStatusKey
  ].isin({c.firmReject, c.currentlyRatedNotHelpful})
  notMisleading = (
    (scoredNotes[c.classificationKey] == c.noteSaysTweetIsNotMisleadingKey).fillna(False)
    if c.classificationKey in scoredNotes.columns
    else False
  )
  # Current positive, or was positive last run but has no positive time this run
  # (e.g. missing prediction + ineligible decidedBy).
  wasPositive = scoredNotes[c.pcrhAboveThresholdTimeKey].fillna(0) > 0
  if previousPcrhNoteIds is not None:
    wasPositive = wasPositive | scoredNotes[c.noteIdKey].isin(previousPcrhNoteIds)
  suppressMask = ((notCore | coreFirmRejected | notMisleading) & wasPositive).to_numpy()

  times = scoredNotes[c.pcrhAboveThresholdTimeKey].to_numpy(dtype=np.double)
  firstFromCur = _pcrh_first_times_array(times)
  firstFromPrev = _pcrh_mapped_first_times(scoredNotes[c.noteIdKey], previousPcrhTimes)
  firstTime = np.where(np.isnan(firstFromCur), firstFromPrev, firstFromCur)

  if currentTimeMillis is not None:
    # Stale window: force non-positive if first-time is outside [T0, T0+12h].
    stale = (
      (times > 0)
      & ~np.isnan(firstTime)
      & ((currentTimeMillis - firstTime) > _PCRH_STALE_SUPPRESSION_MILLIS)
    )
    suppressMask = suppressMask | stale

  scoredNotes[c.pcrhAboveThresholdTimeKey] = _write_suppressed_pcrh_times(
    times, suppressMask, firstTime
  )
  return scoredNotes


def merge_pcrh_results(
  pcrhPredictions: pd.DataFrame,
  scoredNotes: pd.DataFrame,
  auxiliaryNoteInfo: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Merge PCRH predictions into scoredNotes/auxiliaryNoteInfo.

  Ensures all PCRH auxiliary columns exist in auxiliaryNoteInfo.
  Does NOT apply eligibility suppression -- call suppress_pcrh_ineligible
  after decidedBy is assigned.
  """
  if c.pcrhAboveThresholdTimeKey in pcrhPredictions.columns and len(pcrhPredictions) > 0:
    mainCols = [c.noteIdKey, c.pcrhAboveThresholdTimeKey] + (
      [c.pcrhExitProbaKey] if c.pcrhExitProbaKey in pcrhPredictions.columns else []
    )
    pcrhMain = pcrhPredictions[mainCols]
    dropCols = [
      col for col in (c.pcrhAboveThresholdTimeKey, c.pcrhExitProbaKey) if col in scoredNotes.columns
    ]
    if dropCols:
      scoredNotes = scoredNotes.drop(columns=dropCols)
    scoredNotes = scoredNotes.merge(pcrhMain, on=c.noteIdKey, how="left")
    pcrhAux = pcrhPredictions[
      [c.noteIdKey] + [col for col in PCRH_AUX_COLS if col in pcrhPredictions.columns]
    ]
    auxiliaryNoteInfo = auxiliaryNoteInfo.merge(pcrhAux, on=c.noteIdKey, how="left")
  else:
    scoredNotes[c.pcrhAboveThresholdTimeKey] = np.nan
  # Guarantee the raw-proba column exists on scoredNotes even when PCRH didn't run.
  if c.pcrhExitProbaKey not in scoredNotes.columns:
    scoredNotes[c.pcrhExitProbaKey] = np.nan

  # Ensure aux columns exist with correct dtypes even when PCRH didn't run,
  # so downstream TSV column-validation assertions don't fail.
  for col in PCRH_AUX_COLS:
    if col not in auxiliaryNoteInfo.columns:
      if col in (c.pcrhRevokedKey, c.pcrhFinalPredKey):
        auxiliaryNoteInfo[col] = pd.array([pd.NA] * len(auxiliaryNoteInfo), dtype=pd.BooleanDtype())
      else:
        auxiliaryNoteInfo[col] = np.nan

  return scoredNotes, auxiliaryNoteInfo
