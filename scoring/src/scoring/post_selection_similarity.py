import gc
import logging
import sys
from typing import Dict

from . import constants as c

import numpy as np
import pandas as pd


logger = logging.getLogger("birdwatch.post_selection_similarity")
logger.setLevel(logging.INFO)


class PostSelectionSimilarity:
  def __init__(
    self,
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
    pmiRegularization: int = 500,
    smoothedNpmiThreshold: float = 0.55,
    minimumRatingProportionThreshold: float = 0.4,
    minUniquePosts: int = 10,
    minSimPseudocounts: int = 10,
    windowMillis: int = 1000 * 60 * 20,
  ):
    # Compute rater affinity and writer coverage.  Apply thresholds to identify linked pairs.
    helpfulRatings = ratings[ratings[c.helpfulnessLevelKey] == c.helpfulValueTsv]
    self.affinityAndCoverage = self.compute_affinity_and_coverage(helpfulRatings, notes, [1, 5, 20])
    self.suspectPairs = self.get_suspect_pairs(self.affinityAndCoverage)

    # Compute MinSim and NPMI
    self.ratings = _preprocess_ratings(notes, ratings)
    with c.time_block("Compute pair counts dict"):
      self.pairCountsDict = _get_pair_counts_dict(self.ratings, windowMillis=windowMillis)

    self.uniqueRatingsOnTweets = self.ratings[
      [c.tweetIdKey, c.raterParticipantIdKey]
    ].drop_duplicates()
    raterTotals = self.uniqueRatingsOnTweets[c.raterParticipantIdKey].value_counts()
    raterTotalsDict = {
      index: value for index, value in raterTotals.items() if value >= minUniquePosts
    }

    self.pairCountsDict = _join_rater_totals_compute_pmi_and_filter_edges_below_threshold(
      pairCountsDict=self.pairCountsDict,
      raterTotalsDict=raterTotalsDict,
      N=len(self.uniqueRatingsOnTweets),
      pmiPseudocounts=pmiRegularization,
      minSimPseudocounts=minSimPseudocounts,
      smoothedNpmiThreshold=smoothedNpmiThreshold,
      minimumRatingProportionThreshold=minimumRatingProportionThreshold,
    )
    # _get_pair_counts_dict may have packed pair keys as a single int64 (see
    # _can_pack_rater_ids); downstream consumers (aggregate_into_cliques)
    # expect (left, right) tuples. At this point the dict has been filtered
    # down to pairs that passed PMI/minSim thresholds, so unpacking is cheap.
    if self.pairCountsDict and not isinstance(next(iter(self.pairCountsDict)), tuple):
      _ID_MASK = (1 << 32) - 1
      pssPairs = [(k >> 32, k & _ID_MASK) for k in self.pairCountsDict.keys()]
    else:
      pssPairs = list(self.pairCountsDict.keys())
    self.suspectPairs = set(self.suspectPairs + pssPairs)

  # Define helper to get affinity and coverage for a pair over a time window
  def _compute_affinity_and_coverage(self, ratings, notes, latencyMins, minDenom):
    # Identify ratings subset
    ratings = ratings[[c.raterParticipantIdKey, c.noteIdKey, c.createdAtMillisKey]].rename(
      columns={c.createdAtMillisKey: "ratingMillis"}
    )
    notes = notes[[c.noteAuthorParticipantIdKey, c.noteIdKey, c.createdAtMillisKey]].rename(
      columns={c.createdAtMillisKey: "noteMillis"}
    )
    ratings = ratings.merge(notes)
    ratings["latency"] = ratings["ratingMillis"] - ratings["noteMillis"]
    ratings = ratings[ratings["latency"] <= (1000 * 60 * latencyMins)]
    # Compute note and rating totals
    writerTotals = (
      notes[c.noteAuthorParticipantIdKey]
      .value_counts()
      .to_frame()
      .reset_index(drop=False)
      .rename(columns={"count": "writerTotal"})
    )
    raterTotals = (
      ratings[c.raterParticipantIdKey]
      .value_counts()
      .to_frame()
      .reset_index(drop=False)
      .rename(columns={"count": f"raterTotal{latencyMins}m"})
    )
    ratingTotals = (
      ratings[[c.noteAuthorParticipantIdKey, c.raterParticipantIdKey]]
      .value_counts()
      .reset_index(drop=False)
      .rename(columns={"count": f"pairTotal{latencyMins}m"})
    )
    ratingTotals = ratingTotals.merge(writerTotals, how="left")
    ratingTotals = ratingTotals.merge(raterTotals, how="left")
    # Compute ratios
    ratingTotals[f"raterAffinity{latencyMins}m"] = (
      ratingTotals[f"pairTotal{latencyMins}m"] / ratingTotals[f"raterTotal{latencyMins}m"]
    )
    ratingTotals[f"writerCoverage{latencyMins}m"] = (
      ratingTotals[f"pairTotal{latencyMins}m"] / ratingTotals["writerTotal"]
    )
    ratingTotals.loc[
      ratingTotals[f"raterTotal{latencyMins}m"] < minDenom, f"raterAffinity{latencyMins}m"
    ] = pd.NA
    ratingTotals.loc[
      ratingTotals["writerTotal"] < minDenom, f"writerCoverage{latencyMins}m"
    ] = pd.NA
    return ratingTotals[
      [
        c.noteAuthorParticipantIdKey,
        c.raterParticipantIdKey,
        "writerTotal",
        f"raterTotal{latencyMins}m",
        f"pairTotal{latencyMins}m",
        f"raterAffinity{latencyMins}m",
        f"writerCoverage{latencyMins}m",
      ]
    ].astype(
      {
        f"raterTotal{latencyMins}m": pd.Int64Dtype(),
        f"pairTotal{latencyMins}m": pd.Int64Dtype(),
      }
    )

  def compute_affinity_and_coverage(self, ratings, notes, latencyMins, minDenom=10):
    latencyMins = sorted(latencyMins, reverse=True)
    df = self._compute_affinity_and_coverage(ratings, notes, latencyMins[0], minDenom)
    origLen = len(df)
    for latency in latencyMins[1:]:
      df = df.merge(
        self._compute_affinity_and_coverage(ratings, notes, latency, minDenom),
        on=[c.noteAuthorParticipantIdKey, c.raterParticipantIdKey, "writerTotal"],
        how="left",
      )
      assert len(df) == origLen
    cols = [c.noteAuthorParticipantIdKey, c.raterParticipantIdKey, "writerTotal"]
    for latency in sorted(latencyMins):
      cols.extend(
        [
          f"raterTotal{latency}m",
          f"pairTotal{latency}m",
          f"raterAffinity{latency}m",
          f"writerCoverage{latency}m",
        ]
      )
    return df[cols]

  def get_suspect_pairs(self, affinityAndCoverage):
    thresholds = [
      ("writerCoverage1m", 0.2),
      ("writerCoverage5m", 0.3),
      ("writerCoverage20m", 0.4),
      ("raterAffinity1m", 0.2),
      ("raterAffinity5m", 0.45),
      ("raterAffinity20m", 0.7),
    ]
    suspectPairsDF = []
    for col, value in thresholds:
      tmp = affinityAndCoverage[affinityAndCoverage[col] >= value][
        [c.noteAuthorParticipantIdKey, c.raterParticipantIdKey]
      ].copy()
      suspectPairsDF.append(tmp)
    suspectPairsDF = pd.concat(suspectPairsDF)
    suspectPairs = []
    for author, rater in suspectPairsDF[
      [c.noteAuthorParticipantIdKey, c.raterParticipantIdKey]
    ].values:
      suspectPairs.append(tuple(sorted((author, rater))))
    return suspectPairs

  def get_post_selection_similarity_values(self):
    """
    Returns dataframe with [raterParticipantId, postSelectionSimilarityValue] columns.
    postSelectionSimilarityValue is None by default.
    """
    cliqueToUserMap, userToCliqueMap = aggregate_into_cliques(self.suspectPairs)

    # Convert dict to pandas dataframe
    cliquesDfList = []
    for cliqueId in cliqueToUserMap.keys():
      for userId in cliqueToUserMap[cliqueId]:
        cliquesDfList.append({c.raterParticipantIdKey: userId, c.postSelectionValueKey: cliqueId})
    cliquesDf = pd.DataFrame(
      cliquesDfList, columns=[c.raterParticipantIdKey, c.postSelectionValueKey]
    )
    return cliquesDf


def apply_post_selection_similarity(notes, ratings, postSelectionSimilarityValues):
  """
  Filters out ratings after the first on each note from raters who have high post selection similarity,
  or filters all if the note is authored by a user with the same post selection similarity value.
  """
  # Summarize input
  logger.info(f"Total ratings prior to applying post selection similarity: {len(ratings)}")
  logger.info(
    f"Total unique ratings before: {len(ratings[[c.noteIdKey, c.raterParticipantIdKey]].drop_duplicates())}"
  )
  pssSummary = (
    postSelectionSimilarityValues[[c.postSelectionValueKey, c.quasiCliqueValueKey]] > 0
  ).sum()
  logger.info(f"Summary of postSelectionSimilarityValues: \n{pssSummary}")
  # Add additional column with bit flagging correlated raters
  correlatedRaters = set(
    postSelectionSimilarityValues[postSelectionSimilarityValues[c.quasiCliqueValueKey] >= 1][
      c.raterParticipantIdKey
    ]
  )
  ratings[c.correlatedRaterKey] = ratings[c.raterParticipantIdKey].isin(correlatedRaters)
  logger.info(
    f"correlatedRater set on {ratings[c.correlatedRaterKey].sum()} ratings from {len(correlatedRaters)} unique raters"
  )
  # Trim correlated raters out of PSS dataframe and remove drop ratings flagged by NPMI/MinSim
  postSelectionSimilarityValues = (
    postSelectionSimilarityValues[[c.raterParticipantIdKey, c.postSelectionValueKey]]
    .dropna()
    .drop_duplicates()
  )
  ratingsWithPostSelectionSimilarity = (
    ratings.merge(
      postSelectionSimilarityValues,
      on=c.raterParticipantIdKey,
      how="left",
    )
    .merge(notes[[c.noteIdKey, c.noteAuthorParticipantIdKey]], on=c.noteIdKey, how="left")
    .merge(
      postSelectionSimilarityValues,
      left_on=c.noteAuthorParticipantIdKey,
      right_on=c.raterParticipantIdKey,
      how="left",
      suffixes=("", "_note_author"),
    )
  )
  ratingsWithNoPostSelectionSimilarityValue = ratingsWithPostSelectionSimilarity[
    pd.isna(ratingsWithPostSelectionSimilarity[c.postSelectionValueKey])
  ]
  ratingsWithPostSelectionSimilarityValue = ratingsWithPostSelectionSimilarity[
    (~pd.isna(ratingsWithPostSelectionSimilarity[c.postSelectionValueKey]))
    & (
      ratingsWithPostSelectionSimilarity[c.postSelectionValueKey]
      != ratingsWithPostSelectionSimilarity[c.postSelectionValueKey + "_note_author"]
    )
  ]
  ratingsWithPostSelectionSimilarityValue.sort_values(
    by=[c.noteIdKey, c.createdAtMillisKey], ascending=True, inplace=True
  )
  ratingsWithPostSelectionSimilarityValue.drop_duplicates(
    subset=[c.noteIdKey, c.postSelectionValueKey], keep="first", inplace=True
  )

  if len(notes) < c.minNumNotesForProdData:
    return ratings

  ratings = pd.concat(
    [ratingsWithPostSelectionSimilarityValue, ratingsWithNoPostSelectionSimilarityValue], axis=0
  )
  ratings.drop(
    columns={c.noteAuthorParticipantIdKey, c.raterParticipantIdKey + "_note_author"},
    errors="ignore",
    inplace=True,
  )
  logger.info(f"Total ratings after to applying post selection similarity: {len(ratings)}")
  logger.info(
    f"Total unique ratings after: {len(ratings[[c.noteIdKey, c.raterParticipantIdKey]].drop_duplicates())}"
  )
  return ratings


def _preprocess_ratings(notes: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
  """
  Preprocess ratings dataframe.
  """
  ratings = notes[[c.noteIdKey, c.tweetIdKey]].merge(
    ratings[[c.raterParticipantIdKey, c.noteIdKey, c.createdAtMillisKey]],
    on=c.noteIdKey,
    how="inner",
  )
  ratings = ratings[(ratings[c.tweetIdKey] != -1) & (ratings[c.tweetIdKey] != "-1")]
  return ratings


def _join_rater_totals_compute_pmi_and_filter_edges_below_threshold(
  pairCountsDict: Dict,
  raterTotalsDict: Dict,
  N: int,
  pmiPseudocounts: int,
  minSimPseudocounts: int,
  smoothedNpmiThreshold: float,
  minimumRatingProportionThreshold: float,
):
  keys_to_delete = []
  # _get_pair_counts_dict either packs (left, right) into a single int64 key
  # (production path) or uses (left, right) tuples (tests / unpackable IDs).
  # Detect once here so the inner loop stays unconditional.
  use_packed_keys = bool(pairCountsDict) and not isinstance(next(iter(pairCountsDict)), tuple)
  _ID_MASK = (1 << 32) - 1

  with c.time_block("Compute PMI and minSim"):
    for pairKey in pairCountsDict:
      if use_packed_keys:
        leftRaterId = pairKey >> 32
        rightRaterId = pairKey & _ID_MASK
      else:
        leftRaterId, rightRaterId = pairKey
      if leftRaterId not in raterTotalsDict or rightRaterId not in raterTotalsDict:
        keys_to_delete.append(pairKey)
        continue

      leftTotal = raterTotalsDict[leftRaterId]
      rightTotal = raterTotalsDict[rightRaterId]
      coRatings = pairCountsDict[pairKey]

      if type(coRatings) != int:
        # already processed (should only occur when re-running...)
        continue

      # PMI
      pmiNumerator = coRatings * N
      pmiDenominator = (leftTotal + pmiPseudocounts) * (rightTotal + pmiPseudocounts)
      smoothedPmi = np.log(pmiNumerator / pmiDenominator)
      smoothedNpmi = smoothedPmi / -np.log(coRatings / N)

      # minSim
      minTotal = min(leftTotal, rightTotal)
      minSimRatingProp = coRatings / (minTotal + minSimPseudocounts)

      if (smoothedNpmi >= smoothedNpmiThreshold) or (
        minSimRatingProp >= minimumRatingProportionThreshold
      ):
        pairCountsDict[pairKey] = (smoothedNpmi, minSimRatingProp)
      else:
        keys_to_delete.append(pairKey)

    print(f"Pairs dict used {sys.getsizeof(pairCountsDict) * 1e-9}GB RAM at max")

  with c.time_block("Delete unneeded pairs from pairCountsDict"):
    for key in keys_to_delete:
      del pairCountsDict[key]

  print(
    f"Pairs dict used {sys.getsizeof(pairCountsDict) * 1e-9}GB RAM after deleted unneeded pairs"
  )

  return pairCountsDict


def aggregate_into_cliques(suspectPairs):
  with c.time_block("Aggregate into cliques by post selection similarity"):
    userToCliqueMap = dict()
    cliqueToUserMap = dict()

    nextNewCliqueId = 1  # start cliqueIdxs from 1

    for sid, tid in suspectPairs:
      if sid in userToCliqueMap:
        if tid in userToCliqueMap:
          # both in map. merge if not same clique
          if userToCliqueMap[sid] != userToCliqueMap[tid]:
            # merge. assign all member's of target clique to source clique.
            # slow way: iterate over all values here.
            # fast way: maintain a reverse map of cliqueToUserMap.
            sourceDestClique = userToCliqueMap[sid]
            oldTargetCliqueToDel = userToCliqueMap[tid]

            for userId in cliqueToUserMap[oldTargetCliqueToDel]:
              cliqueToUserMap[sourceDestClique].append(userId)
              userToCliqueMap[userId] = sourceDestClique
            del cliqueToUserMap[oldTargetCliqueToDel]
            gc.collect()

        else:
          # source in map; target not. add target to source's clique
          sourceClique = userToCliqueMap[sid]
          userToCliqueMap[tid] = sourceClique
          cliqueToUserMap[sourceClique].append(tid)
      elif tid in userToCliqueMap:
        # target in map; source not. add source to target's clique
        targetClique = userToCliqueMap[tid]
        userToCliqueMap[sid] = targetClique
        cliqueToUserMap[targetClique].append(sid)
      else:
        # new clique
        userToCliqueMap[sid] = nextNewCliqueId
        userToCliqueMap[tid] = nextNewCliqueId
        cliqueToUserMap[nextNewCliqueId] = [sid, tid]
        nextNewCliqueId += 1
  return cliqueToUserMap, userToCliqueMap


def _can_pack_rater_ids(ratings) -> bool:
  """Whether rater IDs can be packed as a single int64 dict key.

  Packing requires non-negative integer IDs strictly less than 2**32 so we can
  encode ``(smaller_id << 32) | larger_id`` without collisions. The upstream
  participant-ID normalization step produces a dense ``int64`` range starting
  at 0 that satisfies this; callers using raw string IDs (e.g. unit tests with
  toy data) fall through to the slower tuple-keyed path.
  """
  rater_col = ratings[c.raterParticipantIdKey]
  if len(rater_col) == 0:
    return False
  if not pd.api.types.is_integer_dtype(rater_col):
    return False
  return int(rater_col.min()) >= 0 and int(rater_col.max()) < (1 << 32)


def _get_pair_counts_dict(ratings, windowMillis):
  """Count co-rating events within a sliding time window, per tweet.

  Pair keys are stored as a single packed Python int rather than a
  ``(left, right)`` tuple when rater IDs are non-negative ints < 2**32:
  ``(smaller_id << 32) | larger_id``. This roughly halves the per-entry memory
  of the dict (which can reach ~10^9 entries at scoring scale) by removing the
  tuple object and the two boxed scalars per pair. When IDs can't be packed
  (string IDs, negative IDs, or IDs >= 2**32), the function falls back to
  tuple-keyed counts. ``_join_rater_totals_...`` and the ``__init__`` caller
  detect which key format was used and unpack on demand.
  """
  pack_keys = _can_pack_rater_ids(ratings)

  pair_counts: Dict = dict()

  # Group by tweetIdKey to process each tweet individually
  grouped_by_tweet = ratings.groupby(c.tweetIdKey, sort=False)

  if pack_keys:
    for _, tweet_group in grouped_by_tweet:
      # Keep track of pairs we've already counted for this tweetId
      pairs_counted_in_tweet: set = set()
      grouped_by_note = tweet_group.groupby(c.noteIdKey, sort=False)
      for _, note_group in grouped_by_note:
        note_group.sort_values(c.createdAtMillisKey, inplace=True)
        # Convert raters to a Python list of ints once per note group so the
        # inner loop avoids per-iteration numpy.int64 boxing (each fresh
        # scalar would be 32 bytes and would be retained by the dict entries
        # that reference it).
        times = note_group[c.createdAtMillisKey].values
        raters = note_group[c.raterParticipantIdKey].values.tolist()
        n = len(note_group)
        window_start = 0
        for i in range(n):
          while times[i] - times[window_start] > windowMillis:
            window_start += 1
          a = raters[i]
          for j in range(window_start, i):
            b = raters[j]
            if a == b:
              continue
            pair = (a << 32) | b if a < b else (b << 32) | a
            if pair not in pairs_counted_in_tweet:
              pairs_counted_in_tweet.add(pair)
              if pair not in pair_counts:
                pair_counts[pair] = 0
              pair_counts[pair] += 1
  else:
    # Fallback path: IDs are strings or don't fit in 32 bits, so dict keys
    # are ``(left, right)`` tuples. Slower and more memory-hungry; used by
    # tests with toy string IDs and by any caller that skips upstream
    # participant-ID normalization.
    for _, tweet_group in grouped_by_tweet:
      pairs_counted_in_tweet = set()
      grouped_by_note = tweet_group.groupby(c.noteIdKey, sort=False)
      for _, note_group in grouped_by_note:
        note_group.sort_values(c.createdAtMillisKey, inplace=True)
        times = note_group[c.createdAtMillisKey].values
        raters = note_group[c.raterParticipantIdKey].values
        n = len(note_group)
        window_start = 0
        for i in range(n):
          while times[i] - times[window_start] > windowMillis:
            window_start += 1
          for j in range(window_start, i):
            if raters[i] != raters[j]:
              left_rater, right_rater = tuple(sorted((raters[i], raters[j])))
              pair = (left_rater, right_rater)
              if pair not in pairs_counted_in_tweet:
                pairs_counted_in_tweet.add(pair)
                if pair not in pair_counts:
                  pair_counts[pair] = 0
                pair_counts[pair] += 1

  return pair_counts
