import gc
import logging
from typing import Dict

from . import constants as c

import numpy as np
import pandas as pd


logger = logging.getLogger("birdwatch.post_selection_similarity")
logger.setLevel(logging.INFO)


class PostSelectionSimilarity:
  def __init__(self):
    pass

  def initialize(
    self,
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
    pmiRegularization: int = 500,
    smoothedNpmiThreshold: float = 0.45,
    minimumRatingProportionThreshold: float = 0.4,
    minUniquePosts: int = 10,
    minSimPseudocounts: int = 10,
    windowMillis: int = 1000 * 60 * 20,
  ):
    self.ratings = _preprocess_ratings(notes, ratings)
    self.pairCounts = _get_pair_tuples(self.ratings, windowMillis=windowMillis)
    self.pairStatsDf = _tuples_to_df(self.pairCounts)
    self.uniqueRatingsOnTweets = self.ratings[
      [c.tweetIdKey, c.raterParticipantIdKey]
    ].drop_duplicates()
    self.pairStatsDf = _join_rater_totals(self.pairStatsDf, self.uniqueRatingsOnTweets)
    self.pmiDf = _compute_pmi(
      self.pairStatsDf, len(self.uniqueRatingsOnTweets), pmiRegularization, minSimPseudocounts
    )

    self.filter_edges_below_threshold(
      smoothedNpmiThreshold, minimumRatingProportionThreshold, minUniquePosts
    )

  def filter_edges_below_threshold(
    self, smoothedNpmiThreshold, minimumRatingProportionThreshold, minUniquePosts
  ):
    self.graphDf = self.pmiDf[
      (self.pmiDf["smoothedNpmi"] >= smoothedNpmiThreshold)
      | (
        (self.pmiDf["minSimRatingProp"] >= minimumRatingProportionThreshold)
        & (self.pmiDf["minTotal"] >= minUniquePosts)
      )
    ]

  def get_high_post_selection_similarity_raters(self):
    highPostSelectionSimilarityRaters = pd.concat(
      [
        self.graphDf[["leftRaterId"]].rename(columns={"leftRaterId": c.raterParticipantIdKey}),
        self.graphDf[["rightRaterId"]].rename(columns={"rightRaterId": c.raterParticipantIdKey}),
      ]
    ).drop_duplicates()
    highPostSelectionSimilarityRaters[c.postSelectionValueKey] = 1
    return highPostSelectionSimilarityRaters

  def get_post_selection_similarity_values(self):
    """
    Returns dataframe with [raterParticipantId, postSelectionSimilarityValue] columns.
    postSelectionSimilarityValue is None by default.
    """
    cliqueToUserMap, userToCliqueMap = aggregate_into_cliques(self.graphDf)

    # Convert dict to pandas dataframe
    cliquesDfList = []
    for cliqueId in cliqueToUserMap.keys():
      for userId in cliqueToUserMap[cliqueId]:
        cliquesDfList.append({c.raterParticipantIdKey: userId, c.postSelectionValueKey: cliqueId})
    cliquesDf = pd.DataFrame(
      cliquesDfList, columns=[c.raterParticipantIdKey, c.postSelectionValueKey]
    )
    return cliquesDf


def filter_ratings_by_post_selection_similarity(notes, ratings, postSelectionSimilarityValues):
  """
  Filters out ratings after the first on each note from raters who have high post selection similarity,
  or filters all if the note is authored by a user with the same post selection similarity value.
  """
  ratingsWithPostSelectionSimilarity = (
    ratings.merge(
      postSelectionSimilarityValues,
      on=c.raterParticipantIdKey,
      how="left",
      unsafeAllowed=c.postSelectionValueKey,
    )
    .merge(notes[[c.noteIdKey, c.noteAuthorParticipantIdKey]], on=c.noteIdKey, how="left")
    .merge(
      postSelectionSimilarityValues,
      left_on=c.noteAuthorParticipantIdKey,
      right_on=c.raterParticipantIdKey,
      how="left",
      suffixes=("", "_note_author"),
      unsafeAllowed={c.postSelectionValueKey, c.postSelectionValueKey + "_note_author"},
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

  ratings = pd.concat(
    [ratingsWithPostSelectionSimilarityValue, ratingsWithNoPostSelectionSimilarityValue], axis=0
  )
  ratings.drop(
    columns={c.noteAuthorParticipantIdKey, c.raterParticipantIdKey + "_note_author"},
    errors="ignore",
    inplace=True,
  )
  return ratings


def filter_all_ratings_by_post_selection_similarity(ratings, highPostSelectionSimilarityRaters):
  """
  Deprecated.
  Filters out all ratings from raters who have high post selection similarity.
  """
  ratings = ratings.merge(
    highPostSelectionSimilarityRaters, on=c.raterParticipantIdKey, how="left", indicator=True
  )
  ratings = ratings[ratings["_merge"] == "left_only"]
  ratings = ratings.drop(columns=["_merge"])
  return ratings


def _compute_pmi(
  pairStatsDf: pd.DataFrame, N: int, pmiPseudocounts: int = 500, minSimPseudocounts: int = 10
) -> pd.DataFrame:
  """
  Compute PMI between raters.
  """
  numerator = pairStatsDf["pairRatings"] * N
  denominator = (pairStatsDf["leftTotal"] + pmiPseudocounts) * (
    pairStatsDf["rightTotal"] + pmiPseudocounts
  )
  pairStatsDf["smoothedPmi"] = np.log(numerator / denominator)
  pairStatsDf["smoothedNpmi"] = pairStatsDf["smoothedPmi"] / -np.log(pairStatsDf["pairRatings"] / N)
  pairStatsDf["minTotal"] = np.minimum(pairStatsDf["leftTotal"], pairStatsDf["rightTotal"])
  pairStatsDf["minSimRatingProp"] = pairStatsDf["pairRatings"] / (
    pairStatsDf["minTotal"] + minSimPseudocounts
  )
  return pairStatsDf


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


def _join_rater_totals(
  pairStatsDf: pd.DataFrame, uniqueRatingsOnTweets: pd.DataFrame, minRatings: int = 10
):
  raterTotals = uniqueRatingsOnTweets[c.raterParticipantIdKey].value_counts().reset_index()
  raterTotals.columns = [c.raterParticipantIdKey, "count"]
  raterTotals = raterTotals[raterTotals["count"] >= minRatings]
  pairStatsDf = pairStatsDf.merge(
    raterTotals.rename(columns={c.raterParticipantIdKey: "leftRaterId", "count": "leftTotal"})
  )
  pairStatsDf = pairStatsDf.merge(
    raterTotals.rename(columns={c.raterParticipantIdKey: "rightRaterId", "count": "rightTotal"})
  )
  return pairStatsDf


def aggregate_into_cliques(graphDf):
  with c.time_block("Aggregate into cliques by post selection similarity"):
    userToCliqueMap = dict()
    cliqueToUserMap = dict()

    nextNewCliqueId = 1  # start cliqueIdxs from 1
    for i, row in graphDf.iterrows():
      sid = row["leftRaterId"]
      tid = row["rightRaterId"]
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


def _make_rater_stats_df(pairCounts):
  with c.time_block("Making rater stats dataframe from pair counts dict"):
    leftRater, rightRater, pairRatings = [], [], []
    for i, ((left, right), count) in enumerate(pairCounts.items()):
      leftRater.append(left)
      rightRater.append(right)
      pairRatings.append(count)
  return pd.DataFrame(
    {
      "leftRaterId": np.array(leftRater),
      "rightRaterId": np.array(rightRater),
      "pairRatings": np.array(pairRatings),
    }
  )

def _get_pair_counts_df_dict(ratings, windowMillis):
    import numpy as np
    import pandas as pd
    from collections import defaultdict

    # Assign column keys to local variables for faster access
    noteIdKey = c.noteIdKey
    createdAtMillisKey = c.createdAtMillisKey
    raterParticipantIdKey = c.raterParticipantIdKey

    # Sort ratings by noteIdKey and createdAtMillisKey
    ratings_sorted = ratings.sort_values([noteIdKey, createdAtMillisKey])

    # Initialize a defaultdict to store counts of pairs
    pair_counts = defaultdict(int)

    # Group by noteIdKey to process each note individually
    grouped = ratings_sorted.groupby(noteIdKey, sort=False)

    for noteId, group in grouped:
        # Extract relevant columns as numpy arrays for efficient computation
        times = group[createdAtMillisKey].values
        raters = group[raterParticipantIdKey].values

        n = len(group)
        window_start = 0

        for i in range(n):
            # Move the window start forward if the time difference exceeds windowMillis
            while times[i] - times[window_start] > windowMillis:
                window_start += 1

            # For all indices within the sliding window (excluding the current index)
            for j in range(window_start, i):
                if raters[i] != raters[j]:
                    left_rater, right_rater = tuple(sorted((raters[i], raters[j])))
                    # Update the count for this pair
                    pair_counts[(left_rater, right_rater)] += 1

    # Convert the pair_counts dictionary to a DataFrame
    if pair_counts:
        pairs = np.array(list(pair_counts.keys()))
        counts = np.array(list(pair_counts.values()))
        df = pd.DataFrame({
            'leftRaterId': pairs[:, 0],
            'rightRaterId': pairs[:, 1],
            'pairRatings': counts
        })
    else:
        # Return an empty DataFrame with appropriate columns
        df = pd.DataFrame(columns=['leftRaterId', 'rightRaterId', 'pairRatings'])

    return df


def _get_pair_ratings_df_optimized(ratings, windowMillis):

    # Assign column keys to local variables for faster access
    noteIdKey = c.noteIdKey
    createdAtMillisKey = c.createdAtMillisKey
    raterParticipantIdKey = c.raterParticipantIdKey
    tweetIdKey = c.tweetIdKey

    # Sort ratings by noteIdKey and createdAtMillisKey
    ratings_sorted = ratings.sort_values([noteIdKey, createdAtMillisKey])

    # Initialize lists to collect data
    left_raters = []
    right_raters = []
    tweet_ids = []

    # Group by noteIdKey to process each note individually
    grouped = ratings_sorted.groupby(noteIdKey, sort=False)

    for noteId, group in grouped:
        # Extract relevant columns as numpy arrays for efficient computation
        times = group[createdAtMillisKey].values
        raters = group[raterParticipantIdKey].values
        tweetId = group[tweetIdKey].iloc[0]  # Assuming tweetIdKey is constant within a note

        n = len(group)
        window_start = 0

        for i in range(n):
            # Move the window start forward if the time difference exceeds windowMillis
            while times[i] - times[window_start] > windowMillis:
                window_start += 1

            # For all indices within the sliding window (excluding the current index)
            for j in range(window_start, i):
                if raters[i] != raters[j]:
                    left_rater, right_rater = tuple(sorted((raters[i], raters[j])))
                    left_raters.append(left_rater)
                    right_raters.append(right_rater)
                    tweet_ids.append(tweetId)

    # Convert lists to numpy arrays for efficient DataFrame creation
    left_raters = np.array(left_raters)
    right_raters = np.array(right_raters)
    tweet_ids = np.array(tweet_ids)

    # Create the DataFrame from the collected data
    df = pd.DataFrame({
        'leftRaterId': left_raters,
        'rightRaterId': right_raters,
        'tweetId': tweet_ids,
    })

    # Drop duplicates
    df = df.drop_duplicates()

    # Group by leftRaterId and rightRaterId and count the number of occurrences
    df = (
        df.groupby(['leftRaterId', 'rightRaterId'], as_index=False)
        .agg(pairRatings=('tweetId', 'count'))
    )
    return df


# get number of ratings per pair in same time window
def _get_pair_tuples(ratings, windowMillis):
  tuples = []
  ratings = ratings.sort_values([c.noteIdKey, c.createdAtMillisKey])
  values = ratings[
    [c.noteIdKey, c.createdAtMillisKey, c.raterParticipantIdKey, c.tweetIdKey]
  ].values
  print(len(values))
  for i in range(len(values)):
    priorNote, priorTs, priorRater, priorTweet = values[i]
    if i == 0 or i == 1000 or i == 100000 or i % 5000000 == 0:
      print(f"i={i}  len(tuples)={len(tuples)}")
    j = i + 1
    while j < len(values):
      nextNote, nextTs, nextRater, nextTweet = values[j]
      assert priorNote <= nextNote, (priorNote, nextNote)
      if nextNote != priorNote:
        break  # break if we're onto a new note
      assert priorTweet == nextTweet, (priorTweet, nextTweet)  # tweet should be same
      assert priorRater != nextRater, (priorRater, nextRater)  # rater should be different
      assert priorTs <= nextTs, (priorTs, nextTs)
      if nextTs > (priorTs + windowMillis):
        break  # break if we're beyond the overlap window
      leftRater, rigthRater = tuple(sorted((priorRater, nextRater)))
      tuples.append((leftRater, rigthRater, priorTweet))
      j += 1
  return tuples

def _get_pair_tuples_optimized(ratings, windowMillis):

    # Sort ratings by noteIdKey and createdAtMillisKey
    ratings_sorted = ratings.sort_values([c.noteIdKey, c.createdAtMillisKey])

    # Initialize an empty list to store the result
    tuples = []

    # Group by noteIdKey to process each note individually
    grouped = ratings_sorted.groupby(c.noteIdKey, sort=False)

    for noteId, group in grouped:
        # Extract relevant columns as numpy arrays for efficient computation
        times = group[c.createdAtMillisKey].values
        raters = group[c.raterParticipantIdKey].values
        priorTweet = group[c.tweetIdKey].iloc[0]

        n = len(group)
        window_start = 0  # Start index of the sliding window

        for i in range(n):
            # Move the window start forward if the time difference exceeds windowMillis
            while times[i] - times[window_start] > windowMillis:
                window_start += 1

            # For all indices within the sliding window (excluding the current index)
            for j in range(window_start, i):
                # Check if raters are different
                if raters[i] != raters[j]:
                    # Sort raters to maintain consistency
                    leftRater, rightRater = tuple(sorted((raters[i], raters[j])))
                    tuples.append((leftRater, rightRater, priorTweet))

    return tuples


import multiprocessing as mp

def _get_pair_tuples_parallel(ratings, windowMillis):
    # Sort and group ratings
    ratings_sorted = ratings.sort_values([c.noteIdKey, c.createdAtMillisKey])
    grouped = ratings_sorted.groupby(c.noteIdKey, sort=False)

    # Prepare arguments for parallel processing
    args = [(group, windowMillis) for _, group in grouped]

    # Use multiprocessing Pool
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(_get_pair_tuples_process_group, args)

    # Flatten the list of results
    tuples = [tup for sublist in results for tup in sublist]
    return tuples

def _get_pair_tuples_process_group(group, windowMillis):
    # Same logic as before, applied to a single group
    times = group[c.createdAtMillisKey].values
    raters = group[c.raterParticipantIdKey].values
    priorTweet = group[c.tweetIdKey].iloc[0]

    n = len(group)
    window_start = 0
    tuples = []

    for i in range(n):
        while times[i] - times[window_start] > windowMillis:
            window_start += 1
        for j in range(window_start, i):
            if raters[i] != raters[j]:
                leftRater, rightRater = tuple(sorted((raters[i], raters[j])))
                tuples.append((leftRater, rightRater, priorTweet))
    return tuples



def _tuples_to_df(tuples, name="pairRatings"):
  leftRater, rightRater, tweetId = zip(*tuples)
  df = pd.DataFrame(
    {
      "leftRaterId": np.array(leftRater),
      "rightRaterId": np.array(rightRater),
      "tweetId": np.array(tweetId),
    }
  )
  print(len(df))
  df = df.drop_duplicates()
  print(len(df))
  df = (
    df.groupby(["leftRaterId", "rightRaterId"])
    .count()
    .reset_index(drop=False)
    .rename(columns={"tweetId": name})
  )
  print(len(df))
  return df


def _get_pair_counts(ratings: pd.DataFrame, windowMillis: int = 1000 * 60 * 30) -> Dict:
  """
  Compute counts of unique posts that were co-rated within windowMillis millis of each other
  by different users.

  Returns dict: (raterId1, raterId2) => count.
  """
  with c.time_block("Computing rating pair counts"):
    counts = dict()
    seen = set()
    ratings = ratings.sort_values([c.noteIdKey, c.createdAtMillisKey])
    values = ratings[
      [c.noteIdKey, c.createdAtMillisKey, c.raterParticipantIdKey, c.tweetIdKey]
    ].values
    logger.info(len(values))
    for i in range(len(values)):
      priorNote, priorTs, priorRater, priorTweet = values[i]
      if i == 0 or i == 1000 or i == 100000 or i % 5000000 == 0:
        logger.info(f"get_pair_counts i={i}")
      j = i + 1
      while j < len(values):
        nextNote, nextTs, nextRater, nextTweet = values[j]
        assert priorNote <= nextNote, (priorNote, nextNote)
        if nextNote != priorNote:
          break  # break if we're onto a new note
        assert priorTweet == nextTweet, (priorTweet, nextTweet)  # tweet should be same
        assert priorRater != nextRater, (priorRater, nextRater)  # rater should be different
        assert priorTs <= nextTs, (priorTs, nextTs)
        if nextTs > (priorTs + windowMillis):
          break  # break if we're beyond windowMillis
        raterPairKey = tuple(sorted((priorRater, nextRater)))
        raterTweetPairKey = (raterPairKey, priorTweet)
        if raterTweetPairKey in seen:
          break  # break if we already counted a match on this tweet
        seen.add(raterTweetPairKey)
        if raterPairKey not in counts:
          counts[raterPairKey] = 0
        counts[raterPairKey] += 1
        j += 1
    return counts
