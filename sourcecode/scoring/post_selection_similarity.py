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

  def get_high_post_selection_similarity_raters(self):
    uniqueRaters = set()
    for r1, r2 in self.pairCountsDict.keys():
      uniqueRaters.add(r1)
      uniqueRaters.add(r2)
    highPostSelectionSimilarityRaters = pd.DataFrame(
      list(uniqueRaters), columns=[c.raterParticipantIdKey]
    )
    highPostSelectionSimilarityRaters[c.postSelectionValueKey] = 1
    return highPostSelectionSimilarityRaters

  def get_post_selection_similarity_values(self):
    """
    Returns dataframe with [raterParticipantId, postSelectionSimilarityValue] columns.
    postSelectionSimilarityValue is None by default.
    """
    cliqueToUserMap, userToCliqueMap = aggregate_into_cliques(self.pairCountsDict)

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

  with c.time_block("Compute PMI and minSim"):
    for leftRaterId, rightRaterId in pairCountsDict:
      if leftRaterId not in raterTotalsDict or rightRaterId not in raterTotalsDict:
        keys_to_delete.append((leftRaterId, rightRaterId))
        continue

      leftTotal = raterTotalsDict[leftRaterId]
      rightTotal = raterTotalsDict[rightRaterId]
      coRatings = pairCountsDict[(leftRaterId, rightRaterId)]

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
        pairCountsDict[(leftRaterId, rightRaterId)] = (smoothedNpmi, minSimRatingProp)
      else:
        keys_to_delete.append((leftRaterId, rightRaterId))

    print(f"Pairs dict used {sys.getsizeof(pairCountsDict) * 1e-9}GB RAM at max")

  with c.time_block("Delete unneeded pairs from pairCountsDict"):
    for key in keys_to_delete:
      del pairCountsDict[key]

  print(
    f"Pairs dict used {sys.getsizeof(pairCountsDict) * 1e-9}GB RAM after deleted unneeded pairs"
  )

  return pairCountsDict


def aggregate_into_cliques(pairCountsDict):
  with c.time_block("Aggregate into cliques by post selection similarity"):
    userToCliqueMap = dict()
    cliqueToUserMap = dict()

    nextNewCliqueId = 1  # start cliqueIdxs from 1

    for sid, tid in pairCountsDict.keys():
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


def _get_pair_counts_dict(ratings, windowMillis):
  pair_counts = dict()

  # Group by tweetIdKey to process each tweet individually
  grouped_by_tweet = ratings.groupby(c.tweetIdKey, sort=False)

  for _, tweet_group in grouped_by_tweet:
    # Keep track of pairs we've already counted for this tweetId
    pairs_counted_in_tweet = set()

    # Group by noteIdKey within the tweet
    grouped_by_note = tweet_group.groupby(c.noteIdKey, sort=False)

    for _, note_group in grouped_by_note:
      note_group.sort_values(c.createdAtMillisKey, inplace=True)

      # Extract relevant columns as numpy arrays for efficient computation
      times = note_group[c.createdAtMillisKey].values
      raters = note_group[c.raterParticipantIdKey].values

      n = len(note_group)
      window_start = 0

      for i in range(n):
        # Move the window start forward if the time difference exceeds windowMillis
        while times[i] - times[window_start] > windowMillis:
          window_start += 1

        # For all indices within the sliding window (excluding the current index)
        for j in range(window_start, i):
          if raters[i] != raters[j]:
            left_rater, right_rater = tuple(sorted((raters[i], raters[j])))
            pair = (left_rater, right_rater)
            # Only count this pair once per tweetId
            if pair not in pairs_counted_in_tweet:
              pairs_counted_in_tweet.add(pair)
              # Update the count for this pair
              if pair not in pair_counts:
                pair_counts[pair] = 0
              pair_counts[pair] += 1

  return pair_counts
