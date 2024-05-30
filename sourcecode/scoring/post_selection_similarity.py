from typing import Dict

from . import constants as c

import numpy as np
import pandas as pd


class PostSelectionSimilarity:
  def __init__(
    self,
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
    pmiRegularization: int = 500,
    smoothedNpmiThreshold: float = 0.60,
  ):
    self.ratings = _preprocess_ratings(notes, ratings)
    self.pairCounts = _get_pair_counts(self.ratings)
    self.pairStatsDf = _make_rater_stats_df(self.pairCounts)
    self.uniqueRatingsOnTweets = self.ratings[
      [c.tweetIdKey, c.raterParticipantIdKey]
    ].drop_duplicates()
    self.pairStatsDf = _join_rater_totals(self.pairStatsDf, self.uniqueRatingsOnTweets)
    self.pmiDf = _compute_pmi(self.pairStatsDf, len(self.uniqueRatingsOnTweets), pmiRegularization)

    self.filter_edges_below_threshold(smoothedNpmiThreshold)

  def filter_edges_below_threshold(self, smoothedNpmiThreshold: float = 0.6):
    graphDf = self.pmiDf[["leftRaterId", "rightRaterId", "smoothedNpmi"]]
    graphDf.columns = ["source", "target", "weight"]
    self.graphDf = graphDf[graphDf["weight"] >= smoothedNpmiThreshold]

  def get_high_post_selection_similarity_raters(self):
    highPostSelectionSimilarityRaters = pd.concat(
      [
        self.graphDf[["source"]].rename(columns={"source": c.raterParticipantIdKey}),
        self.graphDf[["target"]].rename(columns={"target": c.raterParticipantIdKey}),
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


def filter_ratings_by_post_selection_similarity(ratings, highPostSelectionSimilarityRaters):
  """
  Filters out ratings from raters who have high post selection similarity.
  """
  ratings = ratings.merge(
    highPostSelectionSimilarityRaters, on=c.raterParticipantIdKey, how="left", indicator=True
  )
  ratings = ratings[ratings["_merge"] == "left_only"]
  ratings = ratings.drop(columns=["_merge"])
  return ratings


def _compute_pmi(pairStatsDf: pd.DataFrame, N: int, pmiPseudocounts: int = 500) -> pd.DataFrame:
  """
  Compute PMI between raters.
  """
  numerator = pairStatsDf["pairRatings"] * N
  denominator = (pairStatsDf["leftTotal"] + pmiPseudocounts) * (
    pairStatsDf["rightTotal"] + pmiPseudocounts
  )
  pairStatsDf["smoothedPmi"] = np.log(numerator / denominator)
  pairStatsDf["smoothedNpmi"] = pairStatsDf["smoothedPmi"] / -np.log(pairStatsDf["pairRatings"] / N)
  pairStatsDf["minSim"] = pairStatsDf["pairRatings"] / np.minimum(
    pairStatsDf["leftTotal"], pairStatsDf["rightTotal"]
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
      sid = row["source"]
      tid = row["target"]
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


def _get_pair_counts(ratings: pd.DataFrame, windowMillis: int = 1000 * 60 * 10) -> Dict:
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
    print(len(values))
    for i in range(len(values)):
      priorNote, priorTs, priorRater, priorTweet = values[i]
      if i == 0 or i == 1000 or i == 100000 or i % 5000000 == 0:
        print(f"get_pair_counts i={i}")
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
