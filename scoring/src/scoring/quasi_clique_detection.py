# Std libraries
import logging
from typing import Set, Tuple

# Project libraries
from . import constants as c

# 3rd-party libraries
import pandas as pd


logger = logging.getLogger("birdwatch.quasi_clique_detection")
logger.setLevel(logging.INFO)


class QuasiCliqueDetection:
  def __init__(
    self,
    recencyCutoff: int = (1000 * 60 * 60 * 24 * 7 * 13),
    noteInclusionThreshold: float = 0.25,
    raterInclusionThreshold: float = 0.25,
    minCliqueTweets: int = 25,
    minCliqueRaters: int = 5,
    maxCliqueSize: int = 2000,
    minInclusionRatings: int = 4,
    minRaterPairCount: int = 50,
  ):
    """Initialize QuasiCliqueDetection.

    Args:
      recencyCutoff:
      noteInclusionThreshold: At least noteInclusionThreshold must rate a given note in the same way for the note to be included
      raterInclusionThreshold: Each rater must rate at least this fraction of posts in the common way for the rater to be included
      minCliqueTweets: Each quasi-clique must have at least this many posts
      minCliqueRaters: Each clique must have at least this many raters.
      maxCliqueSize: Each quasi-clique can have at most this many raters
      minInclusionRatings: In addition to meeting the noteInclusionThreshold, included notes must get at
        least this many ratings from included raters (or be rated by all included raters, whichever is
        fewer)
      minRaterPairCount: Raters must have at least this many matching ratings (roughly 1/day) to be considered
    """
    self._recencyCutoff = recencyCutoff
    self._minCliqueTweets = minCliqueTweets
    self._minCliqueRaters = minCliqueRaters
    self._maxCliqueSize = maxCliqueSize
    self._noteInclusionThreshold = noteInclusionThreshold
    self._raterInclusionThreshold = raterInclusionThreshold
    self._minInclusionRatings = minInclusionRatings
    self._minRaterPairCount = minRaterPairCount

  def _get_pair_counts(
    self,
    ratings: pd.DataFrame,
    notes: pd.DataFrame,
    cutoff: int,
    minAlignedRatings: int = 5,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return counts of how many times raters rate notes in the same way, and all ratings for raters who do so >5 times."""
    # Identify ratings that are in scope
    logger.info("initial rating length:", len(ratings))
    ratings = ratings[[c.noteIdKey, c.raterParticipantIdKey, c.helpfulNumKey]]
    ratings = ratings.merge(
      notes[[c.noteIdKey, c.tweetIdKey, c.classificationKey, c.createdAtMillisKey]].rename(
        columns={c.createdAtMillisKey: "noteMillis"}
      )
    )
    logger.info("ratings after merges:", len(ratings))
    ratings = ratings[ratings["noteMillis"] > (ratings["noteMillis"].max() - cutoff)]
    logger.info("recent ratings:", len(ratings))
    ratings = ratings[ratings[c.tweetIdKey] != "-1"]
    logger.info("ratings on non-deleted tweets:", len(ratings))
    ratings = ratings[ratings[c.classificationKey] == c.notesSaysTweetIsMisleadingKey]
    logger.info("ratings on misleading posts:", len(ratings))
    # Identify pairs
    ratings = ratings[[c.tweetIdKey, c.noteIdKey, c.helpfulNumKey, c.raterParticipantIdKey]]
    noteCollisions = (
      ratings.groupby([c.tweetIdKey, c.noteIdKey, c.helpfulNumKey])
      .agg(list)
      .reset_index(drop=False)
    )
    tweetNoteCollisions = (
      noteCollisions[[c.tweetIdKey, c.raterParticipantIdKey]]
      .groupby(c.tweetIdKey)
      .agg(list)
      .reset_index(drop=False)
    )
    raterPairCounts = dict()
    logger.info(f"Preparing pairs for {len(tweetNoteCollisions)} tweets")
    for idx, (_, tweetNoteRaters) in enumerate(tweetNoteCollisions.values):
      if idx % 1000 == 0:
        logger.info(idx)
      pairs = set()
      for noteRaters in tweetNoteRaters:
        noteRaters = sorted(noteRaters)
        for i in range(len(noteRaters)):
          for j in range(i + 1, len(noteRaters)):
            pair = (noteRaters[i], noteRaters[j])
            assert pair[0] < pair[1]
            if pair in pairs:
              continue
            pairs.add(pair)
            if pair not in raterPairCounts:
              raterPairCounts[pair] = 0
            raterPairCounts[pair] += 1
    # Return dataframes
    left, right, count = zip(
      *[
        (leftRater, rightRater, count)
        for ((leftRater, rightRater), count) in raterPairCounts.items()
        if count >= minAlignedRatings
      ]
    )
    counts = pd.DataFrame({"left": left, "right": right, "count": count})
    ratings = ratings.merge(pd.DataFrame({c.raterParticipantIdKey: list(set(left + right))}))
    logger.info(f"ratings after filter to raters included in pair counts: {len(ratings)}")
    return counts, ratings

  def _grow_clique(
    self,
    includedRaters: Set[str],
    raterRatings: pd.DataFrame,
  ):
    """Grow a clique from an initial set of raters.  Return all included raters and tweets.

    Given a clique seed containing two raters, greedily grow the clique one rater at a time
    by adding the excluded rater that has the most ratings in common with the actions of the
    clique.  Before each addition, verify that adding the rater will not violate the minimum
    density requirements of the clique.

    Args:
      includedRaters: Set of raters to use to initialize a clique
      raterRatings: DF containing ratings from all raters with 5 or more rating collisions

    Returns:
      Set of raters and tweets that meet density criteria.
    """
    # Expand cliques 1 rater at a time, checking to see if density thresholds are satisifed after expansion
    for _ in range(self._maxCliqueSize):
      # Identify the ratings where there is enough agreement that the rating constitutes a group action
      ratings = raterRatings[raterRatings[c.raterParticipantIdKey].isin(includedRaters)]
      groupRatingCounts = (
        ratings[[c.tweetIdKey, c.noteIdKey, c.helpfulNumKey]].value_counts().reset_index(drop=False)
      )
      groupRatingCounts = groupRatingCounts[
        groupRatingCounts["count"] >= (len(includedRaters) * self._noteInclusionThreshold)
      ]
      groupRatingCounts = groupRatingCounts[
        groupRatingCounts["count"] >= min(len(includedRaters), self._minInclusionRatings)
      ]
      # Find the rater not in the group that most aligned with the group rating events
      alignedRatings = raterRatings.merge(
        groupRatingCounts[[c.tweetIdKey, c.noteIdKey, c.helpfulNumKey]]
      )
      alignedRatings = alignedRatings[~alignedRatings[c.raterParticipantIdKey].isin(includedRaters)]
      alignedTweetPerRater = (
        alignedRatings[[c.tweetIdKey, c.raterParticipantIdKey]]
        .drop_duplicates()[c.raterParticipantIdKey]
        .value_counts()
        .reset_index(drop=False)
      )
      candidate = (
        alignedTweetPerRater.sort_values("count", ascending=False)
        .head(1)[c.raterParticipantIdKey]
        .item()
      )
      # Preserve current set of tweets incase we decide not to add the candidate
      includedTweets = set(groupRatingCounts[c.tweetIdKey].drop_duplicates())
      # Calculate how many tweets would meet the inclusion threshold if the candidate were added
      candidateRaters = includedRaters | {candidate}
      ratings = raterRatings[raterRatings[c.raterParticipantIdKey].isin(candidateRaters)]
      groupRatingCounts = (
        ratings[[c.tweetIdKey, c.noteIdKey, c.helpfulNumKey]].value_counts().reset_index(drop=False)
      )
      groupRatingCounts = groupRatingCounts[
        groupRatingCounts["count"] >= (len(candidateRaters) * self._noteInclusionThreshold)
      ]
      groupRatingCounts = groupRatingCounts[
        groupRatingCounts["count"] >= min(len(candidateRaters), self._minInclusionRatings)
      ]
      satisfiedTweets = groupRatingCounts[c.tweetIdKey].nunique()
      # Calculate how many raters would be below the inclusion threshold if we add the candidate
      matchingRatings = ratings.merge(
        groupRatingCounts[[c.tweetIdKey, c.noteIdKey, c.helpfulNumKey]]
      )
      raterCounts = (
        matchingRatings[[c.raterParticipantIdKey, c.tweetIdKey]]
        .drop_duplicates()
        .value_counts(c.raterParticipantIdKey)
        .reset_index(drop=False)
      )
      ratersBelowThreshold = (
        raterCounts["count"] < (self._raterInclusionThreshold * satisfiedTweets)
      ).sum()
      # Check standards
      if satisfiedTweets >= self._minCliqueTweets and ratersBelowThreshold == 0:
        includedRaters = candidateRaters
      else:
        return includedRaters, includedTweets
    return includedRaters, includedTweets

  def _build_clusters(
    self,
    raterPairCounts: pd.DataFrame,
    raterPairRatings: pd.DataFrame,
  ):
    """Identify disjoint quasi-cliques using a greedy clustering approach.

    Args:
      raterPairCounts: DF containing counts of how often pairs of raters colide.
      raterPairRatings: All ratings from raters with >5 collisions with another rater
    """
    cliques = []
    # Attempt to cluster every rater with at least minRaterPairCount collisions
    logger.info(f"orig raterPairCounts: {len(raterPairCounts)}")
    raterPairCounts = raterPairCounts[raterPairCounts["count"] > self._minRaterPairCount]
    logger.info(f"pruned raterPairCounts: {len(raterPairCounts)}")
    # Build cliques
    while len(raterPairCounts) > 0:
      # Identify seed
      raterPairCounts = raterPairCounts.sort_values("count", ascending=False)
      leftRater, rightRater = raterPairCounts.head(1)[["left", "right"]].values.flatten()
      # Build clique and prune candidate set
      cliqueRaters, cliquePosts = self._grow_clique({leftRater, rightRater}, raterPairRatings)
      raterPairCounts = raterPairCounts[
        ~(
          (raterPairCounts["left"].isin(cliqueRaters))
          | (raterPairCounts["right"].isin(cliqueRaters))
        )
      ]
      # Augment results if clique is large enough
      if len(cliqueRaters) >= self._minCliqueRaters:
        logger.info(
          f"Adding clique  (raters={len(cliqueRaters)}, tweets={len(cliquePosts)}).  Remaining ratePairCounts: {len(raterPairCounts)}"
        )
        cliques.append((cliqueRaters, cliquePosts))
      else:
        logger.info(
          f"Skipping clique  (raters={len(cliqueRaters)}, tweets={len(cliquePosts)}).  Remaining ratePairCounts: {len(raterPairCounts)}"
        )
    # Order from largest to smallest and return
    cliques.sort(key=lambda clique: len(clique[0]))
    return cliques

  def get_quasi_cliques(
    self,
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
  ) -> pd.DataFrame:
    """Obtain quasi-cliques in the rating graph.

    Each clique is defined by a list of raters and posts, and must meet minimum
    size and density requirements with respect to the number of raters, number of
    posts and minium density of ratings connecting the raters and posts.
    """
    # Obtain quasi-cliques
    raterPairCounts, raterPairRatings = self._get_pair_counts(ratings, notes, self._recencyCutoff)
    quasiCliques = self._build_clusters(raterPairCounts, raterPairRatings)
    # Convert to data frame
    cliqueIds = []
    raterIds = []
    for i, (raters, _) in enumerate(quasiCliques):
      for rater in raters:
        cliqueIds.append((i + 1))  # To align with PSS, by convention clique IDs begin at 1
        raterIds.append(rater)
    return pd.DataFrame(
      {
        c.raterParticipantIdKey: raterIds,
        c.quasiCliqueValueKey: cliqueIds,
      }
    )
