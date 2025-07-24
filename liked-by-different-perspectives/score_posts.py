import logging

import public.scoring.constants as c

from . import constants as cbc


logger = logging.getLogger("birdwatch")
logger.setLevel(logging.INFO)

import pandas as pd


FACTOR_SPLIT_POINT = 0.15
MIN_LINGERS = 29
QUANTILE_THRESHOLD = 0.7


def infer_rater_factors(
  users: pd.DataFrame,
  noteRatings: pd.DataFrame,
  scoredNotes: pd.DataFrame,
  raterModelOutput: pd.DataFrame,
) -> pd.DataFrame:
  logger.info(f"{users.loc[users['isAdmitted']==True].shape[0]} admitted users")
  ratingsFromCbUsers = noteRatings[
    [c.noteIdKey, c.raterParticipantIdKey, c.helpfulnessLevelKey]
  ].merge(
    users.loc[users["isAdmitted"] == True][[cbc.userIdKey]],
    left_on=c.raterParticipantIdKey,
    right_on=cbc.userIdKey,
  )
  logger.info(f"{ratingsFromCbUsers.shape[0]} ratings from cb users")
  ratingsFromCbUsers = ratingsFromCbUsers.loc[~(ratingsFromCbUsers[c.helpfulnessLevelKey].isna())]
  logger.info(f"{ratingsFromCbUsers.shape[0]} ratings from cb users with non nan helpfulnessLevel")
  # merge in note factors and note classification, limit to only misleading and scored by core
  factorNotes = scoredNotes.loc[
    (scoredNotes[c.classificationKey] == c.notesSaysTweetIsMisleadingKey)
    & (scoredNotes[c.noteTopicKey].isna())
    & scoredNotes[c.decidedByKey].isin(
      ["CoreModel (v1.1)", "ExpansionModel (v1.1)", "ExpansionPlusModel (v1.1)"]
    )
  ]
  logger.info(f"{factorNotes.shape[0]} eligible notes for inferring factor")

  # for now use core factor only
  factorNotes["factor"] = factorNotes[c.coreNoteFactor1Key]
  ratingsFromCbUsers = ratingsFromCbUsers.merge(
    factorNotes[[c.noteIdKey, "factor"]], on=c.noteIdKey
  )
  logger.info(f"{ratingsFromCbUsers.shape[0]} ratings on valid notes")
  ratingsFromCbUsers["signCorrectedFactor"] = ratingsFromCbUsers.apply(
    lambda row: -row["factor"]
    if (row[c.helpfulnessLevelKey] == c.notHelpfulValueTsv)
    else row["factor"],
    axis=1,
  )
  inferredFactors = (
    ratingsFromCbUsers[[cbc.userIdKey, "signCorrectedFactor"]]
    .groupby(cbc.userIdKey)
    .agg("mean")
    .reset_index()
  )
  logger.info(f"{inferredFactors.shape[0]} inferred factors")
  # c.raterParticipantIdKey is an object in CN output
  inferredFactors[cbc.userIdKey + "_str"] = inferredFactors[cbc.userIdKey].astype(str)
  inferredFactors = inferredFactors.merge(
    raterModelOutput[[c.raterParticipantIdKey, c.coreRaterFactor1Key]],
    left_on=cbc.userIdKey + "_str",
    right_on=c.raterParticipantIdKey,
    how="left",
  )
  inferredFactors = inferredFactors.drop(columns=[cbc.userIdKey + "_str"])
  inferredFactors["finalFactor"] = inferredFactors["signCorrectedFactor"].combine_first(
    inferredFactors[c.coreRaterFactor1Key]
  )
  finalRaterFactors = inferredFactors[[cbc.userIdKey, "finalFactor"]]
  assert (
    len(finalRaterFactors[cbc.userIdKey].unique()) == finalRaterFactors.shape[0]
  ), "nonunique rater factors"
  return finalRaterFactors


def score_posts(
  users: pd.DataFrame,
  noteRatings: pd.DataFrame,
  scoredNotes: pd.DataFrame,
  raterModelOutput: pd.DataFrame,
  postRatings: pd.DataFrame,
  postActions: pd.DataFrame,
) -> pd.DataFrame:
  logger.info("Inferring rater factors")
  raterFactors = infer_rater_factors(users, noteRatings, scoredNotes, raterModelOutput)
  # calculate which posts meet final criteria
  logger.info("Merging rater factors")
  logger.info(f"{postRatings.shape[0]} latest post ratings")
  postRatingsWFactor = postRatings.merge(raterFactors, on=cbc.userIdKey, how="left")
  assert (
    postRatingsWFactor[[cbc.userIdKey, cbc.postIdKey]].drop_duplicates().shape[0]
    == postRatingsWFactor.shape[0]
  ), "multiple actions per user-post"
  logger.info(f"{postRatingsWFactor.shape[0]} post ratings with factor")
  posActionsWFactor = postRatingsWFactor.loc[
    (
      ~(postRatingsWFactor["likeTags"].astype(str).isin(["[]", "['NOT_APPLICABLE']"]))
      & (postRatingsWFactor["dislikeTags"].astype(str).isin(["[]", "['NOT_APPLICABLE']"]))
    )
  ]
  logger.info(f"{posActionsWFactor.shape[0]} positive rating actions")
  negActionsWFactor = postRatingsWFactor.loc[
    ~(postRatingsWFactor["dislikeTags"].astype(str).isin(["[]", "['NOT_APPLICABLE']"]))
  ]
  logger.info(f"{negActionsWFactor.shape[0]} negative rating actions")
  actionsWFactor = (
    postActions.loc[(postActions["isAdmitted"] == True)][
      [
        cbc.actionTweetIdKey,
        cbc.userIdKey,
        cbc.boostIdKey,
        "eligibleForActiveState",
      ]
    ]
    .drop_duplicates()
    .merge(raterFactors, on=cbc.userIdKey, how="left")
  )
  actionsWFactor = actionsWFactor.loc[~(actionsWFactor["finalFactor"].isna())]
  logger.info(f"{actionsWFactor.shape[0]} linger actions")

  for df in [posActionsWFactor, negActionsWFactor, actionsWFactor]:
    df["bucket_L"] = df["finalFactor"] < -FACTOR_SPLIT_POINT
    df["bucket_M"] = (df["finalFactor"] >= -FACTOR_SPLIT_POINT) & (
      df["finalFactor"] < FACTOR_SPLIT_POINT
    )
    df["bucket_R"] = df["finalFactor"] >= FACTOR_SPLIT_POINT
    df["bucket"] = "None"
    df.loc[df["bucket_L"], "bucket"] = "L"
    df.loc[df["bucket_M"], "bucket"] = "M"
    df.loc[df["bucket_R"], "bucket"] = "R"
    print(df["bucket"].value_counts(), flush=True)

  # eligible is True if the tentative pivot was created within the last week
  eligibility = (
    actionsWFactor[[cbc.actionTweetIdKey, cbc.boostIdKey, "eligibleForActiveState"]]
    .sort_values(cbc.boostIdKey, ascending=False)
    .groupby(cbc.actionTweetIdKey)
    .head(1)
  )
  lingerBucketCounts = (
    actionsWFactor[[cbc.actionTweetIdKey, "bucket", cbc.userIdKey]]
    .groupby([cbc.actionTweetIdKey, "bucket"])
    .agg("count")
    .reset_index()
  )
  lingerBucketCountsFlat = pd.pivot_table(
    lingerBucketCounts,
    columns="bucket",
    values=cbc.userIdKey,
    index=cbc.actionTweetIdKey,
  ).fillna(0)
  logger.info(f"{lingerBucketCountsFlat.shape[0]} lingered posts")
  lingerBucketCountsFlat.columns = ["linger_" + col for col in lingerBucketCountsFlat.columns]
  posActionsWFactor.rename(columns={cbc.postIdKey: cbc.actionTweetIdKey}, inplace=True)
  negActionsWFactor.rename(columns={cbc.postIdKey: cbc.actionTweetIdKey}, inplace=True)
  posBucketCounts = (
    posActionsWFactor[[cbc.actionTweetIdKey, "bucket", cbc.userIdKey]]
    .groupby([cbc.actionTweetIdKey, "bucket"])
    .agg("count")
    .reset_index()
  )
  posBucketCountsFlat = pd.pivot_table(
    posBucketCounts,
    columns="bucket",
    values=cbc.userIdKey,
    index=cbc.actionTweetIdKey,
  ).fillna(0)
  negBucketCounts = (
    negActionsWFactor[[cbc.actionTweetIdKey, "bucket", cbc.userIdKey]]
    .groupby([cbc.actionTweetIdKey, "bucket"])
    .agg("count")
    .reset_index()
  )
  negBucketCountsFlat = pd.pivot_table(
    negBucketCounts,
    columns="bucket",
    values=cbc.userIdKey,
    index=cbc.actionTweetIdKey,
  ).fillna(0)
  allBucketCounts = (
    posBucketCountsFlat.merge(
      negBucketCountsFlat, on=cbc.actionTweetIdKey, suffixes=["_pos", "_neg"], how="outer"
    )
    .merge(lingerBucketCountsFlat, how="outer", on=cbc.actionTweetIdKey)
    .fillna(0)
    .merge(eligibility, on=cbc.actionTweetIdKey, how="left")
  )
  logger.info(f"{allBucketCounts.shape[0]} posts after final calculation")

  allBucketCounts["minPos"] = allBucketCounts[["L_pos", "R_pos"]].min(axis=1)
  allBucketCounts["maxPos"] = allBucketCounts[["L_pos", "R_pos"]].max(axis=1)
  allBucketCounts["maxNeg"] = allBucketCounts[["L_neg", "R_neg"]].max(axis=1)
  allBucketCounts["minLingers"] = allBucketCounts[["linger_L", "linger_R"]].min(axis=1)
  allBucketCounts["totalPos"] = (
    allBucketCounts["L_pos"] + allBucketCounts["R_pos"] + allBucketCounts["M_pos"]
  )
  totalPosCutoff = max(
    [
      1,
      allBucketCounts.loc[
        (allBucketCounts["minLingers"] > MIN_LINGERS)
        & (allBucketCounts["maxNeg"] == 0)
        & (allBucketCounts["eligibleForActiveState"] == 1)
      ][["totalPos"]]
      .quantile(QUANTILE_THRESHOLD)
      .item(),
    ]
  )
  logger.info(f"total pos cutoff {totalPosCutoff}")
  minPosCutoff = max(
    [
      1,
      allBucketCounts.loc[
        (allBucketCounts["minLingers"] > MIN_LINGERS)
        & (allBucketCounts["maxNeg"] == 0)
        & (allBucketCounts["eligibleForActiveState"] == 1)
      ][["minPos"]]
      .quantile(QUANTILE_THRESHOLD)
      .item(),
    ]
  )
  logger.info(f"min pos cutoff {minPosCutoff}")

  activePosts = allBucketCounts.loc[
    (allBucketCounts["minLingers"] > MIN_LINGERS)
    & (allBucketCounts["maxNeg"] == 0)
    & (
      (allBucketCounts["totalPos"] >= totalPosCutoff) | (allBucketCounts["minPos"] >= minPosCutoff)
    )
    & (allBucketCounts["eligibleForActiveState"] == 1)
  ].reset_index()

  allBucketCounts["active"] = False
  allBucketCounts.loc[
    allBucketCounts[cbc.actionTweetIdKey].isin(activePosts[cbc.actionTweetIdKey].unique()),
    "active",
  ] = True

  return allBucketCounts
