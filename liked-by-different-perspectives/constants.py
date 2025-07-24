import numpy as np


boostIdKey = "boostId"
userIdKey = "userId"
likeTagsKey = "likeTags"
dislikeTagsKey = "dislikeTags"
negativeTimestampKey = "negativeTimestamp"
contributorTypeKey = "contributorType"
postIdKey = "postId"
updatedAtKey = "updatedAt"
createdAtMillisKey = "createdAt"

actionTweetIdKey = "actionTweetId"

ratingColumnsAndTypes = [
  (boostIdKey, np.int64),
  (userIdKey, np.int64),
  (createdAtMillisKey, np.int64),
  (negativeTimestampKey, np.int64),
  (contributorTypeKey, object),
  (likeTagsKey, object),
  (dislikeTagsKey, object),
  (postIdKey, np.int64),
  (updatedAtKey, np.float64),  # nullable
]

ratingColumns = [col for (col, dtype) in ratingColumnsAndTypes]
ratingTypes = [dtype for (col, dtype) in ratingColumnsAndTypes]
ratingTypeMapping = {col: dtype for (col, dtype) in ratingColumnsAndTypes}

postDataColumnsAndTypes = [
  (actionTweetIdKey, np.int64),
  ("L_pos", np.int64),
  ("M_pos", np.int64),
  ("R_pos", np.int64),
  ("L_neg", np.int64),
  ("M_neg", np.int64),
  ("R_neg", np.int64),
  ("linger_L", np.int64),
  ("linger_M", np.int64),
  ("linger_R", np.int64),
  (boostIdKey, np.int64),
  ("eligibleForActiveState", np.int64),
  ("minPos", np.int64),
  ("maxPos", np.int64),
  ("maxNeg", np.int64),
  ("minLingers", np.int64),
  ("totalPos", np.int64),
  ("active", bool),
  ("scoredAtMillis", np.int64),
]

postDataColumns = [col for (col, dtype) in postDataColumnsAndTypes]
postDataTypes = [dtype for (col, dtype) in postDataColumnsAndTypes]
postDataTypeMapping = {col: dtype for (col, dtype) in postDataColumnsAndTypes}
