import numpy as np


# Note Status Requirements
minRatingsNeeded = 5
maxHistoricalValidRatings = 5
crhThreshold = 0.40
crnhThresholdIntercept = -0.05
crnhThresholdNoteFactorMultiplier = -0.8

# Explanation Tags
minRatingsToGetTag = 2
minTagsNeededForStatus = 2

# Helpfulness Score Thresholds
minMeanNoteScore = 0.05
minCRHVsCRNHRatio = 0.00
minRaterAgreeRatio = 0.66

# Matrix factorization
l2_lambda = 0.03
l2_intercept_multiplier = 5
numFactors = 1
epochs = 200
useGlobalIntercept = True

# Data Filters
minNumRatingsPerRater = 10
minNumRatersPerNote = 5

# Data Filenames
scoredNotesOutputPath = "scoredNotes.tsv"
notesInputPath = "notes-00000.tsv"
ratingsInputPath = "ratings-00000.tsv"
noteStatusHistoryInputPath = "noteStatusHistory-00000.tsv"

# TSV Column Names
participantIdKey = "participantId"
helpfulKey = "helpful"
notHelpfulKey = "notHelpful"
helpfulnessLevelKey = "helpfulnessLevel"
createdAtMillisKey = "createdAtMillis"
summaryKey = "summary"

# TSV Values
notHelpfulValueTsv = "NOT_HELPFUL"
somewhatHelpfulValueTsv = "SOMEWHAT_HELPFUL"
helpfulValueTsv = "HELPFUL"
notesSaysTweetIsMisleadingKey = "MISINFORMED_OR_POTENTIALLY_MISLEADING"
noteSaysTweetIsNotMisleadingKey = "NOT_MISLEADING"

# Fields Transformed From the Raw Data
helpfulNumKey = "helpfulNum"
ratingCreatedBeforeMostRecentNMRLabelKey = "ratingCreatedBeforeMostRecentNMRLabel"
ratingCreatedBeforePublicTSVReleasedKey = "ratingCreatedBeforePublicTSVReleased"

# Timestamps
deletedNoteTombstonesLaunchTime = 1652918400000  # May 19, 2022 UTC
publicTSVTimeDelay = 172800000  # 48 hours

# Explanation Tags
ratingStatusKey = "ratingStatus"
tagCountsKey = "tagCounts"
tiebreakOrderKey = "tiebreakOrder"
firstTagKey = "firstTag"
secondTagKey = "secondTag"

# Model Output
noteInterceptKey = "noteIntercept"
raterInterceptKey = "raterIntercept"
noteFactor1Key = "noteFactor1"
raterFactor1Key = "raterFactor1"

# Ids and Indexes
noteIdKey = "noteId"
tweetIdKey = "tweetId"
classificationKey = "classification"
noteAuthorParticipantIdKey = "noteAuthorParticipantId"
raterParticipantIdKey = "raterParticipantId"
noteIndexKey = "noteIndex"
raterIndexKey = "raterIndex"

# Aggregations
noteCountKey = "noteCount"
ratingCountKey = "ratingCount"
numRatingsKey = "numRatings"

# Helpfulness Score Keys
crhRatioKey = "CRHRatio"
crnhRatioKey = "CRNHRatio"
crhCrnhRatioDifferenceKey = "crhCrnhRatioDifference"
meanNoteScoreKey = "meanNoteScore"
raterAgreeRatioKey = "raterAgreeRatio"
ratingAgreesWithNoteStatusKey = "ratingAgreesWithNoteStatus"

# Note Status Labels
currentlyRatedHelpful = "CURRENTLY_RATED_HELPFUL"
currentlyRatedNotHelpful = "CURRENTLY_RATED_NOT_HELPFUL"
needsMoreRatings = "NEEDS_MORE_RATINGS"

# Boolean Note Status Labels
currentlyRatedHelpfulBoolKey = "crhBool"
currentlyRatedNotHelpfulBoolKey = "crnhBool"


helpfulTagsAndTieBreakOrder = [
  (0, "helpfulOther"),
  (8, "helpfulInformative"),
  (7, "helpfulClear"),
  (3, "helpfulEmpathetic"),
  (4, "helpfulGoodSources"),
  (2, "helpfulUniqueContext"),
  (5, "helpfulAddressesClaim"),
  (6, "helpfulImportantContext"),
  (1, "helpfulUnbiasedLanguage"),
]
helpfulTagsTSVOrder = [tag for (tiebreakOrder, tag) in helpfulTagsAndTieBreakOrder]
helpfulTagsAndTypesTSVOrder = [(tag, np.int64) for tag in helpfulTagsTSVOrder]
helpfulTagsTiebreakOrder = [tag for (tiebreakOrder, tag) in sorted(helpfulTagsAndTieBreakOrder)]

notHelpfulTagsAndTieBreakOrder = [
  (0, "notHelpfulOther"),
  (8, "notHelpfulIncorrect"),
  (2, "notHelpfulSourcesMissingOrUnreliable"),
  (4, "notHelpfulOpinionSpeculationOrBias"),
  (5, "notHelpfulMissingKeyPoints"),
  (12, "notHelpfulOutdated"),
  (10, "notHelpfulHardToUnderstand"),
  (7, "notHelpfulArgumentativeOrBiased"),
  (9, "notHelpfulOffTopic"),
  (11, "notHelpfulSpamHarassmentOrAbuse"),
  (1, "notHelpfulIrrelevantSources"),
  (3, "notHelpfulOpinionSpeculation"),
  (6, "notHelpfulNoteNotNeeded"),
]
notHelpfulTagsTSVOrder = [tag for (tiebreakOrder, tag) in notHelpfulTagsAndTieBreakOrder]
notHelpfulTagsAndTypesTSVOrder = [(tag, np.int64) for tag in notHelpfulTagsTSVOrder]
notHelpfulTagsTiebreakOrder = [
  tag for (tiebreakOrder, tag) in sorted(notHelpfulTagsAndTieBreakOrder)
]

misleadingTags = [
  "misleadingOther",
  "misleadingFactualError",
  "misleadingManipulatedMedia",
  "misleadingOutdatedInformation",
  "misleadingMissingImportantContext",
  "misleadingUnverifiedClaimAsFact",
  "misleadingSatire",
]
misleadingTagsAndTypes = [(tag, np.int64) for tag in misleadingTags]

notMisleadingTags = [
  "notMisleadingOther",
  "notMisleadingFactuallyCorrect",
  "notMisleadingOutdatedButNotWhenWritten",
  "notMisleadingClearlySatire",
  "notMisleadingPersonalOpinion",
]
notMisleadingTagsAndTypes = [(tag, np.int64) for tag in notMisleadingTags]


noteTSVColumnsAndTypes = (
  [
    (noteIdKey, np.int64),
    (participantIdKey, np.object),
    (createdAtMillisKey, np.int64),
    (tweetIdKey, np.int64),
    (classificationKey, np.object),
    ("believable", np.object),
    ("harmful", np.object),
    ("validationDifficulty", np.object),
  ]
  + misleadingTagsAndTypes
  + notMisleadingTagsAndTypes
  + [
    ("trustworthySources", np.int64),
    ("summary", np.object),
  ]
)
noteTSVColumns = [col for (col, dtype) in noteTSVColumnsAndTypes]
noteTSVTypes = [dtype for (col, dtype) in noteTSVColumnsAndTypes]
noteTSVTypeMapping = {col: dtype for (col, dtype) in noteTSVColumnsAndTypes}

ratingTSVColumnsAndTypes = (
  [
    (noteIdKey, np.int64),
    (participantIdKey, np.object),
    (createdAtMillisKey, np.int64),
    ("version", np.int64),
    ("agree", np.int64),
    ("disagree", np.int64),
    (helpfulKey, np.int64),
    (notHelpfulKey, np.int64),
    (helpfulnessLevelKey, np.object),
  ]
  + helpfulTagsAndTypesTSVOrder
  + notHelpfulTagsAndTypesTSVOrder
)

ratingTSVColumns = [col for (col, dtype) in ratingTSVColumnsAndTypes]
ratingTSVTypes = [dtype for (col, dtype) in ratingTSVColumnsAndTypes]
ratingTSVTypeMapping = {col: dtype for (col, dtype) in ratingTSVColumnsAndTypes}


timestampMillisOfNoteFirstNonNMRLabelKey = "timestampMillisOfFirstNonNMRStatus"
firstNonNMRLabelKey = "firstNonNMRStatus"
timestampMillisOfNoteCurrentLabelKey = "timestampMillisOfCurrentStatus"
currentLabelKey = "currentStatus"
timestampMillisOfNoteMostRecentNonNMRLabelKey = "timestampMillisOfLatestNonNMRStatus"
mostRecentNonNMRLabelKey = "mostRecentNonNMRStatus"

noteStatusHistoryTSVColumnsAndTypes = [
  (noteIdKey, np.int64),
  (participantIdKey, np.object),
  (createdAtMillisKey, np.int64),
  (timestampMillisOfNoteFirstNonNMRLabelKey, np.float),  # float because nullable.
  (firstNonNMRLabelKey, np.object),
  (timestampMillisOfNoteCurrentLabelKey, np.float),  # float because nullable.
  (currentLabelKey, np.object),
  (timestampMillisOfNoteMostRecentNonNMRLabelKey, np.float),  # float because nullable.
  (mostRecentNonNMRLabelKey, np.object),
]
noteStatusHistoryTSVColumns = [col for (col, dtype) in noteStatusHistoryTSVColumnsAndTypes]
noteStatusHistoryTSVTypes = [dtype for (col, dtype) in noteStatusHistoryTSVColumnsAndTypes]
noteStatusHistoryTSVTypeMapping = {
  col: dtype for (col, dtype) in noteStatusHistoryTSVColumnsAndTypes
}

scoredNotesColumns = (
  [noteIdKey, helpfulNumKey]
  + helpfulTagsTSVOrder
  + notHelpfulTagsTSVOrder
  + [numRatingsKey, noteInterceptKey, noteFactor1Key, ratingStatusKey, firstTagKey, secondTagKey]
)
