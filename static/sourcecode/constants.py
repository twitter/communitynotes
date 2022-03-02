# Note Status Requirements
minRatingsNeeded = 5
crhThreshold = 0.43
crnhThreshold = 0.00

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

# TSV Column Names
participantIdKey = "participantId"
helpfulKey = "helpful"
notHelpfulKey = "notHelpful"
helpfulnessLevelKey = "helpfulnessLevel"
createdAtMillisKey = "createdAtMillis"

# TSV Values
notHelpfulValueTsv = "NOT_HELPFUL"
somewhatHelpfulValueTsv = "SOMEWHAT_HELPFUL"
helpfulValueTsv = "HELPFUL"

# Fields Transformed From the Raw Data
helpfulNumKey = "helpfulNum"

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
currentlyRatedHelpful = "Currently Rated Helpful"
currentlyRatedNotHelpful = "Currently Rated Not Helpful"
needsMoreRatings = "Needs More Ratings"

# Boolean Note Status Labels
currentlyRatedHelpfulBoolKey = "crhBool"
currentlyRatedNotHelpfulBoolKey = "crnhBool"