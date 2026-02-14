from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import logging
import os
import time
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd


logger = logging.getLogger("birdwatch.constants")
logger.setLevel(logging.INFO)


# Default number of threads to use in torch if os.cpu_count() is unavailable
# and no value is specified.
defaultNumThreads = os.cpu_count() or 8

# Store the timestamp at which the constants module is initialized.  Note
# that module initialization occurs only once regardless of how many times
# the module is imported (see link below).  Storing a designated timestamp
# as a constant allow us to:
#  -Use a consistent notion of "now" throughout scorer execution.
#  -Overwrite "now" when system testing to reduce spurious diffs.
#
# https://docs.python.org/3/tutorial/modules.html#more-on-modules
epochMillis = 1000 * time.time()
useCurrentTimeInsteadOfEpochMillisForNoteStatusHistory = True
# Use this size threshld to isolate code which should be run differently in small
# scale unit tests.
minNumNotesForProdData = 200

# Define limit on how old a note needs to be to lock
noteLockMillis = 14 * 24 * 60 * 60 * 1000


# Explanation Tags
minRatingsToGetTag = 2
minTagsNeededForStatus = 2
tagPercentileForNormalization = 40
intervalHalfWidth = 0.3

# Max flip rates
prescoringAllUnlockedNotesMaxCrhChurn = 0.3
prescoringAllNotesCreatedThreeToThirteenDaysAgoMaxChurn = 0.15
finalUnlockedNotesWithNoNewRatingsMaxCrhChurn = 0.12
finalNotesWithNewRatingsMaxNewCrhChurn = 1.2
finalNotesWithNewRatingsMaxOldCrhChurn = 0.4
finalNotesThatJustFlippedStatusMaxCrhChurn = 1e7
finalNotesThatFlippedRecentlyMaxCrhChurn = 1e7
# TODO(jiansongc): adjust these 2 below
finalNotesNmrDueToMinStableCrhTimeMaxOldCrhChurn = 1.0
finalNotesNmrDueToMinStableCrhTimeMaxNewCrhChurn = 1.0

# Data Filenames
scoredNotesOutputPath = "scoredNotes.tsv"
enrollmentInputPath = "userEnrollment-00000.tsv"
notesInputPath = "notes-00000.tsv"
ratingsInputPath = "ratings"
noteStatusHistoryInputPath = "noteStatusHistory-00000.tsv"

# TSV Column Names
participantIdKey = "participantId"
helpfulKey = "helpful"
notHelpfulKey = "notHelpful"
helpfulnessLevelKey = "helpfulnessLevel"
createdAtMillisKey = "createdAtMillis"
summaryKey = "summary"
noteTopicKey = "noteTopic"
authorTopNotHelpfulTagValues = "authorTopNotHelpfulTagValues"
modelingPopulationKey = "modelingPopulation"
modelingGroupKey = "modelingGroup"
modelingMultiGroupKey = "modelingMultiGroup"
numberOfTimesEarnedOutKey = "numberOfTimesEarnedOut"
defaultIndexKey = "index"
highVolumeRaterKey = "highVolumeRater"
correlatedRaterKey = "correlatedRater"

# Scoring Groups
coreGroups: Set[int] = {1, 2, 3, 6, 8, 9, 10, 11, 13, 14, 19, 21, 25}
coverageGroups: Set[int] = {1, 2, 3, 6, 8, 9, 10, 11, 13, 14, 19, 25}
expansionGroups: Set[int] = {0, 4, 5, 7, 12, 15, 16, 18, 20, 22, 23, 26, 27, 28, 29, 33}
expansionPlusGroups: Set[int] = {17, 24, 30, 31, 32}

# Bins for Gaussian Scorer
quantileRange = np.array(
  [
    -0.92,
    -0.65,
    -0.61,
    -0.58,
    -0.55,
    -0.52,
    -0.49,
    -0.46,
    -0.43,
    -0.41,
    -0.38,
    -0.35,
    -0.33,
    -0.3,
    -0.28,
    -0.25,
    -0.22,
    -0.2,
    -0.17,
    -0.14,
    -0.12,
    -0.09,
    -0.07,
    -0.04,
    -0.01,
    0.01,
    0.04,
    0.07,
    0.09,
    0.12,
    0.14,
    0.17,
    0.2,
    0.22,
    0.25,
    0.28,
    0.3,
    0.33,
    0.35,
    0.38,
    0.41,
    0.43,
    0.46,
    0.49,
    0.52,
    0.55,
    0.58,
    0.61,
    0.65,
    0.92,
  ]
)

smoothedRange = np.concatenate([quantileRange[0::2][:13], quantileRange[25::2]])


@dataclass
class GaussianParams:
  bandwidth: float = 0.1
  smoothingWeight: float = 0.4
  smoothingValue: float = 0.35
  adaptiveWeightBase: Optional[int] = 9
  priorFactor: bool = True
  negWeight: float = 1.75
  minPrior: Optional[float] = 0.2
  weightLim: float = 1e-9
  somewhatHelpfulValue: float = 0.7
  clipLower: float = 0.05
  clipUpper: float = 0.8
  flip: bool = False
  normalize: bool = True


gaussianCrhParams = GaussianParams()

gaussianCrnhParams = GaussianParams(
  smoothingWeight=5.0,
  smoothingValue=0.35,
  bandwidth=0.2,
  minPrior=0.3,
  adaptiveWeightBase=9,
  negWeight=1.25,
  flip=True,
  clipUpper=0.65,
  clipLower=0.25,
  normalize=False,
  weightLim=5e-4,
)

# TSV Values
notHelpfulValueTsv = "NOT_HELPFUL"
somewhatHelpfulValueTsv = "SOMEWHAT_HELPFUL"
helpfulValueTsv = "HELPFUL"
notesSaysTweetIsMisleadingKey = "MISINFORMED_OR_POTENTIALLY_MISLEADING"
noteSaysTweetIsNotMisleadingKey = "NOT_MISLEADING"

# Rating Source Bucketed Values
ratingSourceDefaultValueTsv = "DEFAULT"
ratingSourcePopulationSampledValueTsv = "POPULATION_SAMPLED"

# Fields Transformed From the Raw Data
helpfulNumKey = "helpfulNum"
ratingCreatedBeforeMostRecentNMRLabelKey = "ratingCreatedBeforeMostRecentNMRLabel"
ratingCreatedBeforePublicTSVReleasedKey = "ratingCreatedBeforePublicTSVReleased"

# Timestamps
deletedNoteTombstonesLaunchTime = 1652918400000  # May 19, 2022 UTC
notMisleadingUILaunchTime = 1664755200000  # October 3, 2022 UTC
lastRatingTagsChangeTimeMillis = 1639699200000  # 2021/12/15 UTC
publicTSVTimeDelay = 172800000  # 48 hours

# Explanation Tags
tagCountsKey = "tagCounts"
tiebreakOrderKey = "tiebreakOrder"
firstTagKey = "firstTag"
secondTagKey = "secondTag"
activeFilterTagsKey = "activeFilterTags"

# Contributor Counts
successfulRatingHelpfulCount = "successfulRatingHelpfulCount"
successfulRatingNotHelpfulCount = "successfulRatingNotHelpfulCount"
successfulRatingTotal = "successfulRatingTotal"
unsuccessfulRatingHelpfulCount = "unsuccessfulRatingHelpfulCount"
unsuccessfulRatingNotHelpfulCount = "unsuccessfulRatingNotHelpfulCount"
unsuccessfulRatingTotal = "unsuccessfulRatingTotal"
ratingsAwaitingMoreRatings = "ratingsAwaitingMoreRatings"
ratedAfterDecision = "ratedAfterDecision"
notesCurrentlyRatedHelpful = "notesCurrentlyRatedHelpful"
notesCurrentlyRatedNotHelpful = "notesCurrentlyRatedNotHelpful"
notesAwaitingMoreRatings = "notesAwaitingMoreRatings"

# Meta Scoring Columns
finalRatingStatusKey = "finalRatingStatus"
unlockedRatingStatusKey = "unlockedRatingStatus"
preStabilizationRatingStatusKey = "preStabilizationRatingStatus"
metaScorerActiveRulesKey = "metaScorerActiveRules"
decidedByKey = "decidedBy"
rescoringActiveRulesKey = "rescoringActiveRules"

# Note Status Changes Columns
noteFinalStatusChange = "finalStatusChange"
noteNewRatings = "newRatings"
noteDecidedByChange = "decidedByChange"
noteAllAddedRules = "allAddedRules"
noteAllRemovedRules = "allRemovedRules"
noteDecidedByInterceptChange = "decidedByInterceptChange"

# Internal Scoring Columns.  These columns should be renamed before writing to disk.
internalNoteInterceptKey = "internalNoteIntercept"
internalRaterInterceptKey = "internalRaterIntercept"
internalNoteFactorKeyBase = "internalNoteFactor"
internalRaterFactorKeyBase = "internalRaterFactor"
internalRatingStatusKey = "internalRatingStatus"
internalActiveRulesKey = "internalActiveRules"
internalRaterReputationKey = "internalRaterReputation"
internalNoteInterceptNoHighVolKey = "internalNoteInterceptNoHighVol"
internalNoteInterceptNoCorrelatedKey = "internalNoteInterceptNoCorrelated"
internalNoteInterceptPopulationSampledKey = "internalNoteInterceptPopulationSampled"

# First Round MF Rater Scores (from prescoring)
internalFirstRoundRaterInterceptKey = "internalFirstRoundRaterIntercept"
internalFirstRoundRaterFactor1Key = "internalFirstRoundRaterFactor1"

scorerNameKey = "scorerName"


def note_factor_key(i):
  return internalNoteFactorKeyBase + str(i)


def rater_factor_key(i):
  return internalRaterFactorKeyBase + str(i)


internalNoteFactor1Key = note_factor_key(1)
internalRaterFactor1Key = rater_factor_key(1)

# Output Scoring Columns.
# Core Model
coreNoteInterceptKey = "coreNoteIntercept"
coreNoteFactor1Key = "coreNoteFactor1"
coreRaterInterceptKey = "coreRaterIntercept"
coreRaterFactor1Key = "coreRaterFactor1"
coreRatingStatusKey = "coreRatingStatus"
coreActiveRulesKey = "coreActiveRules"
coreNoteInterceptMaxKey = "coreNoteInterceptMax"
coreNoteInterceptMinKey = "coreNoteInterceptMin"
coreNumFinalRoundRatingsKey = "coreNumFinalRoundRatings"
coreNoteInterceptNoHighVolKey = "coreNoteInterceptNoHighVol"
coreNoteInterceptNoCorrelatedKey = "coreNoteInterceptNoCorrelated"
coreNoteInterceptPopulationSampledKey = "coreNoteInterceptPopulationSampled"
coreFirstRoundRaterInterceptKey = "coreFirstRoundRaterIntercept"
coreFirstRoundRaterFactor1Key = "coreFirstRoundRaterFactor1"
# Core No Topic Model
coreWithTopicsNoteInterceptKey = "coreWithTopicsNoteIntercept"
coreWithTopicsNoteFactor1Key = "coreWithTopicsNoteFactor1"
coreWithTopicsRaterInterceptKey = "coreWithTopicsRaterIntercept"
coreWithTopicsRaterFactor1Key = "coreWithTopicsRaterFactor1"
coreWithTopicsRatingStatusKey = "coreWithTopicsRatingStatus"
coreWithTopicsActiveRulesKey = "coreWithTopicsActiveRules"
coreWithTopicsNoteInterceptMaxKey = "coreWithTopicsNoteInterceptMax"
coreWithTopicsNoteInterceptMinKey = "coreWithTopicsNoteInterceptMin"
coreWithTopicsNumFinalRoundRatingsKey = "coreWithTopicsNumFinalRoundRatings"
coreWithTopicsNoteInterceptNoHighVolKey = "coreWithTopicsNoteInterceptNoHighVol"
coreWithTopicsNoteInterceptNoCorrelatedKey = "coreWithTopicsNoteInterceptNoCorrelated"
# Expansion Model
expansionNoteInterceptKey = "expansionNoteIntercept"
expansionNoteFactor1Key = "expansionNoteFactor1"
expansionRatingStatusKey = "expansionRatingStatus"
expansionNoteInterceptMaxKey = "expansionNoteInterceptMax"
expansionNoteInterceptMinKey = "expansionNoteInterceptMin"
expansionInternalActiveRulesKey = "expansionActiveRules"
expansionNumFinalRoundRatingsKey = "expansionNumFinalRoundRatings"
expansionRaterFactor1Key = "expansionRaterFactor1"
expansionRaterInterceptKey = "expansionRaterIntercept"
expansionNoteInterceptNoHighVolKey = "expansionNoteInterceptNoHighVol"
expansionNoteInterceptNoCorrelatedKey = "expansionNoteInterceptNoCorrelated"
# ExpansionPlus Model
expansionPlusNoteInterceptKey = "expansionPlusNoteIntercept"
expansionPlusNoteFactor1Key = "expansionPlusNoteFactor1"
expansionPlusRatingStatusKey = "expansionPlusRatingStatus"
expansionPlusInternalActiveRulesKey = "expansionPlusActiveRules"
expansionPlusNumFinalRoundRatingsKey = "expansionPlusNumFinalRoundRatings"
expansionPlusRaterFactor1Key = "expansionPlusRaterFactor1"
expansionPlusRaterInterceptKey = "expansionPlusRaterIntercept"
expansionPlusNoteInterceptNoHighVolKey = "expansionPlusNoteInterceptNoHighVol"
expansionPlusNoteInterceptNoCorrelatedKey = "expansionPlusNoteInterceptNoCorrelated"
# Coverage / Helpfulness Reputation Model
coverageNoteInterceptKey = "coverageNoteIntercept"
coverageNoteFactor1Key = "coverageNoteFactor1"
coverageRatingStatusKey = "coverageRatingStatus"
coverageNoteInterceptMaxKey = "coverageNoteInterceptMax"
coverageNoteInterceptMinKey = "coverageNoteInterceptMin"
raterHelpfulnessReputationKey = "raterHelpfulnessReputation"
# Group Model
groupNoteInterceptKey = "groupNoteIntercept"
groupNoteFactor1Key = "groupNoteFactor1"
groupRatingStatusKey = "groupRatingStatus"
groupNoteInterceptMaxKey = "groupNoteInterceptMax"
groupNoteInterceptMinKey = "groupNoteInterceptMin"
groupRaterInterceptKey = "groupRaterIntercept"
groupRaterFactor1Key = "groupRaterFactor1"
groupInternalActiveRulesKey = "groupActiveRules"
groupNumFinalRoundRatingsKey = "groupNumFinalRoundRatings"
groupNoteInterceptNoHighVolKey = "groupNoteInterceptNoHighVol"
groupNoteInterceptNoCorrelatedKey = "groupNoteInterceptNoCorrelated"
# MultiGroup Model
multiGroupNoteInterceptKey = "multiGroupNoteIntercept"
multiGroupNoteFactor1Key = "multiGroupNoteFactor1"
multiGroupRatingStatusKey = "multiGroupRatingStatus"
multiGroupRaterInterceptKey = "multiGroupRaterIntercept"
multiGroupRaterFactor1Key = "multiGroupRaterFactor1"
multiGroupInternalActiveRulesKey = "multiGroupActiveRules"
multiGroupNumFinalRoundRatingsKey = "multiGroupNumFinalRoundRatings"
multiGroupNoteInterceptNoHighVolKey = "multiGroupNoteInterceptNoHighVol"
multiGroupNoteInterceptNoCorrelatedKey = "multiGroupNoteInterceptNoCorrelated"
# Topic Model
topicNoteInterceptKey = "topicNoteIntercept"
topicNoteFactor1Key = "topicNoteFactor1"
topicRatingStatusKey = "topicRatingStatus"
topicNoteConfidentKey = "topicNoteConfident"
topicInternalActiveRulesKey = "topicActiveRules"
topicNumFinalRoundRatingsKey = "topicNumFinalRoundRatings"
topicNoteInterceptNoHighVolKey = "topicNoteInterceptNoHighVol"
topicNoteInterceptNoCorrelatedKey = "topicNoteInterceptNoCorrelated"
# Gaussian Model
gaussianNoteInterceptKey = "gaussianNoteIntercept"
gaussianNoteFactor1Key = "gaussianNoteFactor1"
gaussianRatingStatusKey = "gaussianRatingStatus"
gaussianActiveRulesKey = "gaussianActiveRules"
gaussianNumFinalRoundRatingsKey = "gaussianNumFinalRoundRatings"
gaussianNoteInterceptNoHighVolKey = "gaussianNoteInterceptNoHighVol"
gaussianNoteInterceptNoCorrelatedKey = "gaussianNoteInterceptNoCorrelated"
gaussianNoteInterceptPopulationSampledKey = "gaussianNoteInterceptPopulationSampled"
# Harassment/Abuse Tag
harassmentNoteInterceptKey = "harassmentNoteIntercept"
harassmentNoteFactor1Key = "harassmentNoteFactor1"
harassmentRaterInterceptKey = "harassmentRaterIntercept"
harassmentRaterFactor1Key = "harassmentRaterFactor1"

# Ids and Indexes
noteIdKey = "noteId"
tweetIdKey = "tweetId"
classificationKey = "classification"
noteAuthorParticipantIdKey = "noteAuthorParticipantId"
raterParticipantIdKey = "raterParticipantId"

# Aggregations
noteCountKey = "noteCount"
ratingCountKey = "ratingCount"
numRatingsKey = "numRatings"
numRatingsLast28DaysKey = "numRatingsLast28"
numPopulationSampledRatingsKey = "numPopulationSampledRatings"
ratingFromInitialModelingGroupKey = "ratingFromInitialModelingGroup"
percentFromInitialModelingGroupKey = "percentFromInitialModelingGroup"
numFinalRoundRatingsKey = "numFinalRoundRatings"

# Helpfulness Score Keys
crhRatioKey = "CRHRatio"
crnhRatioKey = "CRNHRatio"
crhCrnhRatioDifferenceKey = "crhCrnhRatioDifference"
meanNoteScoreKey = "meanNoteScore"
raterAgreeRatioKey = "raterAgreeRatio"
ratingAgreesWithNoteStatusKey = "ratingAgreesWithNoteStatus"
aboveHelpfulnessThresholdKey = "aboveHelpfulnessThreshold"
totalHelpfulHarassmentRatingsPenaltyKey = "totalHelpfulHarassmentPenalty"
raterAgreeRatioWithHarassmentAbusePenaltyKey = "raterAgreeRatioKeyWithHarassmentAbusePenalty"

# Note Status Labels
currentlyRatedHelpful = "CURRENTLY_RATED_HELPFUL"
currentlyRatedNotHelpful = "CURRENTLY_RATED_NOT_HELPFUL"
needsMoreRatings = "NEEDS_MORE_RATINGS"
# FIRM_REJECT is set by individual scorers to indicate downstream scorers should not CRH
# a note, but is never set as the finalRatingStatus of a note.
firmReject = "FIRM_REJECT"
# NEEDS_YOUR_HELP is set by individual scorers to indicate that the note should show as
# a NYH pivot if the note has not exceeded the maximum stabilization time.
needsYourHelp = "NEEDS_YOUR_HELP"

# Boolean Note Status Labels
currentlyRatedHelpfulBoolKey = "crhBool"
currentlyRatedNotHelpfulBoolKey = "crnhBool"
awaitingMoreRatingsBoolKey = "awaitingBool"

helpfulOtherTagKey = "helpfulOther"
helpfulInformativeTagKey = "helpfulInformative"
helpfulClearTagKey = "helpfulClear"
helpfulEmpatheticTagKey = "helpfulEmpathetic"
helpfulGoodSourcesTagKey = "helpfulGoodSources"
helpfulUniqueContextTagKey = "helpfulUniqueContext"
helpfulAddressesClaimTagKey = "helpfulAddressesClaim"
helpfulImportantContextTagKey = "helpfulImportantContext"
helpfulUnbiasedLanguageTagKey = "helpfulUnbiasedLanguage"

helpfulTagsAndTieBreakOrder = [
  (0, helpfulOtherTagKey),
  (8, helpfulInformativeTagKey),
  (7, helpfulClearTagKey),
  (3, helpfulEmpatheticTagKey),
  (4, helpfulGoodSourcesTagKey),
  (2, helpfulUniqueContextTagKey),
  (5, helpfulAddressesClaimTagKey),
  (6, helpfulImportantContextTagKey),
  (1, helpfulUnbiasedLanguageTagKey),
]
helpfulTagsTSVOrder = [tag for (tiebreakOrder, tag) in helpfulTagsAndTieBreakOrder]
helpfulTagBoolsAndTypesTSVOrder = [(tag, pd.Int8Dtype()) for tag in helpfulTagsTSVOrder]
helpfulTagsTiebreakOrder = [tag for (tiebreakOrder, tag) in sorted(helpfulTagsAndTieBreakOrder)]
helpfulTagCountsAndTypesTSVOrder = [(tag, pd.Int64Dtype()) for tag in helpfulTagsTSVOrder]


# NOTE: Always add new tags to the end of this list, and *never* change the order of
# elements which are already in the list to maintain compatibility with
# BirdwatchNoteNotHelpfulTags.get in Scala.

notHelpfulIncorrectTagKey = "notHelpfulIncorrect"
notHelpfulOtherTagKey = "notHelpfulOther"
notHelpfulSpamHarassmentOrAbuseTagKey = "notHelpfulSpamHarassmentOrAbuse"
notHelpfulArgumentativeOrBiasedTagKey = "notHelpfulArgumentativeOrBiased"
notHelpfulHardToUnderstandKey = "notHelpfulHardToUnderstand"
notHelpfulNoteNotNeededKey = "notHelpfulNoteNotNeeded"
notHelpfulSourcesMissingOrUnreliableTagKey = "notHelpfulSourcesMissingOrUnreliable"
notHelpfulIrrelevantSourcesTagKey = "notHelpfulIrrelevantSources"
notHelpfulOpinionSpeculationOrBiasTagKey = "notHelpfulOpinionSpeculationOrBias"
notHelpfulMissingKeyPointsTagKey = "notHelpfulMissingKeyPoints"
notHelpfulOutdatedTagKey = "notHelpfulOutdated"
notHelpfulOffTopicTagKey = "notHelpfulOffTopic"
notHelpfulOpinionSpeculationTagKey = "notHelpfulOpinionSpeculation"

## This list is in TSV Order, but with indices for tiebreak order.
notHelpfulTagsAndTieBreakOrder = [
  (0, notHelpfulOtherTagKey),  ## should lose all tiebreaks
  (8, notHelpfulIncorrectTagKey),
  (2, notHelpfulSourcesMissingOrUnreliableTagKey),
  (4, notHelpfulOpinionSpeculationOrBiasTagKey),
  (5, notHelpfulMissingKeyPointsTagKey),
  (12, notHelpfulOutdatedTagKey),  ## should win all tiebreaks
  (10, notHelpfulHardToUnderstandKey),
  (7, notHelpfulArgumentativeOrBiasedTagKey),
  (9, notHelpfulOffTopicTagKey),
  (11, notHelpfulSpamHarassmentOrAbuseTagKey),
  (1, notHelpfulIrrelevantSourcesTagKey),
  (3, notHelpfulOpinionSpeculationTagKey),
  (6, notHelpfulNoteNotNeededKey),
]
notHelpfulTagsTSVOrder = [tag for (tiebreakOrder, tag) in notHelpfulTagsAndTieBreakOrder]
notHelpfulTagsAndTypesTSVOrder = [(tag, pd.Int8Dtype()) for tag in notHelpfulTagsTSVOrder]
notHelpfulTagCountsAndTypesTSVOrder = [(tag, pd.Int64Dtype()) for tag in notHelpfulTagsTSVOrder]
notHelpfulTagsTiebreakOrder = [
  tag for (tiebreakOrder, tag) in sorted(notHelpfulTagsAndTieBreakOrder)
]
notHelpfulTagsTiebreakMapping = {
  tag: priority for (priority, tag) in notHelpfulTagsAndTieBreakOrder
}
notHelpfulTagsEnumMapping = {
  tag: idx for (idx, (_, tag)) in enumerate(notHelpfulTagsAndTieBreakOrder)
}
adjustedSuffix = "Adjusted"
notHelpfulTagsAdjustedColumns = [f"{column}{adjustedSuffix}" for column in notHelpfulTagsTSVOrder]
notHelpfulTagsAdjustedTSVColumnsAndTypes = [
  (tag, np.double) for tag in notHelpfulTagsAdjustedColumns
]
ratioSuffix = "Ratio"
notHelpfulTagsAdjustedRatioColumns = [
  f"{column}{ratioSuffix}" for column in notHelpfulTagsAdjustedColumns
]
notHelpfulTagsAdjustedRatioTSVColumnsAndTypes = [
  (tag, np.double) for tag in notHelpfulTagsAdjustedRatioColumns
]
ratingWeightKey = "ratingWeight"

# Common substrings used for column name filtering
noteInterceptMinSubstring = "NoteInterceptMin"
noteInterceptMaxSubstring = "NoteInterceptMax"
noHighVolSubstring = "NoHighVol"
noCorrelatedSubstring = "NoCorrelated"
populationSampledSubstring = "PopulationSampled"

incorrectTagRatingsMadeByRaterKey = "incorrectTagRatingsMadeByRater"
totalRatingsMadeByRaterKey = "totalRatingsMadeByRater"

noteTfIdfIncorrectScoreKey = "tf_idf_incorrect"
numVotersKey = "num_voters"  # num voters who rated a note
incorrectTagRateByRaterKey = "p_incorrect_user"

noteTfIdfIncorrectScoreIntervalKey = (
  "tf_idf_incorrect_interval"  # note's tf-idf scores from within the interval
)
numVotersIntervalKey = "num_voters_interval"  # num voters (in the interval) who rated a note
sumOfIncorrectTagRateByRaterIntervalKey = (
  "p_incorrect_user_interval"
)  # sum of p_incorrect_user for all raters who rated a note in the interval
notHelpfulIncorrectIntervalKey = (
  "notHelpfulIncorrect_interval"  # notHelpfulIncorrect ratings on the note in the interval
)

lowDiligenceInterceptKey = "lowDiligenceIntercept"


lowDiligenceRaterFactor1Key = "lowDiligenceRaterFactor1"
lowDiligenceRaterInterceptKey = "lowDiligenceRaterIntercept"
lowDiligenceRaterReputationKey = "lowDiligenceRaterReputation"
lowDiligenceNoteFactor1Key = "lowDiligenceNoteFactor1"
lowDiligenceNoteInterceptKey = "lowDiligenceNoteIntercept"
lowDiligenceLegacyNoteInterceptKey = "lowDiligenceIntercept"
lowDiligenceNoteInterceptRound2Key = "lowDiligenceNoteInterceptRound2"
internalNoteInterceptRound2Key = "internalNoteInterceptRound2"
lowDiligenceRaterInterceptRound2Key = "lowDiligenceRaterInterceptRound2"
internalRaterInterceptRound2Key = "internalRaterInterceptRound2"

incorrectFilterColumnsAndTypes = [
  (notHelpfulIncorrectIntervalKey, np.double),
  (sumOfIncorrectTagRateByRaterIntervalKey, np.double),
  (numVotersIntervalKey, np.double),
  (noteTfIdfIncorrectScoreIntervalKey, np.double),
  (lowDiligenceLegacyNoteInterceptKey, np.double),
]
incorrectFilterColumns = [col for (col, _) in incorrectFilterColumnsAndTypes]

misleadingOtherKey = "misleadingOther"
misleadingFactualErrorKey = "misleadingFactualError"
misleadingManipulatedMediaKey = "misleadingManipulatedMedia"
misleadingOutdatedInformationKey = "misleadingOutdatedInformation"
misleadingMissingImportantContextKey = "misleadingMissingImportantContext"
misleadingUnverifiedClaimAsFactKey = "misleadingUnverifiedClaimAsFact"
misleadingSatireKey = "misleadingSatire"

misleadingTags = [
  misleadingOtherKey,
  misleadingFactualErrorKey,
  misleadingManipulatedMediaKey,
  misleadingOutdatedInformationKey,
  misleadingMissingImportantContextKey,
  misleadingUnverifiedClaimAsFactKey,
  misleadingSatireKey,
]
misleadingTagsAndTypes = [(tag, pd.Int8Dtype()) for tag in misleadingTags]

notMisleadingOtherKey = "notMisleadingOther"
notMisleadingFactuallyCorrectKey = "notMisleadingFactuallyCorrect"
notMisleadingOutdatedButNotWhenWrittenKey = "notMisleadingOutdatedButNotWhenWritten"
notMisleadingClearlySatireKey = "notMisleadingClearlySatire"
notMisleadingPersonalOpinionKey = "notMisleadingPersonalOpinion"
notMisleadingTags = [
  notMisleadingOtherKey,
  notMisleadingFactuallyCorrectKey,
  notMisleadingOutdatedButNotWhenWrittenKey,
  notMisleadingClearlySatireKey,
  notMisleadingPersonalOpinionKey,
]
notMisleadingTagsAndTypes = [(tag, pd.Int8Dtype()) for tag in notMisleadingTags]

believableKey = "believable"
harmfulKey = "harmful"
validationDifficultyKey = "validationDifficulty"
trustworthySourcesKey = "trustworthySources"
isMediaNoteKey = "isMediaNote"
isCollaborativeNoteKey = "isCollaborativeNote"

noteTSVColumnsAndTypes = (
  [
    (noteIdKey, np.int64),
    (noteAuthorParticipantIdKey, object),
    (createdAtMillisKey, np.int64),
    (tweetIdKey, np.int64),
    (classificationKey, object),
    (believableKey, "category"),
    (harmfulKey, "category"),
    (validationDifficultyKey, "category"),
  ]
  + misleadingTagsAndTypes
  + notMisleadingTagsAndTypes
  + [
    (trustworthySourcesKey, pd.Int8Dtype()),
    (summaryKey, object),
    (isMediaNoteKey, pd.Int8Dtype()),
    (isCollaborativeNoteKey, pd.Int8Dtype()),
  ]
)
noteTSVColumns = [col for (col, dtype) in noteTSVColumnsAndTypes]
noteTSVTypes = [dtype for (col, dtype) in noteTSVColumnsAndTypes]
noteTSVTypeMapping = {col: dtype for (col, dtype) in noteTSVColumnsAndTypes}

versionKey = "version"
agreeKey = "agree"
disagreeKey = "disagree"
ratedOnTweetIdKey = "ratedOnTweetId"
ratingSourceBucketedKey = "ratingSourceBucketed"
ratingTSVColumnsAndTypes = (
  [
    (noteIdKey, np.int64),
    (raterParticipantIdKey, object),
    (createdAtMillisKey, np.int64),
    (versionKey, pd.Int8Dtype()),
    (agreeKey, pd.Int8Dtype()),
    (disagreeKey, pd.Int8Dtype()),
    (helpfulKey, pd.Int8Dtype()),
    (notHelpfulKey, pd.Int8Dtype()),
    (helpfulnessLevelKey, "category"),
  ]
  + helpfulTagBoolsAndTypesTSVOrder
  + notHelpfulTagsAndTypesTSVOrder
  + [(ratedOnTweetIdKey, np.int64), (ratingSourceBucketedKey, "category")]
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
timestampMillisOfStatusLockKey = "timestampMillisOfStatusLock"
lockedStatusKey = "lockedStatus"
timestampMillisOfRetroLockKey = "timestampMillisOfRetroLock"
currentCoreStatusKey = "currentCoreStatus"
currentExpansionStatusKey = "currentExpansionStatus"
currentGroupStatusKey = "currentGroupStatus"
currentDecidedByKey = "currentDecidedBy"
currentModelingGroupKey = "currentModelingGroup"
timestampMillisOfMostRecentStatusChangeKey = "timestampMillisOfMostRecentStatusChange"
currentMultiGroupStatusKey = "currentMultiGroupStatus"
currentModelingMultiGroupKey = "currentModelingMultiGroup"
timestampMillisOfNmrDueToMinStableCrhTimeKey = "timestampMillisOfNmrDueToMinStableCrhTime"
updatedTimestampMillisOfNmrDueToMinStableCrhTimeKey = (
  "updatedTimestampMillisOfNmrDueToMinStableCrhTime"
)
timestampMinuteOfFinalScoringOutput = "timestampMinuteOfFinalScoringOutput"
timestampMillisOfFirstNmrDueToMinStableCrhTimeKey = "timestampMillisOfFirstNmrDueToMinStableCrhTime"

noteStatusHistoryTSVColumnsAndTypes = [
  (noteIdKey, np.int64),
  (noteAuthorParticipantIdKey, object),
  (createdAtMillisKey, np.int64),
  (timestampMillisOfNoteFirstNonNMRLabelKey, np.double),  # double because nullable.
  (firstNonNMRLabelKey, "category"),
  (timestampMillisOfNoteCurrentLabelKey, np.double),  # double because nullable.
  (currentLabelKey, "category"),
  (timestampMillisOfNoteMostRecentNonNMRLabelKey, np.double),  # double because nullable.
  (mostRecentNonNMRLabelKey, "category"),
  (timestampMillisOfStatusLockKey, np.double),  # double because nullable.
  (lockedStatusKey, "category"),
  (timestampMillisOfRetroLockKey, np.double),  # double because nullable.
  (currentCoreStatusKey, "category"),
  (currentExpansionStatusKey, "category"),
  (currentGroupStatusKey, "category"),
  (currentDecidedByKey, "category"),
  (currentModelingGroupKey, np.double),  # TODO: int
  (timestampMillisOfMostRecentStatusChangeKey, np.double),  # double because nullable.
  (timestampMillisOfNmrDueToMinStableCrhTimeKey, np.double),  # double because nullable.
  (currentMultiGroupStatusKey, "category"),
  (currentModelingMultiGroupKey, np.double),  # TODO: int
  (timestampMinuteOfFinalScoringOutput, np.double),  # double because nullable.
  (timestampMillisOfFirstNmrDueToMinStableCrhTimeKey, np.double),  # double because nullable.
]
noteStatusHistoryTSVColumns = [col for (col, dtype) in noteStatusHistoryTSVColumnsAndTypes]
noteStatusHistoryTSVTypes = [dtype for (col, dtype) in noteStatusHistoryTSVColumnsAndTypes]
noteStatusHistoryTSVTypeMapping = {
  col: dtype for (col, dtype) in noteStatusHistoryTSVColumnsAndTypes
}
# TODO(jiansongc): clean up after new column is in production.
noteStatusHistoryTSVColumnsOld = noteStatusHistoryTSVColumns[:-1]
noteStatusHistoryTSVColumnsAndTypesOld = noteStatusHistoryTSVColumnsAndTypes[:-1]
noteStatusHistoryTSVTypeMappingOld = {
  col: dtype for (col, dtype) in noteStatusHistoryTSVColumnsAndTypesOld
}


# Earn In + Earn Out
enrollmentState = "enrollmentState"
successfulRatingNeededToEarnIn = "successfulRatingNeededToEarnIn"
timestampOfLastStateChange = "timestampOfLastStateChange"
timestampOfLastEarnOut = "timestampOfLastEarnOut"
authorTopNotHelpfulTagValues = "authorTopNotHelpfulTagValues"
maxHistoryEarnOut = 5
successfulRatingHelpfulCount = "successfulRatingHelpfulCount"
earnedIn = "earnedIn"
atRisk = "atRisk"
earnedOutNoAcknowledge = "earnedOutNoAcknowledge"
earnedOutAcknowledged = "earnedOutAcknowledged"
newUser = "newUser"
removed = "removed"
apiTestUser = "apiTestUser"
apiEarnedIn = "apiEarnedIn"
apiEarnedOut = "apiEarnedOut"
isAtRiskCRNHCount = 2
ratingImpactForEarnIn = 5
ratingImpact = "ratingImpact"
enrollmentStateToThrift = {
  earnedIn: 0,
  atRisk: 1,
  earnedOutNoAcknowledge: 2,
  earnedOutAcknowledged: 3,
  newUser: 4,
  removed: 5,
  apiTestUser: 6,
  apiEarnedIn: 7,
  apiEarnedOut: 8,
}
emergingWriterDays = 28
isEmergingWriterKey = "isEmergingWriter"
emergingMeanNoteScore = 0.3
emergingRatingCount = 10
aggregateRatingReceivedTotal = "aggregateRatingReceivedTotal"
core = "CORE"
coreWithTopics = "CORE_WITH_TOPICS"
expansion = "EXPANSION"
expansionPlus = "EXPANSION_PLUS"
topWriterWritingImpact = 10
topWriterHitRate = 0.04
hasCrnhSinceEarnOut = "hasCrnhSinceEarnOut"

userEnrollmentTSVColumnsAndTypes = [
  (participantIdKey, str),
  (enrollmentState, str),
  (successfulRatingNeededToEarnIn, np.int64),
  (timestampOfLastStateChange, np.int64),
  (timestampOfLastEarnOut, np.double),  # double because nullable.
  (modelingPopulationKey, "category"),
  (modelingGroupKey, np.float64),
  (numberOfTimesEarnedOutKey, np.int64),
]
userEnrollmentTSVColumns = [col for (col, _) in userEnrollmentTSVColumnsAndTypes]
userEnrollmentTSVTypes = [dtype for (_, dtype) in userEnrollmentTSVColumnsAndTypes]
userEnrollmentTSVTypeMapping = {col: dtype for (col, dtype) in userEnrollmentTSVColumnsAndTypes}

noteInterceptMaxKey = "internalNoteIntercept_max"
noteInterceptMinKey = "internalNoteIntercept_min"
noteParameterUncertaintyTSVMainColumnsAndTypes = [
  (noteInterceptMaxKey, np.double),
  (noteInterceptMinKey, np.double),
]

negFactorRatingCountKey = "negFactor_ratingCount"
posFactorRatingCountKey = "posFactor_ratingCount"
negFactorMeanHelpfulNumKey = "negFactor_meanHelpfulNum"
posFactorMeanHelpfulNumKey = "posFactor_meanHelpfulNum"
minSignCountKey = "minSignCount"
maxSignCountKey = "maxSignCount"
netMinHelpfulKey = "netMinHelpful"
netMinHelpfulRatioKey = "netMinHelpfulRatio"

# Population sampled per-sign counts
negFactorPopulationSampledRatingCountKey = "negFactor_populationSampledRatingCount"
posFactorPopulationSampledRatingCountKey = "posFactor_populationSampledRatingCount"

# Core population sampled per-sign counts
coreNegFactorPopulationSampledRatingCountKey = "coreNegFactor_populationSampledRatingCount"
corePosFactorPopulationSampledRatingCountKey = "corePosFactor_populationSampledRatingCount"

noteParameterUncertaintyTSVAuxColumnsAndTypes = [
  ("internalNoteFactor1_max", np.double),
  ("internalNoteFactor1_median", np.double),
  ("internalNoteFactor1_min", np.double),
  ("internalNoteFactor1_refit_orig", np.double),
  ("internalNoteIntercept_median", np.double),
  ("internalNoteIntercept_refit_orig", np.double),
  ("ratingCount_all", np.int64),
  ("ratingCount_neg_fac", np.int64),
  ("ratingCount_pos_fac", np.int64),
]
noteParameterUncertaintyTSVColumnsAndTypes = (
  noteParameterUncertaintyTSVAuxColumnsAndTypes + noteParameterUncertaintyTSVMainColumnsAndTypes
)
noteParameterUncertaintyTSVColumns = [
  col for (col, _) in noteParameterUncertaintyTSVColumnsAndTypes
]
noteParameterUncertaintyTSVAuxColumns = [
  col for (col, _) in noteParameterUncertaintyTSVAuxColumnsAndTypes
]
noteParameterUncertaintyTSVMainColumns = [
  col for (col, _) in noteParameterUncertaintyTSVMainColumnsAndTypes
]
noteParameterUncertaintyTSVTypes = [
  dtype for (_, dtype) in noteParameterUncertaintyTSVColumnsAndTypes
]
noteParameterUncertaintyTSVTypeMapping = {
  col: dtype for (col, dtype) in noteParameterUncertaintyTSVColumnsAndTypes
}

auxiliaryScoredNotesTSVColumnsAndTypes = (
  [
    (noteIdKey, np.int64),
    (ratingWeightKey, np.double),
    (createdAtMillisKey, np.int64),
    (noteAuthorParticipantIdKey, object),
    (awaitingMoreRatingsBoolKey, np.int8),
    (numRatingsLast28DaysKey, np.int64),
    (numPopulationSampledRatingsKey, np.int64),
    (currentLabelKey, str),
    (currentlyRatedHelpfulBoolKey, np.int8),
    (currentlyRatedNotHelpfulBoolKey, np.int8),
    (unlockedRatingStatusKey, str),
    (preStabilizationRatingStatusKey, str),
  ]
  + helpfulTagCountsAndTypesTSVOrder
  + notHelpfulTagCountsAndTypesTSVOrder
  + notHelpfulTagsAdjustedTSVColumnsAndTypes
  + notHelpfulTagsAdjustedRatioTSVColumnsAndTypes
  + incorrectFilterColumnsAndTypes
  + [
    (coreNegFactorPopulationSampledRatingCountKey, np.int64),
    (corePosFactorPopulationSampledRatingCountKey, np.int64),
  ]
)
auxiliaryScoredNotesTSVColumns = [col for (col, dtype) in auxiliaryScoredNotesTSVColumnsAndTypes]
auxiliaryScoredNotesTSVTypeMapping = {
  col: dtype for (col, dtype) in auxiliaryScoredNotesTSVColumnsAndTypes
}

deprecatedNoteModelOutputColumns = frozenset(
  {
    coverageNoteInterceptMinKey,
    coverageNoteInterceptMaxKey,
    groupNoteInterceptMinKey,
    groupNoteInterceptMaxKey,
  }
)

prescoringNoteModelOutputTSVColumnsAndTypes = [
  (noteIdKey, np.int64),
  (internalNoteInterceptKey, np.double),
  (internalNoteFactor1Key, np.double),
  (scorerNameKey, str),
  (lowDiligenceNoteInterceptKey, np.double),
  (lowDiligenceNoteFactor1Key, np.double),
  (lowDiligenceNoteInterceptRound2Key, np.double),
  (harassmentNoteInterceptKey, np.double),
  (harassmentNoteFactor1Key, np.double),
]
prescoringNoteModelOutputTSVColumns = [
  col for (col, dtype) in prescoringNoteModelOutputTSVColumnsAndTypes
]
prescoringNoteModelOutputTSVTypeMapping = {
  col: dtype for (col, dtype) in prescoringNoteModelOutputTSVColumnsAndTypes
}

noteModelOutputTSVColumnsAndTypes = [
  (noteIdKey, np.int64),
  (coreNoteInterceptKey, np.double),
  (coreNoteFactor1Key, np.double),
  (finalRatingStatusKey, "category"),
  (firstTagKey, "category"),
  (secondTagKey, "category"),
  # Note that this column was formerly named "activeRules" and the name is now
  # updated to "coreActiveRules".  The data values remain the compatible,
  # but the new column only contains rules that ran when deciding status based on
  # the core model.
  (coreActiveRulesKey, "category"),
  (activeFilterTagsKey, "category"),
  (classificationKey, "category"),
  (createdAtMillisKey, np.int64),
  (coreRatingStatusKey, "category"),
  (metaScorerActiveRulesKey, "category"),
  (decidedByKey, "category"),
  (expansionNoteInterceptKey, np.double),
  (expansionNoteFactor1Key, np.double),
  (expansionRatingStatusKey, "category"),
  (coverageNoteInterceptKey, np.double),
  (coverageNoteFactor1Key, np.double),
  (coverageRatingStatusKey, "category"),
  (coreNoteInterceptMinKey, np.double),
  (coreNoteInterceptMaxKey, np.double),
  (expansionNoteInterceptMinKey, "category"),  # category because always nan
  (expansionNoteInterceptMaxKey, "category"),  # category because always nan
  (coverageNoteInterceptMinKey, "category"),  # category because always nan
  (coverageNoteInterceptMaxKey, "category"),  # category because always nan
  (groupNoteInterceptKey, np.double),
  (groupNoteFactor1Key, np.double),
  (groupRatingStatusKey, "category"),
  (groupNoteInterceptMaxKey, "category"),  # category because always nan
  (groupNoteInterceptMinKey, "category"),  # category because always nan
  (modelingGroupKey, np.float64),
  (numRatingsKey, np.int64),
  (timestampMillisOfNoteCurrentLabelKey, np.double),
  (expansionPlusNoteInterceptKey, np.double),
  (expansionPlusNoteFactor1Key, np.double),
  (expansionPlusRatingStatusKey, "category"),
  (topicNoteInterceptKey, np.double),
  (topicNoteFactor1Key, np.double),
  (topicRatingStatusKey, "category"),
  (noteTopicKey, "category"),
  (topicNoteConfidentKey, pd.BooleanDtype()),
  (expansionInternalActiveRulesKey, "category"),
  (expansionPlusInternalActiveRulesKey, "category"),
  (groupInternalActiveRulesKey, "category"),
  (topicInternalActiveRulesKey, "category"),
  (coreNumFinalRoundRatingsKey, np.double),  # double because nullable.
  (expansionNumFinalRoundRatingsKey, np.double),  # double because nullable.
  (expansionPlusNumFinalRoundRatingsKey, np.double),  # double because nullable.
  (groupNumFinalRoundRatingsKey, np.double),  # double because nullable.
  (topicNumFinalRoundRatingsKey, np.double),  # double because nullable.
  (rescoringActiveRulesKey, "category"),
  (multiGroupNoteInterceptKey, np.double),
  (multiGroupNoteFactor1Key, np.double),
  (multiGroupRatingStatusKey, str),
  (modelingMultiGroupKey, np.float64),
  (multiGroupInternalActiveRulesKey, str),
  (multiGroupNumFinalRoundRatingsKey, np.double),  # double because nullable.
  (coreWithTopicsNoteFactor1Key, np.double),
  (coreWithTopicsNoteInterceptKey, np.double),
  (coreWithTopicsRatingStatusKey, "category"),
  (coreWithTopicsActiveRulesKey, "category"),
  (coreWithTopicsNumFinalRoundRatingsKey, np.double),  # double because nullable.
  (coreWithTopicsNoteInterceptMinKey, np.double),
  (coreWithTopicsNoteInterceptMaxKey, np.double),
  (timestampMillisOfNmrDueToMinStableCrhTimeKey, np.double),  # double because nullable.
  (coreNoteInterceptNoHighVolKey, "category"),
  (coreWithTopicsNoteInterceptNoHighVolKey, "category"),
  (expansionNoteInterceptNoHighVolKey, "category"),
  (expansionPlusNoteInterceptNoHighVolKey, "category"),
  (groupNoteInterceptNoHighVolKey, "category"),
  (multiGroupNoteInterceptNoHighVolKey, "category"),
  (topicNoteInterceptNoHighVolKey, "category"),
  (coreNoteInterceptNoCorrelatedKey, "category"),
  (coreNoteInterceptPopulationSampledKey, np.double),
  (coreWithTopicsNoteInterceptNoCorrelatedKey, "category"),
  (expansionNoteInterceptNoCorrelatedKey, "category"),
  (expansionPlusNoteInterceptNoCorrelatedKey, "category"),
  (groupNoteInterceptNoCorrelatedKey, "category"),
  (multiGroupNoteInterceptNoCorrelatedKey, "category"),
  (topicNoteInterceptNoCorrelatedKey, "category"),
  (gaussianNoteInterceptKey, np.double),
  (gaussianNoteFactor1Key, np.double),
  (gaussianRatingStatusKey, "category"),
  (gaussianActiveRulesKey, "category"),
  (gaussianNoteInterceptNoCorrelatedKey, np.double),
  (gaussianNoteInterceptNoHighVolKey, np.double),
  (gaussianNoteInterceptPopulationSampledKey, np.double),
  (gaussianNumFinalRoundRatingsKey, np.double),  # double because nullable.
]
noteModelOutputTSVColumns = [col for (col, dtype) in noteModelOutputTSVColumnsAndTypes]
noteModelOutputTSVTypeMapping = {col: dtype for (col, dtype) in noteModelOutputTSVColumnsAndTypes}
deprecatedNoteModelOutputTSVColumnsAndTypes = [
  (col, dtype)
  for (col, dtype) in noteModelOutputTSVColumnsAndTypes
  if col in deprecatedNoteModelOutputColumns
]

postSelectionValueKey = "postSelectionValue"
quasiCliqueValueKey = "quasiCliqueValue"

prescoringRaterModelOutputTSVColumnsAndTypes = [
  (raterParticipantIdKey, object),
  (internalRaterInterceptKey, np.double),
  (internalRaterFactor1Key, np.double),
  (internalFirstRoundRaterInterceptKey, np.double),
  (internalFirstRoundRaterFactor1Key, np.double),
  (crhCrnhRatioDifferenceKey, np.double),
  (meanNoteScoreKey, np.double),
  (raterAgreeRatioKey, np.double),
  (aboveHelpfulnessThresholdKey, pd.BooleanDtype()),
  (scorerNameKey, str),
  (internalRaterReputationKey, np.double),
  (lowDiligenceRaterInterceptKey, np.double),
  (lowDiligenceRaterFactor1Key, np.double),
  (lowDiligenceRaterReputationKey, np.double),
  (lowDiligenceRaterInterceptRound2Key, np.double),
  (incorrectTagRatingsMadeByRaterKey, pd.Int64Dtype()),
  (totalRatingsMadeByRaterKey, pd.Int64Dtype()),
  (postSelectionValueKey, pd.Int64Dtype()),
  (successfulRatingHelpfulCount, pd.Int64Dtype()),
  (successfulRatingNotHelpfulCount, pd.Int64Dtype()),
  (unsuccessfulRatingHelpfulCount, pd.Int64Dtype()),
  (unsuccessfulRatingNotHelpfulCount, pd.Int64Dtype()),
  (totalHelpfulHarassmentRatingsPenaltyKey, np.double),
  (raterAgreeRatioWithHarassmentAbusePenaltyKey, np.double),
  (quasiCliqueValueKey, pd.Int64Dtype()),
]

prescoringRaterModelOutputTSVColumns = [
  col for (col, dtype) in prescoringRaterModelOutputTSVColumnsAndTypes
]
prescoringRaterModelOutputTSVTypeMapping = {
  col: dtype for (col, dtype) in prescoringRaterModelOutputTSVColumnsAndTypes
}

raterModelOutputTSVColumnsAndTypes = [
  (raterParticipantIdKey, np.int64),
  (coreRaterInterceptKey, np.double),
  (coreRaterFactor1Key, np.double),
  (crhCrnhRatioDifferenceKey, np.double),
  (meanNoteScoreKey, np.double),
  (raterAgreeRatioKey, np.double),
  (successfulRatingHelpfulCount, pd.Int64Dtype()),
  (successfulRatingNotHelpfulCount, pd.Int64Dtype()),
  (successfulRatingTotal, pd.Int64Dtype()),
  (unsuccessfulRatingHelpfulCount, pd.Int64Dtype()),
  (unsuccessfulRatingNotHelpfulCount, pd.Int64Dtype()),
  (unsuccessfulRatingTotal, pd.Int64Dtype()),
  (ratingsAwaitingMoreRatings, pd.Int64Dtype()),
  (ratedAfterDecision, pd.Int64Dtype()),
  (notesCurrentlyRatedHelpful, pd.Int64Dtype()),
  (notesCurrentlyRatedNotHelpful, pd.Int64Dtype()),
  (notesAwaitingMoreRatings, pd.Int64Dtype()),
  (enrollmentState, pd.Int64Dtype()),
  (successfulRatingNeededToEarnIn, pd.Int64Dtype()),
  (authorTopNotHelpfulTagValues, str),
  (timestampOfLastStateChange, np.double),
  (aboveHelpfulnessThresholdKey, np.float64),  # nullable bool.
  (isEmergingWriterKey, pd.BooleanDtype()),
  (aggregateRatingReceivedTotal, pd.Int64Dtype()),
  (timestampOfLastEarnOut, np.double),
  (groupRaterInterceptKey, np.double),
  (groupRaterFactor1Key, np.double),
  (modelingGroupKey, np.float64),
  (raterHelpfulnessReputationKey, np.double),
  (numberOfTimesEarnedOutKey, np.float64),
  (expansionRaterInterceptKey, np.double),
  (expansionRaterFactor1Key, np.double),
  (expansionPlusRaterInterceptKey, np.double),
  (expansionPlusRaterFactor1Key, np.double),
  (multiGroupRaterInterceptKey, np.double),
  (multiGroupRaterFactor1Key, np.double),
  (modelingMultiGroupKey, np.float64),
  (coreWithTopicsRaterInterceptKey, np.double),
  (coreWithTopicsRaterFactor1Key, np.double),
  (coreFirstRoundRaterInterceptKey, np.double),
  (coreFirstRoundRaterFactor1Key, np.double),
]
raterModelOutputTSVColumns = [col for (col, dtype) in raterModelOutputTSVColumnsAndTypes]
raterModelOutputTSVTypeMapping = {col: dtype for (col, dtype) in raterModelOutputTSVColumnsAndTypes}

noteStatusChangesPrev = "_prev"
noteStatusChangesDerivedColumnsAndTypes = [
  (noteIdKey, np.int64),
  (noteFinalStatusChange, str),
  (noteNewRatings, np.int64),
  (noteDecidedByChange, str),
  (noteAllAddedRules, str),
  (noteAllRemovedRules, str),
  (noteDecidedByInterceptChange, str),
]
noteStatusChangesRemovedCols = [
  col
  for col in noteModelOutputTSVColumns
  if (noteInterceptMinSubstring in col)
  or (noteInterceptMaxSubstring in col)
  or (noHighVolSubstring in col)
  or (noCorrelatedSubstring in col)
]
noteStatusChangesModelOutputColumnsAndTypes = [
  (col, t)
  for (col, t) in noteModelOutputTSVColumnsAndTypes
  if col not in noteStatusChangesRemovedCols + [noteIdKey]
]
noteStatusChangesModelOutputWithPreviousColumnsAndTypes = (
  noteStatusChangesModelOutputColumnsAndTypes
  + [(col + noteStatusChangesPrev, t) for (col, t) in noteStatusChangesModelOutputColumnsAndTypes]
)

noteStatusChangeTSVColumnsAndTypes = noteStatusChangesDerivedColumnsAndTypes + sorted(
  noteStatusChangesModelOutputWithPreviousColumnsAndTypes, key=lambda tup: tup[0]
)
noteStatusChangesTSVColumns = [col for (col, dtype) in noteStatusChangeTSVColumnsAndTypes]
noteStatusChangesTSVTypeMapping = {
  col: dtype for (col, dtype) in noteStatusChangeTSVColumnsAndTypes
}

datasetKeyKey = "datasetKey"
partitionToReadKey = "partitionToRead"
fileNameToReadKey = "fileNameToRead"
inputPathsTSVColumnsAndTypes = [
  (datasetKeyKey, str),
  (partitionToReadKey, str),
  (fileNameToReadKey, str),
]
inputPathsTSVColumns = [col for (col, _) in inputPathsTSVColumnsAndTypes]
inputPathsTSVTypeMapping = {col: dtype for (col, dtype) in inputPathsTSVColumnsAndTypes}


@contextmanager
def time_block(label):
  start = time.time()
  try:
    yield
  finally:
    end = time.time()
    logger.info(f"{label} elapsed time: {end - start:.2f} secs ({((end - start) / 60.0):.2f} mins)")


### TODO: weave through second round intercept.
@dataclass
class ReputationGlobalIntercept:
  firstRound: float
  secondRound: float
  finalRound: float


@dataclass
class PrescoringMetaScorerOutput:
  globalIntercept: Optional[float]
  lowDiligenceGlobalIntercept: Optional[ReputationGlobalIntercept]
  tagFilteringThresholds: Optional[Dict[str, float]]  # tag => threshold
  finalRoundNumRatings: Optional[int]
  finalRoundNumNotes: Optional[int]
  finalRoundNumUsers: Optional[int]


@dataclass
class PrescoringMetaOutput:
  metaScorerOutput: Dict[str, PrescoringMetaScorerOutput]  # scorerName => output


@dataclass
class SharedMemoryDataframeInfo:
  sharedMemoryName: str
  dataSize: int


@dataclass
class ScoringArgsSharedMemory:
  noteTopics: SharedMemoryDataframeInfo
  ratings: SharedMemoryDataframeInfo
  noteStatusHistory: SharedMemoryDataframeInfo
  userEnrollment: SharedMemoryDataframeInfo


@dataclass
class PrescoringArgsSharedMemory(ScoringArgsSharedMemory):
  pass


@dataclass
class FinalScoringArgsSharedMemory(ScoringArgsSharedMemory):
  prescoringNoteModelOutput: SharedMemoryDataframeInfo
  prescoringRaterModelOutput: SharedMemoryDataframeInfo


@dataclass
class ScoringArgs:
  noteTopics: pd.DataFrame
  ratings: pd.DataFrame
  noteStatusHistory: pd.DataFrame
  userEnrollment: pd.DataFrame

  def remove_large_args_for_multiprocessing(self):
    self.noteTopics = None
    self.ratings = None
    self.noteStatusHistory = None
    self.userEnrollment = None


@dataclass
class PrescoringArgs(ScoringArgs):
  pass


@dataclass
class FinalScoringArgs(ScoringArgs):
  prescoringNoteModelOutput: pd.DataFrame
  prescoringRaterModelOutput: pd.DataFrame
  prescoringMetaOutput: PrescoringMetaOutput
  empiricalTotals: Optional[pd.DataFrame]

  def remove_large_args_for_multiprocessing(self):
    self.ratings = None
    self.noteStatusHistory = None
    self.userEnrollment = None
    self.prescoringNoteModelOutput = None
    self.prescoringRaterModelOutput = None


@dataclass
class ModelResult:
  scoredNotes: pd.DataFrame
  helpfulnessScores: pd.DataFrame
  auxiliaryNoteInfo: pd.DataFrame
  scorerName: Optional[str]
  metaScores: Optional[PrescoringMetaScorerOutput]


class RescoringRuleID(Enum):
  ALL_NOTES = 1
  NOTES_WITH_NEW_RATINGS = 2
  NOTES_FLIPPED_PREVIOUS_RUN = 3
  NEW_NOTES_NOT_RESCORED_RECENTLY_ENOUGH = 4
  RECENTLY_FLIPPED_NOTES_NOT_RESCORED_RECENTLY_ENOUGH = 5
  NMR_DUE_TO_MIN_STABLE_CRH_TIME = 6
  NOTES_CREATED_SOMEWHAT_RECENTLY = 7
  LOCKING_ELIGIBLE_RECENT_UNLOCKED_NOTES = 8


@dataclass
class NoteSubset:
  noteSet: Optional[set]
  maxNewCrhChurnRate: float
  maxOldCrhChurnRate: float
  description: RescoringRuleID
