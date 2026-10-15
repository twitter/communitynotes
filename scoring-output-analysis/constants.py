"""Constants for cross-model agreement analysis.

References column names and status values from the Community Notes scoring
algorithm (scoring/src/scoring/constants.py and scoring_rules.py).

Supports two data sources:
  1. scored_notes.tsv (full output from running the scorer locally)
  2. noteStatusHistory-00000.tsv (publicly downloadable, no scorer needed)
"""

# Note status values (scoring/src/scoring/constants.py L399-401)
CURRENTLY_RATED_HELPFUL = "CURRENTLY_RATED_HELPFUL"
CURRENTLY_RATED_NOT_HELPFUL = "CURRENTLY_RATED_NOT_HELPFUL"
NEEDS_MORE_RATINGS = "NEEDS_MORE_RATINGS"
# FIRM_REJECT is set by individual scorers to block downstream models from
# overriding a decision, but is never the final user-visible status.
FIRM_REJECT = "FIRM_REJECT"

ALL_STATUSES = [CURRENTLY_RATED_HELPFUL, CURRENTLY_RATED_NOT_HELPFUL, NEEDS_MORE_RATINGS, FIRM_REJECT]

# Note identity
noteIdKey = "noteId"
createdAtMillisKey = "createdAtMillis"

# --- noteStatusHistory columns (publicly available) ---
currentStatusKey = "currentStatus"
currentDecidedByKey = "currentDecidedBy"
currentCoreStatusKey = "currentCoreStatus"
currentExpansionStatusKey = "currentExpansionStatus"
currentGroupStatusKey = "currentGroupStatus"
currentMultiGroupStatusKey = "currentMultiGroupStatus"
currentModelingGroupKey = "currentModelingGroup"
currentModelingMultiGroupKey = "currentModelingMultiGroup"
firstNonNMRStatusKey = "firstNonNMRStatus"
mostRecentNonNMRStatusKey = "mostRecentNonNMRStatus"
timestampMillisOfFirstNonNMRStatusKey = "timestampMillisOfFirstNonNMRStatus"
timestampMillisOfCurrentStatusKey = "timestampMillisOfCurrentStatus"
timestampMillisOfStatusLockKey = "timestampMillisOfStatusLock"
lockedStatusKey = "lockedStatus"
timestampMillisOfMostRecentStatusChangeKey = "timestampMillisOfMostRecentStatusChange"

# Models available in noteStatusHistory (publicly downloadable)
NSH_MODELS = {
    "Core": currentCoreStatusKey,
    "Expansion": currentExpansionStatusKey,
    "Group": currentGroupStatusKey,
    "MultiGroup": currentMultiGroupStatusKey,
}

# --- scored_notes.tsv columns (requires running scorer) ---
# Meta scoring columns (scoring/src/scoring/constants.py L222-226)
finalRatingStatusKey = "finalRatingStatus"
metaScorerActiveRulesKey = "metaScorerActiveRules"
decidedByKey = "decidedBy"

# Per-model output column keys for rating status
coreRatingStatusKey = "coreRatingStatus"
coreWithTopicsRatingStatusKey = "coreWithTopicsRatingStatus"
expansionRatingStatusKey = "expansionRatingStatus"
expansionPlusRatingStatusKey = "expansionPlusRatingStatus"
groupRatingStatusKey = "groupRatingStatus"
multiGroupRatingStatusKey = "multiGroupRatingStatus"
topicRatingStatusKey = "topicRatingStatus"
gaussianRatingStatusKey = "gaussianRatingStatus"

# Per-model intercept keys
coreNoteInterceptKey = "coreNoteIntercept"
coreWithTopicsNoteInterceptKey = "coreWithTopicsNoteIntercept"
expansionNoteInterceptKey = "expansionNoteIntercept"
expansionPlusNoteInterceptKey = "expansionPlusNoteIntercept"
groupNoteInterceptKey = "groupNoteIntercept"
multiGroupNoteInterceptKey = "multiGroupNoteIntercept"
topicNoteInterceptKey = "topicNoteIntercept"
gaussianNoteInterceptKey = "gaussianNoteIntercept"

# Per-model factor keys
coreNoteFactor1Key = "coreNoteFactor1"
coreWithTopicsNoteFactor1Key = "coreWithTopicsNoteFactor1"
expansionNoteFactor1Key = "expansionNoteFactor1"
expansionPlusNoteFactor1Key = "expansionPlusNoteFactor1"
groupNoteFactor1Key = "groupNoteFactor1"
multiGroupNoteFactor1Key = "multiGroupNoteFactor1"
topicNoteFactor1Key = "topicNoteFactor1"
gaussianNoteFactor1Key = "gaussianNoteFactor1"

# Full models dict for scored_notes.tsv: name -> (status_col, intercept_col, factor_col)
SCORED_NOTES_MODELS = {
    "Core": (coreRatingStatusKey, coreNoteInterceptKey, coreNoteFactor1Key),
    "CoreWithTopics": (
        coreWithTopicsRatingStatusKey,
        coreWithTopicsNoteInterceptKey,
        coreWithTopicsNoteFactor1Key,
    ),
    "Expansion": (expansionRatingStatusKey, expansionNoteInterceptKey, expansionNoteFactor1Key),
    "ExpansionPlus": (
        expansionPlusRatingStatusKey,
        expansionPlusNoteInterceptKey,
        expansionPlusNoteFactor1Key,
    ),
    "Group": (groupRatingStatusKey, groupNoteInterceptKey, groupNoteFactor1Key),
    "MultiGroup": (multiGroupRatingStatusKey, multiGroupNoteInterceptKey, multiGroupNoteFactor1Key),
    "Topic": (topicRatingStatusKey, topicNoteInterceptKey, topicNoteFactor1Key),
    "Gaussian": (gaussianRatingStatusKey, gaussianNoteInterceptKey, gaussianNoteFactor1Key),
}

# decidedBy strings from scoring_rules.py RuleID enum (L48-78)
DECIDED_BY_CORE = "CoreModel (v1.1)"
DECIDED_BY_CORE_WITH_TOPICS = "CoreWithTopicsModel (v1.1)"
DECIDED_BY_EXPANSION = "ExpansionModel (v1.1)"
DECIDED_BY_EXPANSION_PLUS = "ExpansionPlusModel (v1.1)"
DECIDED_BY_COVERAGE = "CoverageModel (v1.1)"
DECIDED_BY_GAUSSIAN = "GaussianModel (v1.0)"
DECIDED_BY_INSUFFICIENT_EXPLANATION = "InsufficientExplanation (v1.0)"
DECIDED_BY_SCORING_DRIFT_GUARD = "ScoringDriftGuard (v1.0)"
DECIDED_BY_NMR_DUE_TO_MIN_STABLE_CRH = "NmrDueToMinStableCrhTime (v1.0)"

# Group model decidedBy strings follow pattern "GroupModelNN (v1.1)"
GROUP_MODEL_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 33]
DECIDED_BY_GROUP_MODELS = {
    gid: f"GroupModel{gid:02d} (v1.1)" for gid in GROUP_MODEL_IDS
}

# Topic model decidedBy strings follow pattern "TopicModelNN (v1.0)"
TOPIC_MODEL_IDS = [1, 2, 3, 4]
DECIDED_BY_TOPIC_MODELS = {
    tid: f"TopicModel{tid:02d} (v1.0)" for tid in TOPIC_MODEL_IDS
}

# Multi-group model decidedBy
DECIDED_BY_MULTI_GROUP_MODELS = {1: "MultiGroupModel01 (v1.0)"}

# CRH intercept threshold (from documentation/under-the-hood/ranking-notes.md)
CRH_INTERCEPT_THRESHOLD = 0.40
CRNH_BASE_INTERCEPT_THRESHOLD = -0.05
CRNH_FACTOR_COEFFICIENT = 0.8
